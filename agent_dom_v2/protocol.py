"""Talking to the model: the one paced call, and the tool-calling protocol around it.

chat() is the ONLY place in the package that reaches the endpoint, and it goes through
eval.rate_limit.call_with_retry: the pacing is calibrated on the endpoint's real TPM, and
a call that bypassed it would break the pacing for every call after it. A grep for
`completions.create` over the package must find this one site.

The rest is the protocol instrumentation both tiers share: which of the four exclusive
buckets an assistant turn falls in (_classify_protocol), how a tool call's arguments are
parsed and how a malformed or unknown one is recorded and answered, and the last forced
call on a finish-tool. Everything here records; nothing repairs — a tool call emitted as
prose is counted and the model is asked to re-emit it, never parsed out of the content.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema

from agent_dom import _add_usage
from eval.rate_limit import call_with_retry

from . import settings

logger = logging.getLogger(__name__)

_RETRY_PROTOCOL_MSG = (
    "Your last message put a tool call in the message content instead of using the "
    "tool-calling protocol. Content is discarded. Re-emit it as a real tool call."
)


class _ToolSchema(GenerateJsonSchema):
    """JSON Schema as the v2.0 literals wrote it, so the models replace them without
    changing what the endpoint reads: no title, no class docstring as description (the
    docstrings are for the reader of the code, the Field descriptions for the model),
    nullable as `"type": [X, "null"]` rather than anyOf, nested models inlined rather than
    $ref'd, no default, keys in construction order. Parity with the frozen literals is a
    test (scratch/refactor/test_tool_schema_parity.py), dict-equal."""

    def field_title_should_be_set(self, schema) -> bool:
        return False

    def model_schema(self, schema):
        out = super().model_schema(schema)
        out.pop("description", None)
        return out

    def nullable_schema(self, schema):
        inner = self.generate_inner(schema["schema"])
        if not inner.get("type"):
            return super().nullable_schema(schema)
        return {**inner, "type": [inner["type"], "null"]}

    def sort(self, value, parent_key=None):
        return value

    def generate(self, schema, mode="validation"):
        out = super().generate(schema, mode=mode)
        return _inline(out, out.pop("$defs", {}))


def _inline(node: Any, defs: dict[str, Any]) -> Any:
    """Resolve $ref onto its definition and drop title/default, recursively."""
    if isinstance(node, dict):
        if "$ref" in node:
            node = {**defs[node["$ref"].rsplit("/", 1)[1]],
                    **{k: v for k, v in node.items() if k != "$ref"}}
        return {k: _inline(v, defs) for k, v in node.items() if k not in ("title", "default")}
    if isinstance(node, list):
        return [_inline(v, defs) for v in node]
    return node


def tool_schema(model: type[BaseModel], name: str, description: str) -> dict[str, Any]:
    """An OpenAI function-calling tool definition whose parameters are `model`'s schema.

    The model is the schema ANNOUNCED to the endpoint, and only that: nothing here, and
    nothing in the loops, validates a response against it — see report._normalize_report
    for how a report is read back, and why.

    Args:
        model: The Pydantic model declaring the arguments.
        name: The tool name as the model calls it.
        description: The tool description shown to the model.

    Returns:
        {"type": "function", "function": {"name", "description", "parameters"}}.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": model.model_json_schema(schema_generator=_ToolSchema),
        },
    }


def chat(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
         tool_choice: Any = None) -> Any:
    """One chat completion, paced and retried by call_with_retry.

    The client is read from settings at call time, not bound at import, so a test that
    swaps `agent_dom_v2.settings.client` — or this module's `call_with_retry` — reaches
    every call of both tiers.

    Args:
        model: Chat model id of the calling tier.
        messages: The conversation so far.
        tools: Tool definitions exposed on this call.
        tool_choice: The SDK's tool_choice, when a finish-tool is forced; None otherwise.

    Returns:
        The SDK response.
    """
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "tools": tools}
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    return call_with_retry(settings.client.chat.completions.create, **kwargs)


def _looks_like_tool_call_in_content(content: str | None,
                                     tool_names: tuple[str, ...]) -> bool:
    """True when `content` names one of this tier's tools and opens a brace after it."""
    if not content:
        return False
    return any(name in content and content.find("{", content.find(name)) != -1
               for name in tool_names)


def _classify_protocol(message: Any, tool_names: tuple[str, ...],
                       metrics: dict[str, Any]) -> str:
    """Bucket one assistant turn, exactly as the single-phase loop does.

    The four buckets are exclusive and counted before anything is dispatched, so the
    protocol numbers of a v2 run stay comparable with a v1 run: both tiers feed the same
    root counters.

    Args:
        message: The assistant message of the response.
        tool_names: This tier's tool names, for the prose detection.
        metrics: Shared metrics dict, mutated in place.

    Returns:
        "dispatch" to serve the tool calls, "retry" to re-ask for the protocol, "stop" when
        the model produced neither a call nor an answer.
    """
    has_tool_calls = bool(message.tool_calls)
    in_content = _looks_like_tool_call_in_content(message.content, tool_names)

    if has_tool_calls and in_content:
        metrics["both_channels"] += 1
        return "dispatch"
    if has_tool_calls:
        metrics["tool_calls_ok"] += 1
        return "dispatch"
    if in_content:
        metrics["tool_call_in_content"] += 1
        return "retry"
    metrics["no_tool_no_answer"] += 1
    return "stop"


def parse_tool_args(tool_call: Any, tier: str, step: int, round_index: int,
                    metrics: dict[str, Any], seq_entry: dict[str, Any],
                    messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Parse one tool call's arguments; on malformed JSON, record, answer and return None.

    The whole malformed_json branch lives here so both tiers count and record it the same
    way: the counter, the invalid_tool_args entry, the raw string under `args.raw` in the
    tool_sequence entry (a dict in every entry, so both lists are read the same way
    downstream), the warning, and the tool result the call must still receive — leaving
    one unanswered makes the next request 400.

    Args:
        tool_call: The SDK tool call object.
        tier: "orchestrator" or "subagent", stamped on the records.
        step: The tier's own step (the round, for the orchestrator).
        round_index: Orchestrator round the call belongs to.
        metrics: Shared metrics dict, mutated in place.
        seq_entry: This call's tool_sequence entry; `args` is filled in here.
        messages: The conversation, appended with the tool result on failure.

    Returns:
        The parsed arguments, a dict, or None when they did not parse into one.
    """
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
        if not isinstance(args, dict):
            raise json.JSONDecodeError("not an object", "", 0)
    except (json.JSONDecodeError, TypeError):
        metrics["tool_call_malformed"] += 1
        metrics["invalid_tool_args"].append(
            {"tool": name, "args": {"raw": tool_call.function.arguments},
             "reason": "malformed_json", "step": step, "tier": tier, "round": round_index})
        seq_entry["args"] = {"raw": tool_call.function.arguments}
        logger.warning("tool_call_malformed at %s step %d (%s): %s", tier, step, name,
                       tool_call.function.arguments)
        messages.append({"role": "tool", "tool_call_id": tool_call.id,
                         "content": "arguments were not valid JSON; retry the call"})
        return None
    seq_entry["args"] = args
    return args


def refuse_unknown_tool(tool_call: Any, args: dict[str, Any], tier: str, step: int,
                        round_index: int, metrics: dict[str, Any],
                        messages: list[dict[str, Any]]) -> None:
    """Record a call on a tool this tier does not expose, and answer it.

    A hallucinated tool is a measurement, not a crash: counted as malformed, recorded with
    its arguments whole, and given a tool result so the conversation stays valid.

    Args:
        tool_call: The SDK tool call object.
        args: Its parsed arguments.
        tier: "orchestrator" or "subagent", stamped on the record.
        step: The tier's own step (the round, for the orchestrator).
        round_index: Orchestrator round the call belongs to.
        metrics: Shared metrics dict, mutated in place.
        messages: The conversation, appended with the tool result.
    """
    name = tool_call.function.name
    metrics["tool_call_malformed"] += 1
    metrics["invalid_tool_args"].append(
        {"tool": name, "args": args, "reason": "unknown_tool", "step": step, "tier": tier,
         "round": round_index})
    logger.warning("unknown tool %r at %s step %d", name, tier, step)
    messages.append({"role": "tool", "tool_call_id": tool_call.id,
                     "content": f"unknown tool {name!r}"})


def force_finish_tool(messages: list[dict[str, Any]], tool_def: dict[str, Any],
                      tool_name: str, model: str, metrics: dict[str, Any],
                      invocation: dict[str, Any] | None = None) -> Any | None:
    """Last forced call on a finish-tool; the raw arguments, or None if it did not comply.

    Shared by both tiers: the subagent forces `report`, the orchestrator forces
    `submit_draft`, and the call is the same — only the finish-tool exposed (1-tool list) on
    top of the named forcing. If the shim rejects the named form, replace tool_choice by
    "required". Goes through call_with_retry like every other call: a 429 here must raise
    and let the runner drop the question as a replayable loss, never be swallowed into a
    row that the resume filter would then freeze as done.

    Counted in n_steps: every LLM call of both tiers is, forced ones included (agent_dom's
    own forced call is not, because in v1 the step budget belonged to the free loop alone).

    Args:
        messages: Conversation so far (must not end on an unanswered tool call).
        tool_def: The finish-tool definition.
        tool_name: Its name, for the forcing and for picking the call out of the answer.
        model: Chat model id.
        metrics: Shared metrics dict, mutated in place.
        invocation: A second usage bucket ({"tokens": ...}) to accumulate into as well —
            the subagent's stats entry; None for the orchestrator.

    Returns:
        The arguments exactly as json.loads produced them — the caller normalizes — or
        None when the model did not call the tool or its arguments did not parse.
    """
    metrics["n_steps"] += 1
    response = chat(model, messages, [tool_def],
                    tool_choice={"type": "function", "function": {"name": tool_name}})
    _add_usage(response, metrics)
    if invocation is not None:
        _add_usage(response, invocation)

    for tool_call in (response.choices[0].message.tool_calls or []):
        if tool_call.function.name != tool_name:
            continue
        try:
            return json.loads(tool_call.function.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            return None
    return None
