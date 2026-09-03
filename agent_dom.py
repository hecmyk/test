"""DOM agent — single-phase extraction driven by the XML skeleton.

run_dom_agent gives the model the skeleton produced by dom_render.build_outline_xml
instead of the full document, and lets it pull what it needs through read tools —
dom_tools.render_section to read a section, dom_tools.build_search_index to locate one by
keyword, and, when the caller supplies a pdf_path, dom_visual.get_visual to look at the
source image of one table or figure — before ending on the submit_draft finish-tool.

Deliberately WITHOUT the text-salvage layer of the prototype agent (agent.py): a tool
call emitted as prose in `content` is counted and reported, never repaired. The point of
this module is to measure the raw tool-calling behaviour of the internal models, so the
metrics must not be polished by a recovery path. Nothing is imported from agent.py: its
helpers differ in signature or in output shape, and its image path is entangled with that
salvage layer — dom_visual carries the crop logic instead, so the fitz/cv2 dependency
enters only through this module and only on a run that asked for the visual channel.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

# Requires PyYAML (external runtime dependency, not declared in this repo).
import yaml
from openai import OpenAI

from config import MISTRAL_AGENT_MODEL, MISTRAL_API_KEY, MISTRAL_BASE_URL
from dom_tools import (DEFAULT_TOP_K, DRAFT_SCHEMA, GET_SECTION, QA_SCHEMA, SEARCH,
                       SUBMIT_DRAFT, all_section_ids, build_get_section_tool,
                       build_search_index, build_search_tool, build_submit_draft_tool,
                       render_section)
from dom_visual import GET_VISUAL, build_get_visual_tool, get_visual

logger = logging.getLogger(__name__)

client = OpenAI(
    base_url=MISTRAL_BASE_URL,
    api_key=MISTRAL_API_KEY,
)
MODEL = MISTRAL_AGENT_MODEL

with open(Path(__file__).parent / "prompts_dom.yaml", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

# (system prompt key, user template key, submit_draft field schema) per mode. The schema is
# shared by the tool definition and the normalizer, so the two cannot drift.
_MODES = {
    "kyc": ("dom_system", "dom_user_template", DRAFT_SCHEMA),
    "qa": ("qa_system", "qa_user_template", QA_SCHEMA),
}

# Detection only, never parsing: a content that names a tool AND opens a brace after it.
# Every exposed tool must be listed, otherwise one emitted as prose is not counted as
# tool_call_in_content and falls through to no_tool_no_answer, which would show up as a
# protocol regression the model never made.
_TOOL_NAMES = (GET_SECTION, SEARCH, GET_VISUAL, SUBMIT_DRAFT)

_RETRY_PROTOCOL_MSG = (
    "Your last message put a tool call in the message content instead of using the "
    "tool-calling protocol. Content is discarded. Re-emit it as a real tool call."
)


# --------------------------------------------------------------------------- #
# Loop helpers                                                                 #
# --------------------------------------------------------------------------- #
def _add_usage(response: Any, metrics: dict[str, Any]) -> None:
    """Accumulate an LLM call's token usage; usage may be missing on this endpoint."""
    usage = getattr(response, "usage", None)
    bucket = metrics["tokens"]
    bucket["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
    bucket["completion"] += getattr(usage, "completion_tokens", 0) or 0
    bucket["total"] += getattr(usage, "total_tokens", 0) or 0


def _looks_like_tool_call_in_content(content: str | None) -> bool:
    """True when `content` names a tool and opens a brace after it (detection only)."""
    if not content:
        return False
    return any(name in content and content.find("{", content.find(name)) != -1
               for name in _TOOL_NAMES)


def _normalize_draft(fields: Any, field_definitions: dict[str, str],
                     schema: dict[str, Any]) -> dict[str, Any]:
    """Coerce raw submit_draft arguments into the canonical draft shape.

    The tool schema is advisory on this endpoint, so a missing field becomes an entry with
    every schema key at None, and fields outside `field_definitions` are dropped. Values
    are kept as the model sent them — no str() coercion, so a list of source_pages stays
    a list.

    Args:
        fields: The raw `fields` object from the tool call.
        field_definitions: Mapping from field name to its definition text.
        schema: The per-field property schema advertised to the model; its keys are the
            ones read back.

    Returns:
        Draft mapping each expected field to one entry per schema key.
    """
    fields = fields if isinstance(fields, dict) else {}
    draft = {}
    for name in field_definitions:
        entry = fields.get(name)
        entry = entry if isinstance(entry, dict) else {}
        draft[name] = {key: entry.get(key) for key in schema}
    return draft


def _init_metrics() -> dict[str, Any]:
    """Pre-initialize every metric key; insertion order is load-bearing for the run log."""
    return {
        "n_steps": 0,
        "n_get_section_calls": 0,
        "n_unique_sections_fetched": 0,
        "tokens": {"prompt": 0, "completion": 0, "total": 0},
        "hit_max_iterations": False,
        "submit_draft_source": "none",
        "tool_calls_ok": 0,
        "tool_call_in_content": 0,
        "tool_call_malformed": 0,
        "both_channels": 0,
        "no_tool_no_answer": 0,
        "duration_s": 0.0,
        # Appended LAST on purpose: every pre-existing key keeps its position in the run
        # log. Trace of what the agent asked for, in call order, duplicates kept — a call
        # with no section_id records None, which is the signal you want when debugging.
        "get_section_calls": [],
        # Also appended last, same reason. One entry per tool call the loop refused, as
        # {"tool", "args", "reason", "step"}: `tool` and `reason` are what you group on,
        # `args` is kept whole so a future tool (get_table, get_figure, ...) needs no new
        # key. The refusal is recorded, never repaired.
        "invalid_tool_args": [],
        # Appended after invalid_tool_args so no pre-existing key moves. Unlike
        # n_get_section_calls, which counts every attempt, these two count SERVED calls
        # only: a refused search is in invalid_tool_args and nowhere else.
        "n_search_calls": 0,
        "search_calls": [],
        # Appended after search_calls, same key-order discipline. The two counters are NOT
        # redundant: get_visual_calls records every attempt in call order (refusals and
        # repeats included, like get_section_calls), n_get_visual_calls counts the crops
        # actually SENT to the model. Their gap is the wasted-call rate, and
        # invalid_tool_args splits it between refusals and repeats.
        "n_get_visual_calls": 0,
        "get_visual_calls": [],
        "n_unique_visuals_fetched": 0,
        # Appended after n_unique_visuals_fetched, same key-order discipline. Chronological
        # order of the tool calls SERVED through the tool-calling protocol, one entry per
        # call as {"step", "tool", "args"} — what tells three parallel calls in one step
        # from three sequential steps, which the per-tool traces cannot. The entry is
        # appended before the argument parsing, so submit_draft, unknown_tool and
        # malformed_json are all in it; `args` is filled in after, whole, and holds the raw
        # string under "raw" when the JSON did not parse — same shape as
        # invalid_tool_args, so both are read the same way downstream. A served search
        # also carries "hits", one {"section_id", "n_matches", "score"} per result in rank
        # order, so a later get_section can be tested against what search actually offered
        # and against how strong the hit was. Two
        # exclusions: tool calls emitted as prose (the tool_call_in_content branch never
        # reaches this loop; they are counted there), and the forced call of
        # _force_submit_draft (meta, outside the agent's free sequence). This key is the
        # source of truth for order — the three per-tool lists are the same information
        # projected per tool.
        "tool_sequence": [],
    }


def _check_pdf_matches_dom(dom: dict, pdf_path: str | Path) -> None:
    """Warn when the PDF handed in is not the one the dom was built from.

    A mismatch is silent corruption of the worst kind: every id resolves, every crop comes
    back, and every one of them shows the wrong document. Only warned about — build_dom
    records pdf_name as optional, so an absent name is not an error.
    """
    expected = (dom["header"].get("source") or {}).get("pdf")
    if expected and Path(pdf_path).name != expected:
        logger.warning("pdf %r is not the dom's source pdf %r: crops will not match the "
                       "atoms", Path(pdf_path).name, expected)


def _serve_visual(atom_id: Any, dom: dict, pdf_path: str | Path | None,
                  sent: dict[str, int], step: int, metrics: dict[str, Any],
                  image_parts: list[dict[str, Any]]) -> str:
    """Serve one get_visual call: record it, queue its crop, return the tool-result text.

    The crop is not returned here. It is appended to `image_parts`, which the loop emits as
    a single user message after the step's tool results — the only channel this endpoint is
    known to accept an image on (see dom_visual's module docstring).

    Args:
        atom_id: The raw `atom_id` argument, exactly as the model sent it.
        dom: build_dom output.
        pdf_path: Source PDF, or None when the run was started without one.
        sent: atom_id -> step the crop was first served at; mutated.
        step: Current loop step.
        metrics: Shared metrics dict, mutated in place.
        image_parts: Content blocks accumulated for this step, mutated.

    Returns:
        The text to put in the tool result.
    """
    metrics["get_visual_calls"].append(atom_id)

    if pdf_path is None:
        # The tool was not exposed this run; the model named it anyway. Answered rather
        # than raised, and recorded — a hallucinated tool is a measurement, not a crash.
        metrics["invalid_tool_args"].append(
            {"tool": GET_VISUAL, "args": {"atom_id": atom_id}, "reason": "no_pdf",
             "step": step})
        return "get_visual is not available in this run; use get_section instead."

    if atom_id in sent:
        return (f"The image of {atom_id} was already provided at step {sent[atom_id]}; "
                "look at it again in the conversation above.")

    result = get_visual(atom_id, dom, pdf_path)
    if "error" in result:
        metrics["invalid_tool_args"].append(
            {"tool": GET_VISUAL, "args": {"atom_id": atom_id}, "reason": result["error"],
             "step": step})
        return json.dumps(result, ensure_ascii=False)

    sent[atom_id] = step
    metrics["n_get_visual_calls"] += 1
    label, page = result["label"], result["page"]
    image_parts.append({"type": "text", "text": f"Image of {atom_id} ({label}, page {page}):"})
    image_parts.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{result['image_b64']}"}})

    head = f"{label} {atom_id}, page {page}. The image is attached in the next message."
    if not result["text"]:
        return f"{head} OCR produced no text for this atom — read it from the image."
    return f"{head}\nOCR text on file for this atom:\n{result['text']}"


def _force_submit_draft(messages: list[dict[str, Any]], submit_tool: dict[str, Any],
                        field_definitions: dict[str, str], schema: dict[str, Any],
                        model: str, metrics: dict[str, Any]) -> dict[str, Any] | None:
    """Last forced call on the finish-tool, or None if the model still does not comply.

    Args:
        messages: Conversation so far (must not end on an unanswered tool call).
        submit_tool: The finish-tool definition.
        field_definitions: Mapping from field name to its definition text.
        schema: The per-field property schema advertised to the model.
        model: Chat model id.
        metrics: Shared metrics dict, mutated in place.

    Returns:
        The normalized draft, or None.
    """
    # Endpoint robustness: only the finish-tool is exposed (1-tool list) on top of the
    # named forcing. If the shim rejects the named form, replace tool_choice by "required".
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[submit_tool],
        tool_choice={"type": "function", "function": {"name": SUBMIT_DRAFT}},
    )
    _add_usage(response, metrics)
    message = response.choices[0].message
    for tool_call in (message.tool_calls or []):
        if tool_call.function.name != SUBMIT_DRAFT:
            continue
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            return None
        return _normalize_draft(args.get("fields"), field_definitions, schema)
    return None


# --------------------------------------------------------------------------- #
# run_dom_agent                                                                #
# --------------------------------------------------------------------------- #
def run_dom_agent(dom: dict, xml_skeleton: str, field_definitions: dict[str, str],
                  max_steps: int = 10, model: str = MODEL, mode: str = "kyc",
                  question: str = "",
                  pdf_path: str | Path | None = None,
                  prompts: dict[str, str] | None = None) -> tuple[dict, dict]:
    """Extract structured fields by navigating the document skeleton.

    Single phase: the model reads the skeleton, locates sections with search and pulls
    them with get_section, then ends on the finish-tool. No text-salvage — a tool call
    emitted as prose is counted and the model is asked to re-emit it, never parsed.

    With `pdf_path`, get_visual is exposed as well and the run becomes multimodal: the
    model can pull the source crop of a table or a figure. Without it the run is exactly
    the text-only one it was before — the dom stores the PDF's NAME, never a usable path,
    so the visual channel cannot be opened from the dom alone.

    Args:
        dom: build_dom output ({header, index, chrome, tree}).
        xml_skeleton: build_outline_xml output for the same dom.
        field_definitions: Mapping from field name to its definition text.
        max_steps: Maximum agent turns before forcing the finish-tool.
        model: Chat model id.
        mode: "kyc" (multi-field extraction, the default) or "qa" (single question). It
            selects the prompt pair AND the finish-tool field schema.
        question: The question. "qa" mode only.
        pdf_path: Path to the PDF this dom was built from. None (the default) leaves
            get_visual unexposed.
        prompts: Prompt bank to read `mode`'s system and user templates from. None (the
            default) uses the one loaded from prompts_dom.yaml at import; a benchmark
            with its own wording passes its own parsed YAML instead.

    Returns:
        `(draft, metrics)`: `draft` maps each field to one entry per schema key of the
        mode, `metrics` holds step counts, token usage, tool-calling protocol stats and
        the ordered traces of the section ids, keywords and atom ids the agent asked for.
    """
    started = time.perf_counter()
    metrics = _init_metrics()

    system_key, user_key, schema = _MODES[mode]
    prompts = PROMPTS if prompts is None else prompts
    submit_tool = build_submit_draft_tool(field_definitions, schema)
    tools = [build_get_section_tool(), build_search_tool()]
    if pdf_path is not None:
        _check_pdf_matches_dom(dom, pdf_path)
        tools.append(build_get_visual_tool())
    tools.append(submit_tool)   # kept last, as the model reads the list in order
    valid_ids = all_section_ids(dom["tree"])
    # Built once per run and held like `fetched`: the BM25 corpus is derived from the dom,
    # and the index carries the cache that makes a repeated query free.
    search_index = build_search_index(dom)

    # Both templates get the full kwarg set: str.format ignores what a template does not
    # name, so neither mode needs its own formatting branch.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts[system_key]},
        {"role": "user", "content": prompts[user_key].format(
            skeleton=xml_skeleton,
            field_definitions=json.dumps(field_definitions, indent=2, ensure_ascii=False),
            question=question,
        )},
    ]

    fetched: dict[str, int] = {}   # section_id -> step it was first served at
    visuals_sent: dict[str, int] = {}   # atom_id -> step its crop was first served at
    draft: dict[str, Any] | None = None

    for step in range(max_steps):
        metrics["n_steps"] = step + 1
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tools,
        )
        _add_usage(response, metrics)
        message = response.choices[0].message

        # ── Tool-calling protocol instrumentation ──────────────────────────────
        has_tool_calls = bool(message.tool_calls)
        in_content = _looks_like_tool_call_in_content(message.content)

        if has_tool_calls and in_content:
            metrics["both_channels"] += 1
            logger.warning("both_channels at step %d: %s", step, (message.content or "")[:200])
        elif has_tool_calls:
            metrics["tool_calls_ok"] += 1
        elif in_content:
            metrics["tool_call_in_content"] += 1
            logger.warning("tool_call_in_content at step %d: %s", step, message.content[:200])
            # Plain assistant/user alternation: no orphan tool_calls, and the turn is not
            # left dangling on an assistant message.
            messages.append({"role": "assistant", "content": message.content})
            messages.append({"role": "user", "content": _RETRY_PROTOCOL_MSG})
            continue
        else:
            metrics["no_tool_no_answer"] += 1
            logger.warning("no_tool_no_answer at step %d", step)
            break

        messages.append(message.model_dump(exclude_none=True))

        # Crops served this step, emitted as ONE user message once every tool call has its
        # result: an image cannot ride inside a role="tool" message on this endpoint.
        image_parts: list[dict[str, Any]] = []

        # Every tool call gets a tool result, even once the draft is in: leaving one
        # unanswered makes the next request 400 and corrupts the replayable trace.
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            seq_entry: dict[str, Any] = {"step": step, "tool": name}
            metrics["tool_sequence"].append(seq_entry)
            try:
                args = json.loads(tool_call.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise json.JSONDecodeError("not an object", "", 0)
            except (json.JSONDecodeError, TypeError):
                metrics["tool_call_malformed"] += 1
                # The arguments did not parse, so there is no dict to record: the raw
                # string is kept under "raw" to keep `args` a dict in every entry.
                metrics["invalid_tool_args"].append(
                    {"tool": name, "args": {"raw": tool_call.function.arguments},
                     "reason": "malformed_json", "step": step})
                seq_entry["args"] = {"raw": tool_call.function.arguments}
                logger.warning("tool_call_malformed at step %d (%s): %s", step, name,
                               tool_call.function.arguments)
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": "arguments were not valid JSON; retry the call"})
                continue

            seq_entry["args"] = args

            if name == GET_SECTION:
                metrics["n_get_section_calls"] += 1
                section_id = args.get("section_id")
                metrics["get_section_calls"].append(section_id)
                if section_id in fetched:
                    result = (f"Already provided at step {fetched[section_id]}; "
                              "re-read it in the conversation above.")
                else:
                    section = render_section(section_id, dom)
                    if isinstance(section, str):
                        fetched[section_id] = step
                        result = section
                    else:
                        metrics["invalid_tool_args"].append(
                            {"tool": GET_SECTION, "args": {"section_id": section_id},
                             "reason": section["error"], "step": step})
                        result = json.dumps(section, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": result})

            elif name == SEARCH:
                keyword = args.get("keyword")
                hits = search_index.search(keyword, args.get("top_k", DEFAULT_TOP_K))
                # An empty keyword is the only refusal: no match is a legitimate answer
                # and comes back as [], which is exactly the signal the model needs to
                # re-query with different words.
                if isinstance(hits, dict):
                    metrics["invalid_tool_args"].append(
                        {"tool": SEARCH, "args": args, "reason": hits["error"],
                         "step": step})
                else:
                    metrics["n_search_calls"] += 1
                    metrics["search_calls"].append(keyword)
                    # In rank order: the id the next get_section is checked against, plus
                    # the two numbers that say how strong the hit was — n_matches (how many
                    # atoms of the section matched) and score (the BM25 value, within-
                    # document only, never comparable across documents). snippet, page and
                    # kind are dropped: the snippets are the bulk of a hit and answer no
                    # question these three do not. Absent on a refused search — that call
                    # is in invalid_tool_args and returned nothing to follow.
                    seq_entry["hits"] = [
                        {"section_id": h["section_id"], "n_matches": h["n_matches"],
                         "score": h["score"]}
                        for h in hits]
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": json.dumps(hits, ensure_ascii=False)})

            elif name == GET_VISUAL:
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": _serve_visual(args.get("atom_id"), dom, pdf_path,
                                                          visuals_sent, step, metrics,
                                                          image_parts)})

            elif name == SUBMIT_DRAFT:
                draft = _normalize_draft(args.get("fields"), field_definitions, schema)
                metrics["submit_draft_source"] = "tool"
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": "draft received"})

            else:
                metrics["tool_call_malformed"] += 1
                metrics["invalid_tool_args"].append(
                    {"tool": name, "args": args, "reason": "unknown_tool", "step": step})
                logger.warning("unknown tool %r at step %d", name, step)
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": f"unknown tool {name!r}"})

        # After the tool results, never before: the assistant turn's tool calls must all be
        # answered before another role opens, or the next request 400s.
        if image_parts:
            messages.append({"role": "user", "content": image_parts})

        if draft is not None:
            break
    else:
        # Loop exhausted without breaking: the cap was genuinely hit. An early break on
        # no_tool_no_answer also lands in the forced call below, but is NOT a cap hit.
        metrics["hit_max_iterations"] = True

    if draft is None:
        draft = _force_submit_draft(messages, submit_tool, field_definitions, schema,
                                    model, metrics)
        if draft is not None:
            metrics["submit_draft_source"] = "forced"
        else:
            # No salvage by design: an empty draft is returned so the run's metrics
            # survive, instead of raising and losing the measurement.
            logger.error("%s never produced after %d steps and one forced call",
                         SUBMIT_DRAFT, metrics["n_steps"])
            draft = _normalize_draft({}, field_definitions, schema)

    # .get, not []: the qa schema has no source_section_id to validate.
    for name, entry in draft.items():
        source = entry.get("source_section_id")
        if source is not None and source not in valid_ids:
            logger.warning("field %r cites unknown section_id %r", name, source)

    metrics["n_unique_sections_fetched"] = len(fetched)
    metrics["n_unique_visuals_fetched"] = len(visuals_sent)
    metrics["duration_s"] = round(time.perf_counter() - started, 2)
    return draft, metrics
