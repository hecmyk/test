"""The subagent — one ephemeral read per delegation.

run_subagent pre-loads the sections into the message that opens the invocation, lets the
model look at visuals and report, and drops the whole context on return: only the report
goes up. It has no get_section, by construction. See the package docstring for why that
split is the point of v2.
"""

import logging
from pathlib import Path
from typing import Any

from agent_dom import _add_usage, _serve_visual
from dom_visual import GET_VISUAL, build_get_visual_tool

from .protocol import (_RETRY_PROTOCOL_MSG, _classify_protocol, chat, force_finish_tool,
                       parse_tool_args, refuse_unknown_tool)
from .report import REPORT, _descriptive_fallback, _normalize_report, build_report_tool
from .sections import build_subagent_context
from .settings import PROMPTS_V2, SUBAGENT_MAX_TOOL_CALLS, SUBAGENT_MODEL
from .verifications import Verdict, check_key_figures, check_page_ranges

logger = logging.getLogger(__name__)

# Detection only, never parsing — same contract as agent_dom._TOOL_NAMES, one tuple per
# tier (the orchestrator's is in orchestrator.py): a tool missing from its tier's tuple,
# when emitted as prose, is not counted as tool_call_in_content and falls through to
# no_tool_no_answer, which reads as a protocol regression the model never made.
# agent_dom's own helper closes over ITS tuple, so the detection is re-expressed here
# rather than imported.
_SUBAGENT_TOOL_NAMES = (GET_VISUAL, REPORT)


def _force_report(messages: list[dict[str, Any]], report_tool: dict[str, Any],
                  model: str, metrics: dict[str, Any],
                  invocation: dict[str, Any]) -> dict[str, Any] | None:
    """force_finish_tool on `report`, normalized; usage also counted on the invocation."""
    args = force_finish_tool(messages, report_tool, REPORT, model, metrics, invocation)
    return None if args is None else _normalize_report(args)


def run_subagent(focused_query: str, section_ids: list[str], dom: dict,
                 maps: dict[str, Any], pdf_path: str | Path | None,
                 metrics: dict[str, Any], round_index: int, visuals_seen: set[str],
                 sections_seen: set[str], verifications: list[Verdict],
                 model: str = SUBAGENT_MODEL,
                 prompts: dict[str, str] | None = None,
                 max_tool_calls: int = SUBAGENT_MAX_TOOL_CALLS
                 ) -> tuple[dict[str, Any], dict[str, Any]]:
    """One ephemeral read: pre-load the sections, look at what is needed, report back.

    The invocation owns nothing that outlives it. Its context is dropped on return and only
    the report goes up, which is the whole point of the split: the tool results that made
    the single-phase loop silt up never reach the navigation tier.

    Two pieces of state are deliberately scoped differently. `sent` is per invocation,
    because "the image was already provided, look at it above" is only true inside the
    context that received it — a later invocation never saw it. `visuals_seen` is
    run-level, because n_unique_visuals_fetched must mean in v2 what it meant in v1.

    Args:
        focused_query: The self-contained sub-question written by the orchestrator.
        section_ids: Sections to deliver, already resolved, in delivery order.
        dom: build_dom output.
        maps: build_section_maps output.
        pdf_path: Source PDF, or None to keep the invocation text-only.
        metrics: Shared metrics dict, mutated in place.
        round_index: Orchestrator round this invocation belongs to.
        visuals_seen: Run-level set of atom ids whose crop was served; mutated.
        sections_seen: Run-level set of sections actually delivered; mutated.
        verifications: Run-level sink of the check_* verdict records; mutated.
        model: Chat model id of this tier.
        prompts: Prompt bank; defaults to the module's.
        max_tool_calls: Ceiling on this invocation's turns.

    Returns:
        `(report, debug)` — the canonical report, and the artefacts a debug run dumps
        ({"round", "section_ids", "context", "crumbs", "report"}).
    """
    prompts = PROMPTS_V2 if prompts is None else prompts
    context, loaded = build_subagent_context(section_ids, dom, maps)
    delivered = [record["section_id"] for record in loaded]

    # The delivered sections are recorded under the v1 trace names: in v2 the code reads on
    # the model's behalf, so this is what "which sections did this run read" means. Stated
    # in _init_metrics_v2 and in the run's notes, because the name did not change.
    metrics["n_get_section_calls"] += len(section_ids)
    metrics["get_section_calls"].extend(section_ids)
    metrics["subagent_invocations"] += 1
    sections_seen.update(delivered)

    invocation: dict[str, Any] = {
        "round": round_index,
        "section_ids": list(delivered),
        "tokens": {"prompt": 0, "completion": 0, "total": 0},
        "tool_sequence": [],
        "n_get_visual": 0,
        "hit_cap": False,
        "report_source": "none",
    }
    metrics["subagent_stats"].append(invocation)
    debug = {"round": round_index, "section_ids": list(section_ids), "context": context,
             "crumbs": {record["section_id"]: record["crumb"] for record in loaded},
             "report": None}

    # Nothing was delivered: the orchestrator named only unknown ids, or the search found
    # nothing. Answered without an LLM call — there is no document text to read.
    if not loaded:
        report = _descriptive_fallback(section_ids)
        debug["report"] = report
        return report, debug

    report_tool = build_report_tool()
    tools = [build_get_visual_tool()] if pdf_path is not None else []
    tools.append(report_tool)   # kept last, as the model reads the list in order

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts["subagent_system"]},
        {"role": "user", "content": prompts["subagent_user_template"].format(
            focused_query=focused_query, context=context)},
    ]

    sent: dict[str, int] = {}
    report = None

    for step in range(max_tool_calls):
        metrics["n_steps"] += 1
        response = chat(model, messages, tools)
        _add_usage(response, metrics)
        _add_usage(response, invocation)
        message = response.choices[0].message

        action = _classify_protocol(message, _SUBAGENT_TOOL_NAMES, metrics)
        if action == "retry":
            messages.append({"role": "assistant", "content": message.content})
            messages.append({"role": "user", "content": _RETRY_PROTOCOL_MSG})
            continue
        if action == "stop":
            break

        messages.append(message.model_dump(exclude_none=True))
        image_parts: list[dict[str, Any]] = []

        # Every tool call gets a tool result, even once the report is in: leaving one
        # unanswered makes the next request 400 and corrupts the replayable trace.
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            seq_entry: dict[str, Any] = {"step": step, "tool": name, "tier": "subagent",
                                         "round": round_index}
            metrics["tool_sequence"].append(seq_entry)
            invocation["tool_sequence"].append(name)

            args = parse_tool_args(tool_call, "subagent", step, round_index, metrics,
                                   seq_entry, messages)
            if args is None:
                continue

            if name == GET_VISUAL:
                # _serve_visual is agent_dom's, imported unchanged, so its refusal entries
                # carry no tier: they are stamped here, on whatever it just appended.
                before_refusals = len(metrics["invalid_tool_args"])
                before_served = metrics["n_get_visual_calls"]
                content = _serve_visual(args.get("atom_id"), dom, pdf_path, sent, step,
                                        metrics, image_parts)
                for entry in metrics["invalid_tool_args"][before_refusals:]:
                    entry["tier"] = "subagent"
                    entry["round"] = round_index
                served = metrics["n_get_visual_calls"] - before_served
                if served:
                    invocation["n_get_visual"] += served
                    visuals_seen.add(args.get("atom_id"))
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": content})

            elif name == REPORT:
                report = _normalize_report(args)
                invocation["report_source"] = "tool"
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": "report received"})

            else:
                refuse_unknown_tool(tool_call, args, "subagent", step, round_index, metrics,
                                    messages)

        # After the tool results, never before: the assistant turn's tool calls must all be
        # answered before another role opens, or the next request 400s.
        if image_parts:
            messages.append({"role": "user", "content": image_parts})

        if report is not None:
            break
    else:
        # Loop exhausted without breaking: the ceiling was genuinely hit. An early break on
        # no_tool_no_answer also lands in the forced call below, but is NOT a ceiling hit.
        invocation["hit_cap"] = True

    if report is None:
        report = _force_report(messages, report_tool, model, metrics, invocation)
        invocation["report_source"] = "forced" if report is not None else "none"
    if report is None:
        logger.error("subagent produced no report at round %d, even forced", round_index)
        report = _descriptive_fallback(section_ids)

    check_key_figures(report, loaded, verifications, round_index)
    check_page_ranges(report, delivered, maps, verifications, round_index)

    debug["report"] = report
    return report, debug
