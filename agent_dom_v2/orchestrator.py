"""The orchestrator — navigation on the skeleton, reads delegated to the subagent.

run_dom_agent_v2 is the entry point of the package. The orchestrator sees the skeleton, the
question and the reports it receives; it never sees section text or a crop. Its two
delegation tools, delegate_read and search_and_read, both end in the same subagent
invocation (_delegate); submit_draft is its finish-tool, shared with the single-phase
agent through dom_tools.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_dom import _add_usage, _check_pdf_matches_dom, _normalize_draft
from dom_tools import QA_SCHEMA, SUBMIT_DRAFT, build_search_index, build_submit_draft_tool

from .metrics import _init_metrics_v2
from .protocol import (_RETRY_PROTOCOL_MSG, _classify_protocol, chat, force_finish_tool,
                       parse_tool_args, refuse_unknown_tool, tool_schema)
from .sections import build_section_maps, resolve_section_ids
from .settings import (LOG_SUBAGENT_DETAILS, ORCHESTRATOR_MAX_ROUNDS, ORCHESTRATOR_MODEL,
                       PROMPTS_V2, SEARCH_TOP_K, SUBAGENT_MAX_TOOL_CALLS, SUBAGENT_MODEL)
from .subagent import run_subagent
from .verifications import Verdict, _aggregate_verifications

logger = logging.getLogger(__name__)

# (orchestrator system key, orchestrator user template key, finish-tool field schema). Only
# "qa" exists in v2: the KYC arm has no v2 prompt bank, and an unknown mode must raise here
# rather than run the wrong pair.
_MODES_V2 = {
    "qa": ("orchestrator_system", "orchestrator_user_template", QA_SCHEMA),
}

# Orchestrator tools. get_section and search are NOT among them: the orchestrator never
# reads a section, and its search goes through search_and_read, which forwards to the
# subagent within the same call.
DELEGATE_READ = "delegate_read"
SEARCH_AND_READ = "search_and_read"

# Detection only, never parsing — same contract as agent_dom._TOOL_NAMES, one tuple per
# tier (the subagent's is in subagent.py): a tool missing from its tier's tuple, when
# emitted as prose, is not counted as tool_call_in_content and falls through to
# no_tool_no_answer, which reads as a protocol regression the model never made.
# agent_dom's own helper closes over ITS tuple, so the detection is re-expressed here
# rather than imported.
_ORCHESTRATOR_TOOL_NAMES = (DELEGATE_READ, SEARCH_AND_READ, SUBMIT_DRAFT)


# --------------------------------------------------------------------------- #
# Orchestrator tools                                                           #
# --------------------------------------------------------------------------- #
class DelegateReadArgs(BaseModel):
    """Arguments of delegate_read.

    Schéma annoncé au modèle, jamais appliqué à sa réponse — voir _normalize_report pour la
    lecture ; here the reading is _delegate's, which serves what it can of a partly wrong
    call (unknown ids dropped and recorded, the rest delivered) rather than refusing it.
    """
    model_config = ConfigDict(extra="forbid")

    section_ids: list[str] = Field(
        description=("Section ids from the skeleton. An id of the form pN_eM is an atom id "
                     "and cannot be sent."))
    focused_query: str = Field(
        description=("Self-contained sub-question. The reader sees neither the skeleton "
                     "nor the original question."))


class SearchAndReadArgs(BaseModel):
    """Arguments of search_and_read.

    Schéma annoncé au modèle, jamais appliqué à sa réponse — voir _normalize_report pour la
    lecture ; here the reading is _delegate's. top_k's description is a template: the
    run's k is only known at call time, so build_search_and_read_tool formats it in.
    """
    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(
        description=("2 to 5 content words, not a question. Matching is lexical: a synonym "
                     "will not match."))
    top_k: int = Field(
        default=0,
        description=("How many sections to send. Defaults to {default_top_k}, which is "
                     "also the maximum."))
    focused_query: str = Field(
        description=("Self-contained sub-question. The reader sees neither the skeleton "
                     "nor the original question."))


def build_delegate_read_tool() -> dict[str, Any]:
    """Explicit delegation: the orchestrator picks the sections itself from the skeleton."""
    return tool_schema(DelegateReadArgs, DELEGATE_READ, (
        "Send one or more sections to the reader with the sub-question you want "
        "answered from them. Asking for a section delivers it AND all of its "
        "subsections. Use it when the skeleton already tells you where to look."))


def build_search_and_read_tool(default_top_k: int) -> dict[str, Any]:
    """Auto-forward: BM25 runs code-side and its hits go straight to the reader.

    Args:
        default_top_k: The run's configured k, advertised as the default and used as a
            ceiling — see run_dom_agent_v2.

    Returns:
        The tool definition, OpenAI function-calling shape.
    """
    tool = tool_schema(SearchAndReadArgs, SEARCH_AND_READ, (
        "Locate a term lexically and send the matching sections STRAIGHT to the "
        "reader with your sub-question — one call, no stop in between. Use it when "
        "no section title obviously names what the question asks about. The "
        "sections that were sent come back in the result, so you can see what the "
        "keyword actually reached and retry with better words."))
    top_k = tool["function"]["parameters"]["properties"]["top_k"]
    top_k["description"] = top_k["description"].format(default_top_k=default_top_k)
    return tool


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #
def _force_submit_draft_v2(messages: list[dict[str, Any]], submit_tool: dict[str, Any],
                           field_definitions: dict[str, str], schema: dict[str, Any],
                           model: str, metrics: dict[str, Any]) -> dict[str, Any] | None:
    """force_finish_tool on `submit_draft`, normalized (agent_dom's own version is not
    reused: it does not count the forced call into n_steps, v2 does)."""
    args = force_finish_tool(messages, submit_tool, SUBMIT_DRAFT, model, metrics)
    return None if args is None else _normalize_draft(args.get("fields"), field_definitions,
                                                      schema)


def run_dom_agent_v2(dom: dict, xml_skeleton: str, field_definitions: dict[str, str],
                     max_rounds: int = ORCHESTRATOR_MAX_ROUNDS,
                     model: str = ORCHESTRATOR_MODEL,
                     subagent_model: str = SUBAGENT_MODEL,
                     mode: str = "qa", question: str = "",
                     pdf_path: str | Path | None = None,
                     prompts: dict[str, str] | None = None,
                     search_top_k: int = SEARCH_TOP_K,
                     subagent_max_tool_calls: int = SUBAGENT_MAX_TOOL_CALLS,
                     log_subagent_details: bool = LOG_SUBAGENT_DETAILS,
                     verifications: list[Verdict] | None = None
                     ) -> tuple[dict, dict, list[dict[str, Any]]]:
    """Answer one question with the orchestrator / ephemeral-subagent split.

    The orchestrator sees the skeleton, the question and the reports it has received. It
    never sees section text and never sees a crop: every read happens inside a subagent
    invocation whose context is dropped on return.

    Signature deliberately mirrors agent_dom.run_dom_agent so a runner's run_one is a
    near-drop-in; the third return value is the only difference, and it is empty unless
    log_subagent_details is on.

    Args:
        dom: build_dom output.
        xml_skeleton: dom_render.build_outline_xml output.
        field_definitions: Mapping from field name to its definition text.
        max_rounds: Delegation budget of the orchestrator.
        model: Chat model id of the orchestrator.
        subagent_model: Chat model id of the reader; the same in v2.0, by design.
        mode: Prompt/schema pair; only "qa" exists in v2.
        question: The user's question.
        pdf_path: Source PDF, or None to keep the run text-only.
        prompts: Prompt bank; defaults to the module's.
        search_top_k: Sections a search_and_read delivers, and the ceiling on what the
            model may ask for.
        subagent_max_tool_calls: Ceiling on one invocation's turns.
        log_subagent_details: Collect the heavy per-invocation artefacts for a debug dump.
        verifications: Caller-owned sink for the per-verdict records of check_key_figures
            and check_page_ranges, appended in place — the same pattern as `metrics`. None
            keeps the records local: the four aggregate keys are computed either way, only
            the per-verdict detail is dropped.

    Returns:
        `(draft, metrics, debug)`. draft maps each field to one entry per schema key, the
        shape run_one reads; debug is one record per invocation, or [] when the flag is off.
    """
    started = time.perf_counter()
    metrics = _init_metrics_v2()
    system_key, user_key, schema = _MODES_V2[mode]
    prompts = PROMPTS_V2 if prompts is None else prompts

    if pdf_path is not None:
        _check_pdf_matches_dom(dom, pdf_path)

    maps = build_section_maps(dom)
    search_index = build_search_index(dom)
    submit_tool = build_submit_draft_tool(field_definitions, schema)
    tools = [build_delegate_read_tool(), build_search_and_read_tool(search_top_k),
             submit_tool]   # submit kept last, as the model reads the list in order

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts[system_key]},
        {"role": "user", "content": prompts[user_key].format(
            skeleton=xml_skeleton,
            field_definitions=json.dumps(field_definitions, indent=2, ensure_ascii=False),
            question=question)},
    ]

    visuals_seen: set[str] = set()
    sections_seen: set[str] = set()
    debug: list[dict[str, Any]] = []
    verifications = [] if verifications is None else verifications
    draft = None

    for round_index in range(max_rounds):
        metrics["orchestrator_rounds"] = round_index + 1
        metrics["n_steps"] += 1
        response = chat(model, messages, tools)
        _add_usage(response, metrics)
        message = response.choices[0].message

        action = _classify_protocol(message, _ORCHESTRATOR_TOOL_NAMES, metrics)
        if action == "retry":
            messages.append({"role": "assistant", "content": message.content})
            messages.append({"role": "user", "content": _RETRY_PROTOCOL_MSG})
            continue
        if action == "stop":
            break

        messages.append(message.model_dump(exclude_none=True))

        # Every tool call gets a tool result, even once the draft is in: leaving one
        # unanswered makes the next request 400 and corrupts the replayable trace.
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            seq_entry: dict[str, Any] = {"step": round_index, "tool": name,
                                         "tier": "orchestrator", "round": round_index}
            metrics["tool_sequence"].append(seq_entry)
            metrics["orchestrator_tool_sequence"].append(name)

            args = parse_tool_args(tool_call, "orchestrator", round_index, round_index,
                                   metrics, seq_entry, messages)
            if args is None:
                continue

            if name in (DELEGATE_READ, SEARCH_AND_READ):
                payload = _delegate(name, args, dom, maps, search_index, search_top_k,
                                    pdf_path, metrics, round_index, seq_entry,
                                    visuals_seen, sections_seen, verifications,
                                    subagent_model, prompts, subagent_max_tool_calls,
                                    debug, log_subagent_details)
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": json.dumps(payload, ensure_ascii=False)})

            elif name == SUBMIT_DRAFT:
                draft = _normalize_draft(args.get("fields"), field_definitions, schema)
                metrics["submit_draft_source"] = "tool"
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": "draft received"})

            else:
                refuse_unknown_tool(tool_call, args, "orchestrator", round_index, round_index,
                                    metrics, messages)

        if draft is not None:
            break
    else:
        # Budget exhausted without a draft. The forced call below answers from the reports
        # already gathered rather than abstaining out of exhaustion — an abstention must be
        # a judgement on the evidence, never a side effect of the round budget.
        metrics["hit_max_iterations"] = True

    if draft is None:
        draft = _force_submit_draft_v2(messages, submit_tool, field_definitions, schema,
                                       model, metrics)
        metrics["submit_draft_source"] = "forced" if draft is not None else "none"
    if draft is None:
        logger.error("no draft after the forced call; answering empty")
        draft = _normalize_draft({}, field_definitions, schema)

    metrics["n_unique_sections_fetched"] = len(sections_seen)
    metrics["n_unique_visuals_fetched"] = len(visuals_seen)
    metrics["n_cited_pages"] = sum(
        len(entry.get("source_pages") or []) if isinstance(entry.get("source_pages"), list)
        else 0 for entry in draft.values())
    _aggregate_verifications(verifications, metrics)
    metrics["duration_s"] = round(time.perf_counter() - started, 2)
    return draft, metrics, debug


def _delegate(tool: str, args: dict[str, Any], dom: dict, maps: dict[str, Any],
              search_index: Any, search_top_k: int, pdf_path: str | Path | None,
              metrics: dict[str, Any], round_index: int, seq_entry: dict[str, Any],
              visuals_seen: set[str], sections_seen: set[str],
              verifications: list[Verdict], subagent_model: str,
              prompts: dict[str, str], subagent_max_tool_calls: int,
              debug: list[dict[str, Any]], log_subagent_details: bool) -> dict[str, Any]:
    """Serve one delegation — both paths end in the same subagent invocation.

    delegate_read and search_and_read differ only in where the candidate ids come from:
    the orchestrator's own reading of the skeleton, or BM25. Everything after that is
    shared, which is why they share a body.

    The search path returns its hits to the orchestrator alongside the report. It costs a
    few hundred tokens and buys back the keyword-reformulation loop the auto-forward would
    otherwise remove: without it, seeing that a keyword reached the wrong sections would
    cost a whole further invocation.

    Args:
        tool: DELEGATE_READ or SEARCH_AND_READ.
        args: Parsed tool arguments.
        dom: build_dom output.
        maps: build_section_maps output.
        search_index: dom_tools.build_search_index output.
        search_top_k: Configured k, used as the default AND the ceiling.
        pdf_path: Source PDF, or None.
        metrics: Shared metrics dict, mutated in place.
        round_index: Current orchestrator round.
        seq_entry: This call's tool_sequence entry, mutated to carry the search hits.
        visuals_seen: Run-level set of atom ids served; mutated.
        sections_seen: Run-level set of sections delivered; mutated.
        verifications: Run-level sink of the check_* verdict records; mutated.
        subagent_model: Chat model id of the reader.
        prompts: Prompt bank.
        subagent_max_tool_calls: Ceiling on the invocation.
        debug: Per-invocation artefacts; appended to only when the flag is on.
        log_subagent_details: Whether to collect them.

    Returns:
        The tool result payload, serialized by the caller.
    """
    focused_query = args.get("focused_query")
    payload: dict[str, Any] = {}

    if tool == SEARCH_AND_READ:
        keyword = args.get("keyword")
        asked = args.get("top_k")
        top_k = asked if isinstance(asked, int) and not isinstance(asked, bool) else search_top_k
        # k is the variable v2.0 holds fixed, so a model asking for more is clamped rather
        # than obeyed. Not recorded: 0 occurrence on the pilot run.
        top_k = min(top_k, search_top_k)

        hits = search_index.search(keyword, max(top_k, 1))
        if isinstance(hits, dict):          # refusal; empty_keyword is the only one
            metrics["invalid_tool_args"].append(
                {"tool": tool, "args": {"keyword": keyword}, "reason": hits["error"],
                 "step": round_index, "tier": "orchestrator", "round": round_index})
            return {"search": hits, "report": None}

        metrics["n_search_calls"] += 1
        metrics["search_calls"].append(keyword)
        seq_entry["hits"] = [{"section_id": hit["section_id"], "n_matches": hit["n_matches"],
                              "score": hit["score"]} for hit in hits]
        candidates = [hit["section_id"] for hit in hits]
        payload["search"] = {
            "keyword": keyword,
            "hits": [{"section_id": hit["section_id"], "score": round(hit["score"], 2),
                      "n_matches": hit["n_matches"],
                      "path": maps["crumb"].get(hit["section_id"], "")} for hit in hits],
        }
    else:
        candidates = args.get("section_ids")

    section_ids = resolve_section_ids(candidates, maps, metrics, round_index, tool)
    payload["delivered"] = section_ids

    if not isinstance(focused_query, str) or not focused_query.strip():
        metrics["invalid_tool_args"].append(
            {"tool": tool, "args": {"focused_query": focused_query},
             "reason": "empty_focused_query", "step": round_index,
             "tier": "orchestrator", "round": round_index})
        payload["report"] = None
        payload["error"] = ("focused_query is required: the reader sees neither the "
                            "skeleton nor the question, so it needs the sub-question.")
        return payload

    report, invocation_debug = run_subagent(
        focused_query, section_ids, dom, maps, pdf_path, metrics, round_index,
        visuals_seen, sections_seen, verifications, model=subagent_model, prompts=prompts,
        max_tool_calls=subagent_max_tool_calls)
    if log_subagent_details:
        debug.append(invocation_debug)

    payload["report"] = report
    return payload
