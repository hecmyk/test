"""The metrics dict of a v2 run — the contract with eval/analyze.py.

_init_metrics_v2 is the single place a metric key is declared. The first twenty keys are
agent_dom._init_metrics name for name and position for position; everything v2 adds comes
after them, flat. Nothing else in the package adds a key: the loops and the checks only
assign or append to keys that exist here.
"""

from typing import Any


def _init_metrics_v2() -> dict[str, Any]:
    """Pre-initialize every metric key; the v1 block keeps its exact v1 order.

    The first twenty keys are agent_dom._init_metrics, name for name and position for
    position. eval/analyze.py only strips the `agent_metrics.` and `tokens.` prefixes off
    the flattened columns, so a v2 metric nested under a tier would come out as a column of
    its own AND leave tokens_total empty — every cost table would break. The v2 keys are
    therefore appended flat, after the v1 block, and nothing above them ever moves.

    Two v1 keys are read differently in v2, deliberately, under unchanged names:
      n_steps            — LLM calls of BOTH tiers, so analyze's tokens_per_step stays
                           tokens over calls rather than tokens over rounds.
      get_section_calls  — the sections PRE-LOADED for the subagent. In v2 the code reads
                           them on the model's behalf, so used_section means "sections were
                           delivered", not "the model chose to read". Recorded in the run's
                           notes, which is the only place a run says what it changed.

    Returns:
        The metrics dict, every key present.
    """
    return {
        # --- v1 block: same keys, same order, same meaning unless said otherwise ------
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
        "get_section_calls": [],
        "invalid_tool_args": [],
        "n_search_calls": 0,
        "search_calls": [],
        "n_get_visual_calls": 0,
        "get_visual_calls": [],
        "n_unique_visuals_fetched": 0,
        "tool_sequence": [],
        # --- v2 block: additive, flat, never at the cost of a key above ---------------
        # Entries of tool_sequence and invalid_tool_args carry an extra "tier" field for
        # the same reason: added INSIDE the entry, so both lists stay readable exactly as
        # they are today by anything that ignores the new field.
        "orchestrator_rounds": 0,
        "orchestrator_tool_sequence": [],
        "subagent_invocations": 0,
        # One entry per invocation: {round, section_ids, tokens, tool_sequence,
        # n_get_visual, hit_cap, report_source}. A list of dicts survives json_normalize
        # untouched, the way invalid_tool_args already does. The real prompt tokens of each
        # invocation are in its `tokens`, from the endpoint's usage — no estimate.
        "subagent_stats": [],
        # page_recall is pure recall (one cited page inside gold scores 1.0, no precision
        # penalty), so a tier that cites more pages scores better for free. Logged to tell
        # a real gain from that artefact when v2 is put next to v1.
        "n_cited_pages": 0,
        # Aggregates of the non-repairing checks (check_key_figures, check_page_ranges).
        # The per-verdict records go to the caller's `verifications` sink and, from the
        # runners, to runs/{run_id}/verifications.jsonl; only these four flat keys stay in
        # the row. The denominators are kept next to the rates because a mean of
        # per-question rates is not the run's pooled rate. A rate is None, not 0.0, when
        # nothing was checked: 0.0 would read as "everything hallucinated".
        "n_key_figures_checked": 0,
        "key_figures_grounded_rate": None,
        "n_pages_checked": 0,
        "pages_in_range_rate": None,
    }
