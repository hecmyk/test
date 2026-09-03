"""Analysis tables for one scored benchmark run — the single source of truth.

Every function takes the DataFrame returned by load_scored and returns another one;
printing is left to the notebook. Three grouping functions cover the breakdowns asked of
the harness: cost_by answers "how much did it cost", quality_by answers "how good was
it", tool_set_mix answers "what did it reach for". Call them with "outcome", "page_ok",
"category", "answer_type" or "tool_set".

TOOL-AGNOSTIC BY CONSTRUCTION. The same functions run on a run with three tools
(get_section + search + get_visual), two, one, or none at all (the fullctx/oracle
baselines). A tool absent from a run contributes a structural zero, never a NaN that
would silently drop rows from a mean, and never a crash.

The three raw counters in the run log do NOT count the same thing — n_get_section_calls
counts every attempt, n_search_calls counts served calls, n_get_visual_calls counts
served FIRST-TIME calls — so no cost table reports them. They are replaced by a
reconstructed, homogeneous `attempts` per tool; the exact formulas are in
tools_overview's docstring. This is why there is no "calls_mean" column anywhere: the
old one counted get_section alone while its name said otherwise.

WHAT IS NOT MEASURED — ORDER. The log holds one ordered trace PER TOOL and no global
sequence; only refused calls carry a `step`. The interleaving between tools is not
recoverable from a run on disk. `tool_set` is a SET rendered in a fixed display order
and joined with "|": "search|section|visual" does NOT say the agent searched first.

Page conventions: an atom id carries a 0-indexed page, `gold_pages` is 1-indexed. The
comparison goes through metrics.page_recall, which owns that boundary for the whole
harness, so this module never holds a shifted page number.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # eval/, for metrics — self-sufficient import

from metrics import as_pages, page_recall   # noqa: E402

# Defined here rather than imported from run_benchmark on purpose: that import pulls
# agent_dom and with it openai, yaml, rank_bm25, fitz and tqdm, none of which an analysis
# session needs. Same directory as run_benchmark.RUNS, which sits one level down in
# eval/finragbench/ and builds it from its own _HERE.
RUNS = _HERE / "finragbench" / "runs"

# Display labels for the FinRAGBench-V categories: family prefix, task suffix. Applied in
# load_scored so every table, crosstab and plot is abbreviated the same way, including
# ad-hoc groupbys written in the notebook. A category outside this map passes through
# unchanged, and the original string is always kept in `category_full`.
CATEGORY_SHORT = {
    "Text-Inference": "TXT-Inf",
    "Text-MultiPage": "TXT-MP",
    "Table-Numerical Calculation": "TAB-Num",
    "Table-Compare and Sort": "TAB-Cmp",
    "Table-MultiPage": "TAB-MP",
    "Chart-Information Extraction": "CHA-Info",
    "Chart-Time Sensitive": "CHA-Time",
    "Chart-Numerical Calculation": "CHA-Num",
}

# Per tool: the ordered trace, the tool name as written in the log, the unique-fetched
# counter, and whether that trace already holds the refusals its own branch produced.
# trace_holds_refusals is the whole subtlety of `attempts` — see tools_overview.
TOOL_SPECS: dict[str, dict] = {
    "search": {
        "trace": "search_calls", "log_name": "search",
        "unique": None, "trace_holds_refusals": False,
    },
    "section": {
        "trace": "get_section_calls", "log_name": "get_section",
        "unique": "n_unique_sections_fetched", "trace_holds_refusals": True,
    },
    "visual": {
        "trace": "get_visual_calls", "log_name": "get_visual",
        "unique": "n_unique_visuals_fetched", "trace_holds_refusals": True,
    },
}
# Display order of the tool_set label only, NOT a claim about call order: the real
# chronological order is in tool_sequence. Fixed so that one set has exactly one label.
_DISPLAY_ORDER = ("search", "section", "visual")
_SEPARATOR = "|"
_MALFORMED = "malformed_json"
_ATOM_PAGE = r"^p(\d+)_e\d+$"

# Columns whose presence in the RAW frame proves the row came from a tool-using agent.
# Read before the baseline defaults are filled in, otherwise every row would qualify.
_AGENT_MARKERS = ("n_steps", "get_section_calls", "search_calls", "get_visual_calls",
                  "submit_draft_source")

# Statistics of overview() that cannot be compared across runs, see its docstring.
_NOT_CROSS_RUN = {"rows_with_invalid"}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _trace_len(df: pd.DataFrame, column: str) -> pd.Series:
    """Per-row length of one ordered trace; 0 where the run predates the column.

    A run made before get_visual existed, and every baseline row, simply has no
    get_visual_calls. Absent is not "unknown" here, it is "never called" — zero keeps
    such rows in the denominators instead of dropping them from a rate.
    """
    if column not in df.columns:
        return pd.Series(0, index=df.index)
    return df[column].apply(lambda v: len(v) if isinstance(v, list) else 0)


def _refusal_counts(df: pd.DataFrame, log_name: str,
                    reasons: set[str] | None = None) -> pd.Series:
    """Per-row number of refused calls on one tool, optionally filtered by reason.

    Counted row-wise off invalid_tool_args rather than through an explode-and-regroup on
    query_id: the alignment is positional, so it needs no assumption that query_id is
    unique in the frame (it is not, once two runs are concatenated).
    """
    if "invalid_tool_args" not in df.columns:
        return pd.Series(0, index=df.index)

    def count(entries) -> int:
        if not isinstance(entries, list):
            return 0
        return sum(1 for e in entries
                   if isinstance(e, dict) and e.get("tool") == log_name
                   and (reasons is None or e.get("reason") in reasons))

    return df["invalid_tool_args"].apply(count)


def _attempts(df: pd.DataFrame, tool: str) -> pd.Series:
    """Per-row attempts on one tool: its trace, plus the refusals the trace misses."""
    spec = TOOL_SPECS[tool]
    missing = {_MALFORMED} if spec["trace_holds_refusals"] else None
    return _trace_len(df, spec["trace"]) + _refusal_counts(df, spec["log_name"], missing)


def _column(df: pd.DataFrame, name: str | None) -> pd.Series:
    """`df[name]` with NaN as 0, or all-zeros when the column is absent. Read-only."""
    if name is None or name not in df.columns:
        return pd.Series(0.0, index=df.index)
    return df[name].fillna(0)


def _tool_set_labels(df: pd.DataFrame, has_agent: pd.Series) -> pd.Series:
    """The `tool_set` column: which tools each row used.

    "n/a" when the row carries no agent metrics at all — a baseline (fullctx, oracle_*)
    has no tools by construction, and so does an agent row whose run crashed before any
    metric was recorded. Filter on df["error"].notna() to tell those two apart.
    "none" is an agent row that read nothing before submitting.
    """
    flags = df[[f"used_{t}" for t in _DISPLAY_ORDER]]
    labels = flags.apply(
        lambda row: _SEPARATOR.join(t for t, used in zip(_DISPLAY_ORDER, row) if used)
        or "none",
        axis=1)
    return labels.where(has_agent, "n/a")


def _visual_refusals(df: pd.DataFrame) -> pd.DataFrame:
    """(query_id, id, reason) for every refused get_visual call, first reason per id."""
    empty = pd.DataFrame(columns=["query_id", "id", "reason"])
    if "invalid_tool_args" not in df.columns:
        return empty

    rows = []
    for query_id, entries in zip(df["query_id"], df["invalid_tool_args"]):
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get("tool") == "get_visual":
                args = entry.get("args") or {}
                rows.append({"query_id": query_id, "id": args.get("atom_id"),
                             "reason": entry.get("reason")})
    if not rows:
        return empty
    return (pd.DataFrame(rows)
            .groupby(["query_id", "id"], as_index=False, dropna=False)["reason"]
            .first())


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def load_scored(run_id: str) -> pd.DataFrame:
    """Flat table of a run's scored rows, one row per query.

    agent_metrics is flattened into top-level columns, so a row whose agent crashed (its
    metrics hold only wall_clock_s) comes out with NaN there rather than a missing key.
    Baseline runs (fullctx, oracle_*) don't emit the agent-only columns; they're filled
    with neutral defaults so no downstream function needs to branch.

    Derived columns added here, so every other function can assume them:
      outcome, page_ok, cited, tokens_per_step, n_invalid   — as before;
      attempts_search / attempts_section / attempts_visual, used_*, attempts, tool_set
        — the tool dimension, reconstructed from the ordered traces (see tools_overview).

    `category` is REPLACED by its short display label (see CATEGORY_SHORT) and the
    original string kept in `category_full`. The rewrite is deliberate and lossless: it
    is what makes every table, crosstab and plot — including ad-hoc ones written in the
    notebook — carry the same abbreviations without each having to remember to map.

    Args:
        run_id: The iteration to load.

    Returns:
        The scored rows plus the derived columns the other functions group on.
    """
    path = RUNS / run_id / "scored.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]

    # Runs predating the list schema carry a scalar pred_source_page. Folded onto the
    # current name here, on the raw rows, so exactly ONE column reaches the frame and no
    # function downstream has to know which shape its run was written in.
    for row in rows:
        legacy = row.pop("pred_source_page", None)
        row["pred_source_pages"] = as_pages(row.get("pred_source_pages") or legacy)

    df = pd.json_normalize(rows)
    df.columns = [c.replace("agent_metrics.", "").replace("tokens.", "tokens_")
                  for c in df.columns]

    # BEFORE the defaults below: once they are filled, every row looks like an agent row.
    markers = [c for c in _AGENT_MARKERS if c in df.columns]
    has_agent = (df[markers].notna().any(axis=1) if markers
                 else pd.Series(False, index=df.index))

    # Compat baseline : les runs baseline (fullctx, oracle_*) n'ont pas les colonnes
    # agent-only. Remplies ici avec des neutres pour que les tables tournent sans if
    # partout. Fallback tokens : les baselines utilisent reader_tokens.
    _agent_only_defaults = {
        "tokens_total": float("nan"),
        "n_steps": float("nan"),
        "n_unique_sections_fetched": float("nan"),
        "n_unique_visuals_fetched": float("nan"),
        "hit_max_iterations": float("nan"),
        "duration_s": float("nan"),
        "submit_draft_source": "n/a",
    }
    for col, default in _agent_only_defaults.items():
        if col not in df.columns:
            df[col] = default
    if "invalid_tool_args" not in df.columns:
        df["invalid_tool_args"] = [[] for _ in range(len(df))]
    if "tool_sequence" not in df.columns:
        df["tool_sequence"] = [[] for _ in range(len(df))]
    if df["tokens_total"].isna().all() and "reader_tokens" in df.columns:
        df["tokens_total"] = df["reader_tokens"]

    if "category" in df.columns:
        df["category_full"] = df["category"]
        df["category"] = df["category"].map(lambda c: CATEGORY_SHORT.get(c, c))

    # Ordered so the stronger signal wins: a crashed run is not an abstention, and an
    # abstention is not a wrong answer even though the judge scores both 0.
    outcome = pd.Series("wrong", index=df.index)
    outcome[df["judge_score"] == 1] = "correct"
    outcome[df["judge_score"].isna()] = "judge_failed"
    outcome[df["pred_answer"].isna()] = "abstention"
    outcome[df["error"].notna()] = "error"

    df["outcome"] = outcome
    df["page_ok"] = df["page_recall"] == 1
    # Length, not notna(): an empty list IS notna(), so an abstention would read as cited.
    df["cited"] = df["pred_source_pages"].str.len() > 0
    df["tokens_per_step"] = df["tokens_total"] / df["n_steps"]
    df["n_invalid"] = df["invalid_tool_args"].str.len()

    # used_* is derived from ATTEMPTS, not from the served counters: a query whose three
    # get_visual calls were all refused DID use the tool, and a counter-based flag would
    # have made it indistinguishable from a query that never tried.
    for tool in TOOL_SPECS:
        attempts = _attempts(df, tool)
        df[f"attempts_{tool}"] = attempts
        df[f"used_{tool}"] = attempts > 0
    df["attempts"] = sum(df[f"attempts_{t}"] for t in TOOL_SPECS)
    df["tool_set"] = _tool_set_labels(df, has_agent)
    return df


# --------------------------------------------------------------------------- #
# Run-level                                                                    #
# --------------------------------------------------------------------------- #
def overview(df: pd.DataFrame) -> pd.DataFrame:
    """Run-level rates, one row per statistic.

    The `cross_run` column says whether a statistic may be compared between two runs.
    Only rows_with_invalid is marked False: it counts refusals over every exposed tool,
    so a run with get_visual is mechanically more "faulty" than one without, for reasons
    that have nothing to do with the navigation getting worse.

    attempts_mean is comparable — it counts tool calls, all tools together — but a search
    call and a get_section call do not cost the same, so tokens_mean stays the honest
    cost comparator between runs.

    Args:
        df: load_scored output.

    Returns:
        One row per statistic: `value` and `cross_run`.
    """
    cited = df[df["cited"]]
    stats = {
        "n_queries": len(df),
        "n_judge_failed": df["outcome"].eq("judge_failed").sum(),
        "accuracy": df["judge_score"].mean(),
        "token_f1": df["token_f1"].mean(),
        "page_recall": df["page_recall"].mean(),
        "page_recall_cited_only": cited["page_recall"].mean(),
        "abstention_rate": (df["outcome"] == "abstention").mean(),
        "error_rate": (df["outcome"] == "error").mean(),
        "forced_rate": (df["submit_draft_source"] == "forced").mean(),
        "cap_rate": df["hit_max_iterations"].mean(),
        "rows_with_invalid": (df["n_invalid"] > 0).mean(),
        "attempts_mean": df["attempts"].mean(),
        "attempts_median": df["attempts"].median(),
        "tokens_mean": df["tokens_total"].mean(),
    }
    table = pd.DataFrame({"value": stats}).round(3)
    table["cross_run"] = [name not in _NOT_CROSS_RUN for name in table.index]
    return table


# --------------------------------------------------------------------------- #
# Grouped tables                                                               #
# --------------------------------------------------------------------------- #
def cost_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Tool-call and token cost grouped by `column`.

    attempts_* count EVERY attempt on a tool, refusals and repeats included, and are
    homogeneous across the three (see tools_overview for the formulas). The per-tool
    columns break the total down; a tool absent from the run shows a flat zero.

    The median and cap_rate sit next to the mean on purpose: a run that hits MAX_STEPS
    has its call count truncated, so a group that caps often sees its mean pulled toward
    the cap and understates the real gap.

    Args:
        df: load_scored output, pre-filtered when the scope needs it (e.g. df[df.cited]).
        column: Column to group on.

    Returns:
        One row per group, largest group first.
    """
    g = df.groupby(column)
    table = pd.DataFrame({
        "n": g.size(),
        "attempts_mean": g["attempts"].mean(),
        "attempts_median": g["attempts"].median(),
        "att_section": g["attempts_section"].mean(),
        "att_search": g["attempts_search"].mean(),
        "att_visual": g["attempts_visual"].mean(),
        "unique_sections": g["n_unique_sections_fetched"].mean(),
        "unique_visuals": g["n_unique_visuals_fetched"].mean(),
        "steps_mean": g["n_steps"].mean(),
        "tokens_mean": g["tokens_total"].mean(),
        "tokens_per_step": g["tokens_per_step"].mean(),
        "cap_rate": g["hit_max_iterations"].mean(),
        "invalid_rate": g["n_invalid"].apply(lambda s: (s > 0).mean()),
        "duration_mean": g["duration_s"].mean(),
    })
    return table.sort_values("n", ascending=False).round(2)


def quality_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Accuracy, page recall and abstention rate grouped by `column`.

    page_recall counts an abstention as 0 by design, so page_recall_cited is given beside
    it: without that column, "did not find the page" is dominated by rows citing none.

    Args:
        df: load_scored output.
        column: Column to group on, "tool_set" included.

    Returns:
        One row per group, largest group first.
    """
    g = df.groupby(column)
    cited = df[df["cited"]].groupby(column)
    table = pd.DataFrame({
        "n": g.size(),
        "accuracy": g["judge_score"].mean(),
        "token_f1": g["token_f1"].mean(),
        "page_recall": g["page_recall"].mean(),
        "page_recall_cited": cited["page_recall"].mean(),
        "n_cited": cited.size(),
        "abstention_rate": g["outcome"].apply(lambda s: (s == "abstention").mean()),
        "forced_rate": g["submit_draft_source"].apply(lambda s: (s == "forced").mean()),
    })
    return table.sort_values("n", ascending=False).round(3)


# --------------------------------------------------------------------------- #
# Tool usage                                                                   #
# --------------------------------------------------------------------------- #
def tools_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Run-level usage of every read tool, one row per tool.

    The `attempts` columns are RECONSTRUCTED, because the three raw counters in the run
    log do not count the same thing: n_get_section_calls counts every attempt,
    n_search_calls counts served calls, and n_get_visual_calls counts served FIRST-TIME
    calls. Putting those three side by side would invite a comparison that is not valid,
    so they are reported nowhere in this module. Exact formulas, per row:

        section   len(get_section_calls) + #{invalid: tool=get_section,
                                                      reason=malformed_json}
        search    len(search_calls)      + #{invalid: tool=search}
        visual    len(get_visual_calls)  + #{invalid: tool=get_visual,
                                                      reason=malformed_json}

    Why the two shapes. get_section_calls and get_visual_calls are appended at the top of
    their dispatch branch, before it validates anything, so they ALREADY hold the
    refusals that branch produced (unknown id, repeat, degenerate bbox, no_pdf...); the
    only attempt missing from them is one whose JSON never parsed, since the loop
    `continue`s on it before reaching the branch. search_calls is appended only once a
    call is served, so ALL of its refusals — empty_keyword and malformed_json alike —
    have to be added back.

    unique_mean is the served side: distinct sections fetched, distinct crops shown. It
    has no meaning for search (no unique counter) and comes out NaN there.

    Args:
        df: load_scored output.

    Returns:
        One row per tool: use_rate, attempts_mean/median/max, unique_mean, n_refused,
        refusal_rate (share of queries with at least one refusal on that tool). A tool
        the run never exposed comes out as a row of zeros, not as a missing row.
    """
    n = len(df)
    rows = {}
    for tool, spec in TOOL_SPECS.items():
        attempts = df[f"attempts_{tool}"]
        refused = _refusal_counts(df, spec["log_name"])
        rows[tool] = {
            "use_rate": (attempts > 0).mean(),
            "attempts_mean": attempts.mean(),
            "attempts_median": attempts.median(),
            "attempts_max": attempts.max(),
            "unique_mean": _column(df, spec["unique"]).mean() if spec["unique"]
                           else float("nan"),
            "n_refused": refused.sum(),
            "refusal_rate": (refused > 0).mean() if n else float("nan"),
        }
    return pd.DataFrame(rows).T.round(3)


def tool_set_mix(df: pd.DataFrame, column: str = "category",
                 normalize: bool = True) -> pd.DataFrame:
    """Share of each tool set within every group of `column`.

    This is the "is there a routing pattern per question type" table: read a row and you
    see what the agent reached for on that kind of question — not in which order.

    Args:
        df: load_scored output.
        column: Column to group on, e.g. "category", "answer_type" or "outcome".
        normalize: Rates per row (the default); False gives raw counts.

    Returns:
        Groups in rows, tool sets in columns, with the group size as the first column.
        Two runs whose agents had different tools produce different column sets — compare
        the tables side by side, do not concatenate them.
    """
    table = pd.crosstab(df[column], df["tool_set"],
                        normalize="index" if normalize else False)
    table.insert(0, "n", df.groupby(column).size())
    return table.sort_values("n", ascending=False).round(3)


def tools_per_sample(df: pd.DataFrame, column: str = "category") -> pd.DataFrame:
    """Mean tool attempts per query, per tool, within every group of `column`.

    Intensity, where tool_set_mix gives routing: that one says WHICH tools a group
    reached for, this one says HOW MANY times. Counts ATTEMPTS (refusals and repeats
    included, see tools_overview), which is why n_invalid_mean sits next to them — a
    group that looks tool-hungry may only be retrying a bad id.

    Args:
        df: load_scored output.
        column: Column to group on, e.g. "category" or "outcome".

    Returns:
        Groups in rows, one mean per tool plus n, total and n_invalid_mean, biggest
        group first.
    """
    g = df.groupby(column)
    table = pd.DataFrame({"n": g.size()})
    for tool in TOOL_SPECS:
        table[tool] = g[f"attempts_{tool}"].mean()
    table["total"] = g["attempts"].mean()
    table["n_invalid_mean"] = g["n_invalid"].mean()
    return table.sort_values("n", ascending=False).round(2)


def tool_sequence_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per tool call, in chronological order — the base table of the order views.

    Reads tool_sequence, so it holds only the calls served through the tool-calling
    protocol: prose calls and the forced submit_draft are outside it by construction (see
    _init_metrics in agent_dom). A run predating that key contributes nothing, which is
    indistinguishable from an agent that called no tool — filter on n_steps before
    comparing across runs.

    Args:
        df: load_scored output.

    Returns:
        Columns query_id, position (0-based rank in the query's sequence), step, tool,
        section_id (the id a get_section asked for, NaN elsewhere) and hit_ids (the ids a
        served search returned, in rank order, NaN elsewhere). The other arguments and the
        hits' scores stay in the raw column, for ad-hoc work in the notebook.
    """
    ex = df[["query_id"]].join(df["tool_sequence"].explode().rename("call"))
    ex = ex[ex["call"].notna()].copy()
    ex["position"] = ex.groupby("query_id").cumcount()
    ex["step"] = ex["call"].str["step"]
    ex["tool"] = ex["call"].str["tool"]
    ex["section_id"] = ex["call"].str["args"].str["section_id"]
    ex["hit_ids"] = ex["call"].str["hits"].map(
        lambda hs: [h["section_id"] for h in hs] if isinstance(hs, list) else None)
    return ex.drop(columns="call")


def search_follow_through(df: pd.DataFrame) -> pd.DataFrame:
    """One row per get_section, told apart by whether an earlier search offered that id.

    Membership is tested against the UNION of every search served so far in the query, not
    against the last one only: an agent that runs two searches then fetches a section from
    the first is still following search.

    A RATE READ ALONE MEANS NOTHING. The XML skeleton already lists every section_id, so
    the agent can land on a hit without search having anything to do with it, and the
    chance of that grows as search returns a larger share of the document. Compare runs
    and groups against each other, never a single number against intuition.

    Args:
        df: load_scored output.

    Returns:
        Columns query_id, position, section_id, after_search (a search was served before
        this call) and from_search (the id was among its hits). Rows where after_search is
        False carry no signal — filter them out before taking a mean.
    """
    search_name = TOOL_SPECS["search"]["log_name"]
    section_name = TOOL_SPECS["section"]["log_name"]
    rows = []
    for query_id, group in tool_sequence_long(df).groupby("query_id", sort=False):
        seen: set = set()
        for call in group.itertuples():
            if call.tool == search_name and isinstance(call.hit_ids, list):
                seen |= set(call.hit_ids)
            elif call.tool == section_name:
                rows.append({"query_id": query_id, "position": call.position,
                             "section_id": call.section_id,
                             "after_search": bool(seen),
                             "from_search": call.section_id in seen})
    return pd.DataFrame(rows, columns=["query_id", "position", "section_id",
                                       "after_search", "from_search"])


def visual_usage(df: pd.DataFrame, column: str = "category") -> pd.DataFrame:
    """Where the visual channel was used, and what the answers looked like there.

    acc_with / acc_without ARE NOT AN EFFECT OF get_visual. The model calls it when the
    text failed it, so the two columns compare hard queries against easy ones and
    acc_delta is expected to be negative even if the crop helped every time it was used.
    What the table does answer: did the agent reach for images on the categories where
    charts and tables carry the answer, or somewhere else entirely.

    attempts_mean_users beside served_mean_users is the waste signal: a gap between them
    is calls the loop refused or answered from cache, never crops the model saw.

    Args:
        df: load_scored output.
        column: Column to group on, e.g. "category" or "answer_type".

    Returns:
        One row per group, largest first. A run with no visual channel at all comes back
        with n and visual_rate filled and every other column NaN.
    """
    g = df.groupby(column)
    users = df[df["used_visual"]].groupby(column)
    table = pd.DataFrame({
        "n": g.size(),
        "visual_rate": g["used_visual"].mean(),
        "n_with": users.size(),
        "attempts_mean_users": users["attempts_visual"].mean(),
        "served_mean_users": users["n_unique_visuals_fetched"].mean(),
        "acc_with": users["judge_score"].mean(),
        "acc_without": df[~df["used_visual"]].groupby(column)["judge_score"].mean(),
    })
    table["acc_delta"] = table["acc_with"] - table["acc_without"]
    return table.sort_values("n", ascending=False).round(3)


def visual_calls(df: pd.DataFrame) -> pd.DataFrame:
    """One row per get_visual attempt, with where it pointed and whether it was served.

    on_gold_page is the aiming metric: it says whether the crop the agent asked for sits
    on a page the dataset marks as holding the answer. Low on_gold_page with a high
    visual_rate means the tool is being called, but at the wrong figures.

    A refused id is matched back by (query_id, id), so if the same id was refused twice
    with two different reasons only the first is shown. Repeats of a SERVED id are not
    refusals — the loop answers them from `visuals_sent` — and appear here as extra rows
    with refused False.

    Args:
        df: load_scored output.

    Returns:
        Columns query_id, category, outcome, atom_id, page (0-indexed), on_gold_page,
        refused, reason. Empty frame when the run has no get_visual trace.
    """
    columns = ["query_id", "category", "outcome", "atom_id", "page", "on_gold_page",
               "refused", "reason"]
    if "get_visual_calls" not in df.columns:
        return pd.DataFrame(columns=columns)

    base = df[["query_id", "category", "outcome", "gold_pages"]]
    ex = base.join(df["get_visual_calls"].explode().rename("atom_id"))
    ex = ex[ex["atom_id"].notna()]
    if ex.empty:
        return pd.DataFrame(columns=columns)

    # A malformed call can put anything in atom_id, hence astype(str) then a strict match:
    # what does not look like pN_eM gets page <NA> and on_gold_page False.
    page = ex["atom_id"].astype(str).str.extract(_ATOM_PAGE, expand=False)
    ex["page"] = pd.to_numeric(page, errors="coerce").astype("Int64")
    # isinstance, not `gold or []`: a missing gold_pages comes through as float NaN, which
    # is truthy, and list(nan) raises.
    ex["on_gold_page"] = [
        False if pd.isna(p) or not isinstance(gold, (list, tuple))
        else page_recall(int(p), list(gold)) == 1.0
        for p, gold in zip(ex["page"], ex["gold_pages"])
    ]

    # merge, not join-on-MultiIndex: an explicit two-key left merge is the one form whose
    # behaviour does not depend on how the right-hand index was built.
    refusals = _visual_refusals(df)
    if refusals.empty:
        ex["reason"] = pd.NA
    else:
        ex = ex.merge(refusals, how="left",
                      left_on=["query_id", "atom_id"], right_on=["query_id", "id"])
    ex["refused"] = ex["reason"].notna()
    return ex[columns]


def invalid_calls(df: pd.DataFrame) -> pd.DataFrame:
    """One row per tool call the loop refused, with the shape of the id asked for.

    The id is read from whichever key the tool records it under — get_section stores
    section_id, get_visual stores atom_id, and a call whose JSON never parsed stores raw.
    Reading section_id alone (as this function used to) sent every search and get_visual
    refusal into form "other", which then looked like a malformed id and was not one.

    Args:
        df: load_scored output.

    Returns:
        Columns query_id, outcome, tool, reason, step, id, form — where form is
        section (sN), atom (pN_eM), hybrid (sN_eM), none (the call carried no id) or
        other.
    """
    calls = df["invalid_tool_args"].explode().rename("call")
    ex = df[["query_id", "outcome"]].join(calls)
    ex = ex[ex["call"].notna()]

    ex["tool"] = ex["call"].str["tool"]
    ex["reason"] = ex["call"].str["reason"]
    ex["step"] = ex["call"].str["step"]
    args = ex["call"].str["args"]
    ex["id"] = (args.str["section_id"]
                .fillna(args.str["atom_id"])
                .fillna(args.str["raw"]))

    form = pd.Series("other", index=ex.index)
    form[ex["id"].isna()] = "none"
    form[ex["id"].str.fullmatch(r"s\d+", na=False)] = "section"
    form[ex["id"].str.fullmatch(r"p\d+_e\d+", na=False)] = "atom"
    form[ex["id"].str.fullmatch(r"s\d+_e\d+", na=False)] = "hybrid"
    ex["form"] = form
    return ex.drop(columns="call")


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #
def plot_attempts_by_outcome(df: pd.DataFrame) -> None:
    """Boxplot of total tool attempts per outcome, group size under each box."""
    groups = list(df.groupby("outcome")["attempts"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([s.dropna() for _, s in groups])
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"{name}\nn={len(s)}" for name, s in groups])
    ax.set_ylabel("tool attempts (tous outils)")
    ax.set_title("Coût d'appels par outcome")
    fig.tight_layout()


def plot_recall_by_category(df: pd.DataFrame, min_n: int = 10) -> None:
    """Bar chart of page recall per category; groups under `min_n` are dropped.

    Category labels are the short ones (CATEGORY_SHORT), applied at load time.
    """
    table = quality_by(df, "category")
    table = table[table["n"] >= min_n].sort_values("page_recall")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(table.index.astype(str), table["page_recall"])
    for bar, n in zip(bars, table["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("page recall")
    ax.set_ylim(0, 1)
    ax.set_title(f"Page recall par catégorie (n >= {min_n})")
    fig.tight_layout()


def plot_tool_mix_by_step(df: pd.DataFrame, min_n: int = 10) -> None:
    """Stacked bars of the tool mix at each step; steps under `min_n` queries are dropped.

    The cut is not cosmetic. The number of queries still running collapses with the step,
    so without it the right-hand bars are one or two queries rendered as percentages —
    exactly the part of the plot one is tempted to read as a trend.
    """
    long = tool_sequence_long(df)
    alive = long.groupby("step")["query_id"].nunique()
    mix = pd.crosstab(long["step"], long["tool"], normalize="index")
    mix = mix.loc[alive[alive >= min_n].index]
    if mix.empty:
        # Baselines and runs predating tool_sequence have nothing to draw, and pandas'
        # bar chart raises on an empty frame instead of producing an empty one.
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    mix.plot.bar(stacked=True, ax=ax, width=0.8)
    ax.set_xticklabels([f"{s}\nn={alive[s]}" for s in mix.index], rotation=0)
    ax.set_ylabel("part des appels du step")
    ax.set_title(f"Mix d'outils par step (n >= {min_n})")
    ax.legend(title="", bbox_to_anchor=(1.0, 1.0), loc="upper left")
    fig.tight_layout()
