"""Isolated-visual ablation: the crops the agent saw, re-asked with nothing else.

One call per query, the same crops and the same question, no skeleton, no tool history and
no system prompt — what it measures is whether the surrounding text was helping the visual
reasoning or competing with it. Output is a normal run under eval/runs/, scored by the
normal score_results, so eval/analyze.py loads it like any other; isolation_matrix at
the bottom of this file is what compares the two arms.
"""

import base64
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))  # repo root, for config / dom_visual / eval.baselines
sys.path.insert(0, str(_HERE.parent))      # eval/, for metrics
sys.path.insert(0, str(_HERE))             # eval/finragbench/, for run_benchmark and score_results

from config import MISTRAL_AGENT_MODEL, MISTRAL_API_KEY, MISTRAL_BASE_URL  # noqa: E402
from dom_visual import get_visual                                          # noqa: E402
from eval.baselines.reader import _parse_reader_json                       # noqa: E402
from metrics import page_recall                                            # noqa: E402
from run_benchmark import (ARTIFACTS, RUNS, SCORED_NAME,                   # noqa: E402
                           load_or_build_dom, pdf_for)
from score_results import score_results                                    # noqa: E402

client = OpenAI(base_url=MISTRAL_BASE_URL, api_key=MISTRAL_API_KEY)

# The whole prompt. No system message, no role, no instruction beyond the output contract
# — every sentence added here is context put back, which is what this arm removes. The
# second key is an analysis field: it buys the failure taxonomy of the abstentions, and it
# is never scored.
PROMPT = (
    "Answer the following question based on the provided image(s). Return a JSON object "
    "with two keys:\n"
    "- 'answer': your answer, or null if you cannot answer confidently.\n"
    "- 'reasoning': what you see in the image(s) relevant to the question, and if you "
    "cannot answer, explain what specifically is missing or unclear.\n"
    "Question: {question}")

# Crop margin for the padded arm, in the raster pixels the dom bboxes live in (150 dpi
# here). Fixed, not a percentage of the bbox: what gets clipped is an axis label row, a
# legend, a unit — roughly 10-12pt, ~25px at 150 dpi, and that size does NOT scale with
# the figure. A percentage would under-pad small charts and swallow neighbouring columns
# on full-page ones. Sweep it (0 / 40 / 80) rather than tuning it once.
CROP_PAD_PX = 40


def _outcome(row: dict) -> str:
    """Outcome label of one scored row, by analyze.load_scored's rule (same order)."""
    if row.get("error") is not None:
        return "error"
    if row.get("pred_answer") is None:
        return "abstention"
    if row.get("judge_score") not in (0, 1):
        return "judge_failed"
    return "correct" if row["judge_score"] == 1 else "wrong"


def _crops(atom_ids: list[str], dom: dict, pdf_path: Path,
           pad: int = 0) -> tuple[list[dict], list[str]]:
    """Crops of one query as image blocks, plus the ids that made it in.

    The crops are REGENERATED, not replayed: get_visual is deterministic on
    (atom_id, dom, pdf), so re-cropping the frozen dom gives back the image the agent was
    shown — and an id the agent's loop refused re-fails here and is dropped, which is why
    the trace needs no refusal filter. `pad` is the one exception to "same image": it
    deliberately widens the crop, and the eligibility of an atom stays unchanged (see
    dom_visual._clamp_bbox) so a padded run covers exactly the same atoms.
    """
    blocks, sent = [], []
    for atom_id in atom_ids:
        result = get_visual(atom_id, dom, pdf_path, pad=pad)
        if "error" in result:
            continue
        blocks.append({"type": "image_url",
                       "image_url": {"url": f"data:image/png;base64,{result['image_b64']}"}})
        sent.append(atom_id)
    return blocks, sent


def _ask(question: str, images: list[dict]) -> tuple[str | None, str | None, int]:
    """One isolated call; `(answer, reasoning, tokens)`, both None when unusable.

    `reasoning` is an analysis field only — it exists to hand-taxonomise the failure
    modes of the abstentions (unreadable chart, information absent, missing context,
    question misread). Nothing downstream reads it: not the judge, not the matrix.
    """
    # temperature/response_format follow the baselines' single-call convention, not the
    # agent loop (which sets neither). Parsing stays defensive: response_format is
    # advisory on this endpoint. The images carry no label — the atom id and the page
    # would be context handed back.
    response = client.chat.completions.create(
        model=MISTRAL_AGENT_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(question=question)},
            *images,
        ]}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    usage = getattr(response, "usage", None)
    tokens = ((getattr(usage, "prompt_tokens", 0) or 0)
              + (getattr(usage, "completion_tokens", 0) or 0))

    data = _parse_reader_json((response.choices[0].message.content or "").strip())
    if data is None:
        return None, None, tokens
    answer, reasoning = data.get("answer"), data.get("reasoning")
    return ((answer if isinstance(answer, str) and answer.strip() else None),
            (reasoning if isinstance(reasoning, str) and reasoning.strip() else None),
            tokens)


def _by_document(rows: list[dict]):
    """Group rows by document and resolve its dom and PDF once; yields (stem, dom, pdf, rows).

    A query belongs to the document whose folder name prefixes its id, as in run_benchmark.
    A document with no PDF on disk is skipped whole: no PDF, no crop, no arm — better
    absent than recorded as imageless rows that would pollute a denominator.
    """
    stems = sorted(path.parent.name for path in ARTIFACTS.glob("*/doc_ocr.json"))
    by_stem: dict[str, list[dict]] = {}
    for row in rows:
        stem = next((s for s in stems if row["query_id"].startswith(s + "_")), None)
        if stem is not None:
            by_stem.setdefault(stem, []).append(row)

    for stem, doc_rows in by_stem.items():
        pdf_path = pdf_for(stem)
        if pdf_path is None:
            continue
        dom, _ = load_or_build_dom(ARTIFACTS / stem)   # `_` is the skeleton: the ablation
        yield stem, dom, pdf_path, doc_rows


def run_one(row: dict, dom: dict, pdf_path: Path, pad: int = 0) -> dict:
    """Replay one query in isolation; the row schema run_benchmark.run_one writes.

    Identity columns are copied from the source row, never re-read from the dataset: the
    two arms are joined on query_id and compared field by field, and a second read is a
    second chance to differ.
    """
    started = time.perf_counter()
    images, sent = _crops(row["atom_ids"], dom, pdf_path, pad=pad)
    if not images:
        answer, reasoning, tokens = None, None, 0
        error = "no_crop: every atom id failed to re-crop"
    else:
        try:
            answer, reasoning, tokens = _ask(row["question"], images)
            error = None
        except Exception as e:
            answer, reasoning, tokens, error = None, None, 0, f"{type(e).__name__}: {e}"

    return {
        "query_id": row["query_id"],
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "gold_pages": row.get("gold_pages"),
        "answer_type": row.get("answer_type"),
        "category": row.get("category"),
        "pred_answer": answer,
        # Neither is asked of this arm. An empty pred_source_pages means score_results
        # writes a flat page_recall of 0.0 for every row here — structural, not a result,
        # do not read it.
        "pred_quote": "",
        "pred_source_pages": [],
        "agent_metrics": {
            "mode": "isolated_visual",
            "source_run": row["source_run"],
            "agent_outcome": row["agent_outcome"],
            "atom_ids": sent,
            "n_images": len(images),
            # Recorded per row, not only in run_metadata: dump_crops reads it back to
            # redraw exactly what was sent, and a frame holding two arms keeps the
            # variable that separates them.
            "crop_pad_px": pad,
            "tokens_total": tokens,
            "wall_clock_s": round(time.perf_counter() - started, 2),
            # Analysis field: read by hand to taxonomise the abstentions. Never scored.
            "reasoning": reasoning,
        },
        "error": error,
    }


def run_isolated(source_run_id: str, run_id: str, limit: int | None = None,
                 pad: int = 0) -> Path:
    """Replay a scored run's chart+visual queries in isolation, then score the result.

    Selection: a chart category, at least one crop served, and an outcome worth comparing
    against — abstentions included, since an agent that had the right figure and could not
    read it is exactly the case this arm is meant to catch.

    No resume mode: a re-run of the same run_id rewrites its per-document files, so use a
    fresh run_id, which also gives the judge a fresh cache.

    Args:
        source_run_id: The agent run to replay, e.g. "run8".
        run_id: Name of this arm's run, e.g. "isolated_visual_from_run8".
        limit: Stop after N queries in total (pilot runs); None replays them all.
        pad: Crop margin in raster px (see CROP_PAD_PX). 0, the default, reproduces the
            tight-crop arm. The A/B is padded-vs-tight ISOLATED, never padded-vs-agent:
            one variable at a time.

    Returns:
        Path of the scored JSONL.
    """
    rows = [json.loads(line) for line in
            (RUNS / source_run_id / SCORED_NAME).read_text(encoding="utf-8").splitlines()
            if line.strip()]

    selected = []
    for row in rows:
        metrics = row.get("agent_metrics") or {}
        outcome = _outcome(row)
        # Chart-* is the raw label the log holds; analyze abbreviates it at load time only.
        if not (row.get("category") or "").startswith("Chart-"):
            continue
        if not metrics.get("n_get_visual_calls"):
            continue
        if outcome not in ("correct", "wrong", "abstention"):
            continue
        # Deduplicated in first-call order: a repeat was answered from the loop's cache and
        # produced no second image. Refused ids stay in and are dropped by _crops.
        atom_ids = list(dict.fromkeys(metrics.get("get_visual_calls") or []))
        selected.append({**row, "atom_ids": atom_ids, "agent_outcome": outcome,
                         "source_run": source_run_id})
    if limit is not None:
        selected = selected[:limit]

    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "notes": (f"isolated visual ablation of {source_run_id}: the crops get_visual "
                  "served, re-asked with no skeleton, no tool history and no system "
                  "prompt. judge_score is the only comparable metric."),
        "source_run": source_run_id,
        "model": MISTRAL_AGENT_MODEL,
        "prompt": PROMPT,
        "crop_pad_px": pad,
        "n_selected": len(selected),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{run_id}] {len(selected)} queries sélectionnées dans {source_run_id}")

    for stem, dom, pdf_path, doc_rows in _by_document(selected):
        with (run_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as out:
            for row in tqdm(doc_rows, desc=stem, unit="q"):
                out.write(json.dumps(run_one(row, dom, pdf_path, pad=pad),
                                     ensure_ascii=False) + "\n")
                out.flush()

    return score_results(run_id)


# --------------------------------------------------------------------------- #
# Analysis                                                                     #
# --------------------------------------------------------------------------- #
def _outcome_cells(pair: pd.DataFrame) -> pd.Series:
    """The four cells of one 2x2 block plus the two accuracies, over `agent_ok`/`iso_ok`."""
    agent, iso = pair["agent_ok"], pair["iso_ok"]
    return pd.Series({
        "n": len(pair),
        "both": int((agent & iso).sum()),
        "agent_only": int((agent & ~iso).sum()),
        "isolated_only": int((~agent & iso).sum()),
        "neither": int((~agent & ~iso).sum()),
        "accuracy_agent": agent.mean(),
        "accuracy_isolated": iso.mean(),
    })


def isolation_matrix(agent_run_id: str, isolated_run_id: str,
                     column: str = "category") -> pd.DataFrame:
    """Paired agent-vs-isolated contingency table, one block per group.

    Lives here and not in analyze.py: it reads two runs where every other analysis
    function reads one frame, and it only means anything for this experiment.

    The isolated run holds a SUBSET of the agent run's queries, so the join is inner and
    driven by the isolated side; `n` reports what was actually paired. The 2x2 is the four
    count columns, over outcome == "correct" against everything else — the off-diagonal is
    the point, since the two accuracies can be equal while the arms disagree on half the
    queries, and delta_net (= (isolated_only - agent_only) / n) only reports their balance.

    ABSTENTIONS COUNT AS INCORRECT, on both sides, which is what the judge already does.
    Read the two runs' abstention rates beside this table: an arm that swaps a wrong
    answer for an abstention moves nothing in `neither` and is not the same result.

    Args:
        agent_run_id: The source agent run, e.g. "run8".
        isolated_run_id: The ablation run, e.g. "isolated_visual_from_run8".
        column: Column of the isolated run to block on; "category" by default.

    Returns:
        One row per group plus an "ALL" row, counts as ints and rates rounded.
    """
    # Imported here, not at module scope: analyze pulls matplotlib, which the run path has
    # no use for — the mirror image of analyze's own reason for not importing this side.
    from analyze import load_scored

    agent = load_scored(agent_run_id).set_index("query_id")
    isolated = load_scored(isolated_run_id).set_index("query_id")

    pair = (isolated[[column, "outcome"]]
            .join(agent["outcome"].rename("agent_outcome"), how="inner"))
    pair["agent_ok"] = pair["agent_outcome"].eq("correct")
    pair["iso_ok"] = pair["outcome"].eq("correct")

    table = pair.groupby(column)[["agent_ok", "iso_ok"]].apply(_outcome_cells)
    table = pd.concat([table, _outcome_cells(pair).to_frame("ALL").T])

    table["delta_net"] = table["accuracy_isolated"] - table["accuracy_agent"]
    counts = ["n", "both", "agent_only", "isolated_only", "neither"]
    table[counts] = table[counts].astype(int)
    return table.round({c: 3 for c in table.columns if c not in counts})


def dump_crops(run_id: str, out_dir: str | Path | None = None,
               pad: int | None = None) -> int:
    """Write every crop of a run to disk, as the bytes that went (or would go) to the model.

    Reads the run's own `atom_ids` — the ids whose image was actually sent — and re-crops
    them with the same get_visual call, which is deterministic on (atom_id, dom, pdf): at
    the run's own pad, the PNG written here is byte-for-byte what sat in the data: URL.

    Filenames are the diagnosis, readable from a directory listing:
    `{query_id}_{atom_id}_p{page}_{GOLD|off}_pad{N}.png`. GOLD means the crop sits on a
    page the dataset marks as holding the answer — the 0-/1-indexed boundary goes through
    metrics.page_recall, never a hand-written +1. The pad is always in the name, so a
    preview at a pad no run has used can never be mistaken for a record of what was sent.

    Args:
        run_id: A run produced by run_isolated; supplies the atom ids and the gold pages.
        out_dir: Directory to write into, created if absent. Defaults to `crops/` (or
            `crops_pad{N}/` when previewing) inside the run's own directory — a relative
            path would resolve against the caller's cwd, which in a notebook is rarely the
            repo root.
        pad: Crop margin to redraw at. None, the default, uses each row's own
            `crop_pad_px` and reproduces what the model saw. An explicit value is a
            PREVIEW: it lets you look at a wider crop on an existing run's atoms before
            paying for a padded run.

    Returns:
        The number of PNGs written.
    """
    if out_dir is None:
        name = "crops" if pad is None else f"crops_pad{pad}"
        out_dir = RUNS / run_id / name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in
            (RUNS / run_id / SCORED_NAME).read_text(encoding="utf-8").splitlines()
            if line.strip()]

    written = 0
    for _stem, dom, pdf_path, doc_rows in _by_document(rows):
        for row in doc_rows:
            gold = row.get("gold_pages") or []
            # The pad the run actually used unless one was asked for: read back rather
            # than re-typed, so a plain dump cannot drift from what the model saw.
            row_pad = row["agent_metrics"].get("crop_pad_px", 0) if pad is None else pad
            for atom_id in row["agent_metrics"]["atom_ids"]:
                result = get_visual(atom_id, dom, pdf_path, pad=row_pad)
                if "error" in result:
                    continue
                page = result["page"]
                tag = "GOLD" if page_recall(page, list(gold)) == 1.0 else "off"
                name = f"{row['query_id']}_{atom_id}_p{page}_{tag}_pad{row_pad}.png"
                (out_dir / name).write_bytes(base64.b64decode(result["image_b64"]))
                written += 1

    print(f"[{run_id}] {written} crops (pad="
          f"{'row' if pad is None else pad}) -> {out_dir.resolve()}")
    return written
