"""Score one benchmark run: token F1, page recall, LLM judge.

Reads eval/runs/{run_id}/all.jsonl and writes scored.jsonl next to it, with the metrics
appended to each row. The judge calls are what the thread pool is for; the two other
metrics are local computation.

The judgement cache lives IN the run directory: it is keyed on query_id alone, so a cache
shared between runs would hand run N the judgement of run N-1 for a prediction that has
changed.
"""

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # eval/finragbench/, for run_benchmark
sys.path.insert(0, str(_HERE.parent))   # eval/, for judges and metrics

from judges import judge_answer, load_cache, save_cache   # noqa: E402
from metrics import page_recall, token_f1                 # noqa: E402
from run_benchmark import AGGREGATE_NAME, RUNS, SCORED_NAME, aggregate_run   # noqa: E402

logger = logging.getLogger(__name__)


def score_results(run_id: str) -> Path:
    """Score every row of a run and write eval/runs/{run_id}/scored.jsonl.

    Args:
        run_id: The iteration to score.

    Returns:
        Path of the scored JSONL.
    """
    logger.info("scoring run %s", run_id)
    aggregate_run(run_id)   # always re-aggregated: a stale all.jsonl would score a subset

    run_dir = RUNS / run_id
    cache_path = run_dir / "judge_cache.json"
    rows = [json.loads(line)
            for line in (run_dir / AGGREGATE_NAME).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    cache = load_cache(cache_path)

    def score(row: dict) -> dict:
        judgement = judge_answer(row["question"], row["gold_answer"], row["pred_answer"],
                                 row["query_id"], cache)
        return {
            **row,
            "token_f1": round(token_f1(row["pred_answer"], row["gold_answer"]), 4),
            "page_recall": page_recall(row.get("pred_source_pages",
                                              row.get("pred_source_page")),
                                       row["gold_pages"]),
            "judge_score": judgement["score"],
            "judge_reasoning": judgement["reasoning"],
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        scored = list(tqdm(pool.map(score, rows), total=len(rows), unit="q"))

    save_cache(cache, cache_path)
    out = run_dir / SCORED_NAME
    with out.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("scored %d rows -> %s", len(scored), out)
    print(f"[{run_id}] {len(scored)} lignes -> {out}")
    return out
