"""Stage 3 of the official protocol: the short answers scored by the official rules.

Joins the run (all.jsonl), the extractions (extracted.jsonl) and the gold (samples.json)
on query_id, then hands each row to the vendored eval_score. Nothing is reimplemented —
the whole point of this module is to feed official/eval_score.py rows of the shape it
expects and to stay out of the way.

A row whose extraction could not be parsed gets NO `score` key and is left out of what is
scored: eval_acc_and_f1 filters on that key, so an unreadable extraction lowers nothing on
its own. It is reported as a count instead — a silent 0.0 would be indistinguishable from
a wrong answer and would move the published quantity.
"""

import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # eval/mmlongbench/, for the imports below

from dataset import samples_by_query_id                              # noqa: E402
from official.eval_score import eval_acc_and_f1, eval_score, show_results   # noqa: E402
from render import NOT_ANSWERABLE                                    # noqa: E402
from run_mm import AGGREGATE_NAME, EXTRACTED_NAME, RUNS, SCORED_NAME  # noqa: E402

RESULTS_NAME = "results.txt"

# Fields eval_score, eval_acc_and_f1 and show_results read off a row. evidence_pages and
# evidence_sources must arrive as the RAW strings of samples.json ("[3, 5]"): show_results
# calls eval() on them itself.
GOLD_FIELDS = ("answer", "answer_format", "doc_type", "evidence_pages", "evidence_sources")


def _read_jsonl(path: Path) -> list[dict]:
    """Every non-empty line of a JSONL file, parsed."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def score_run(run_id: str) -> Path:
    """Score every extracted row of a run and write runs/{run_id}/scored.jsonl.

    Args:
        run_id: The iteration to score.

    Returns:
        Path of the scored JSONL.
    """
    run_dir = RUNS / run_id
    extracted = {e["query_id"]: e for e in _read_jsonl(run_dir / EXTRACTED_NAME)}
    gold = samples_by_query_id()

    scored, unparsed, missing = [], [], []
    for row in _read_jsonl(run_dir / AGGREGATE_NAME):
        query_id = row["query_id"]
        entry = extracted.get(query_id)
        if entry is None:
            missing.append(query_id)
            continue

        sample = gold[query_id]
        out = {**row, **{key: sample[key] for key in GOLD_FIELDS},
               "pred": entry["pred"], "parse_ok": entry["parse_ok"]}
        if not entry["parse_ok"]:
            unparsed.append(query_id)          # pas de cle "score": exclu du calcul
        else:
            out["score"] = eval_score(sample["answer"], entry["pred"],
                                      sample["answer_format"])
        scored.append(out)

    out_path = run_dir / SCORED_NAME
    with out_path.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # show_results REASSIGNS evidence_pages/evidence_sources on the rows it is given, so it
    # gets shallow copies: scored.jsonl is already written, and a second call in the same
    # session would otherwise eval() a list and raise.
    evaluated = [dict(row) for row in scored if "score" in row]
    show_results(evaluated, show_path=str(run_dir / RESULTS_NAME))

    acc, f1 = eval_acc_and_f1(evaluated)
    print(f"[{run_id}] scorées {len(evaluated)} / {len(scored)} lignes -> {out_path}")
    print(f"[{run_id}] Acc {acc:.4f} | F1 {f1:.4f}")
    if unparsed:
        print(f"[{run_id}] {len(unparsed)} extractions illisibles, hors calcul: "
              f"{unparsed[:5]}{' ...' if len(unparsed) > 5 else ''}")
    if missing:
        print(f"[{run_id}] {len(missing)} lignes sans extraction: "
              f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")
    return out_path


def sanity_counters(run_id: str) -> None:
    """Print the two counts to read BEFORE trusting the F1.

    The official F1 builds its precision denominator on `pred != "Not answerable"`, a
    case-sensitive match on the exact literal, with no trailing period. A variant slips
    through as a wrong affirmative answer — and because get_clean_string strips peripheral
    punctuation, that same row can still score 1.0 in the accuracy. The symptom is
    therefore acc and F1 disagreeing, not both dropping.

    "Fail to answer" is the extractor's own sentinel for "the model could not read the
    document". The scoring treats it as a wrong affirmative answer. It should be marginal
    (< 2%); a high count means draft_to_analysis is producing something that reads as an
    inability to read rather than as an answer.

    Args:
        run_id: The iteration to inspect.
    """
    entries = _read_jsonl(RUNS / run_id / EXTRACTED_NAME)
    preds = [e["pred"] for e in entries if e["pred"] is not None]
    if not preds:
        print(f"[{run_id}] aucune extraction lisible")
        return

    variants = Counter(p for p in preds if "answerable" in p.lower())
    exact = variants.get(NOT_ANSWERABLE, 0)
    print(f"[{run_id}] abstentions extraites: {sum(variants.values())} "
          f"dont {exact} au littéral exact {NOT_ANSWERABLE!r}")
    for value, count in variants.most_common():
        flag = "" if value == NOT_ANSWERABLE else "   <-- VARIANTE, ne matche pas le F1"
        print(f"    {count:5d}  {value!r}{flag}")

    n_fail = sum(1 for p in preds if p.strip() == "Fail to answer")
    print(f"[{run_id}] 'Fail to answer': {n_fail} / {len(preds)} "
          f"({n_fail / len(preds):.1%}) — doit rester sous 2%")
