"""Phase 1 QA benchmark runner.

Runs run_dom_agent(mode="qa") over every query whose document has a frozen doc_ocr.json,
and appends one JSON line per query to eval/runs/{run_id}/{stem}.jsonl. Every iteration of
the benchmark gets its own run_id, so a new run never overwrites the previous one and the
two stay comparable side by side.

A query already present in the run's file is skipped. That is a crash net, not a resume
mode: to start over, pass a new run_id.

A query belongs to the document whose folder name prefixes its query-id — no assumption
on the id format beyond that separator.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))  # repo root, for the imports below

from agent_dom import run_dom_agent          # noqa: E402
from dom_build import build_dom              # noqa: E402
from dom_render import build_outline_xml     # noqa: E402
from pdf_utils import page_dims_from_pdf     # noqa: E402

logger = logging.getLogger(__name__)

ROOT = Path("/home/datacamp/data/FinRAGBench-V")
PDFS = ROOT / "pdfs"
QUERIES = ROOT / "queries_en_clean.json"

ARTIFACTS = _HERE / "artifacts"
RUNS = _HERE / "runs"
AGGREGATE_NAME = "all.jsonl"
SCORED_NAME = "scored.jsonl"    # produit par score_results, dans le meme dossier

FIELD_DEFINITIONS = {"qa": "Answer the question grounded in the document."}
MAX_STEPS = 10


def _init_run(run_id: str, notes: str) -> Path:
    """The run directory, created with its metadata on first use.

    run_metadata.json is written ONCE per run_id: a later call on the same run is a
    continuation, and rewriting it would move created_at away from the code that actually
    produced the earliest rows. `notes` is therefore only read on the call that creates
    the run — and since nothing else identifies the code version, it is the only record
    of what this run changes.

    Args:
        run_id: Name of this benchmark iteration.
        notes: Free text describing what this run changes.

    Returns:
        The eval/runs/{run_id} directory.
    """
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = run_dir / "run_metadata.json"
    if not metadata.exists():
        metadata.write_text(json.dumps({
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "notes": notes,
            "MAX_STEPS": MAX_STEPS,
            "FIELD_DEFINITIONS": FIELD_DEFINITIONS,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def load_or_build_dom(doc_dir: Path) -> tuple[dict, str]:
    """DOM and XML skeleton of a document, built from doc_ocr.json on first use.

    Both are persisted next to doc_ocr.json (doc_dom.json, skeleton.xml) and reused as-is
    afterwards. Building is NOT free or reproducible: build_dom's default title_leveler
    calls an LLM, so a rebuilt DOM can nest its sections differently and shift every
    section_id. Freezing them is what keeps a continued run comparable to the one before.

    Args:
        doc_dir: The eval/artifacts/{stem} directory.

    Returns:
        `(dom, xml_skeleton)`.
    """
    dom_path = doc_dir / "doc_dom.json"
    xml_path = doc_dir / "skeleton.xml"

    if dom_path.exists():
        dom = json.loads(dom_path.read_text(encoding="utf-8"))
    else:
        stem = doc_dir.name
        doc_ocr = json.loads((doc_dir / "doc_ocr.json").read_text(encoding="utf-8"))
        dom = build_dom(doc_ocr, page_dims_from_pdf(PDFS / f"{stem}.pdf"),
                        doc_id=stem, pdf_name=f"{stem}.pdf")
        dom_path.write_text(json.dumps(dom, ensure_ascii=False), encoding="utf-8")

    # Checked separately from the DOM: the two writes are not atomic, so a crash between
    # them leaves a DOM with no skeleton — rebuilt here instead of raising.
    if xml_path.exists():
        skeleton = xml_path.read_text(encoding="utf-8")
    else:
        skeleton = build_outline_xml(dom)
        xml_path.write_text(skeleton, encoding="utf-8")

    return dom, skeleton


def pdf_for(stem: str) -> Path | None:
    """The document's source PDF, or None when it is not on disk.

    get_visual needs the file itself; doc_dom.json only records its name. A PDF that is
    absent costs that document its visual channel and nothing else — run_dom_agent then
    exposes the text-only tool set it had before, and the run goes on.

    Args:
        stem: The document folder name, which is also the PDF's stem.

    Returns:
        The path, or None when the file is missing.
    """
    pdf_path = PDFS / f"{stem}.pdf"
    if pdf_path.exists():
        return pdf_path
    logger.warning("%s: %s not found, get_visual disabled for this document",
                   stem, pdf_path)
    return None


def run_one(dom: dict, skeleton: str, query: dict, pdf_path: Path | None = None) -> dict:
    """Answer one query and build its result row; an agent failure becomes `error`.

    `pdf_path` defaults to None, which keeps the call text-only: the row shape is the same
    either way, so a run with the visual channel off stays comparable to one with it on.
    """
    started = time.perf_counter()
    try:
        draft, metrics = run_dom_agent(dom, skeleton, FIELD_DEFINITIONS,
                                       max_steps=MAX_STEPS, mode="qa", question=query["query"],
                                       pdf_path=pdf_path)
        answer, error = draft["qa"], None
    except Exception as e:
        answer, metrics, error = {}, {}, f"{type(e).__name__}: {e}"
    metrics["wall_clock_s"] = round(time.perf_counter() - started, 2)

    return {
        "query_id": query["query-id"],
        "question": query["query"],
        "gold_answer": query["answer"],
        "gold_pages": [int(p) for p in str(query["from_pages"]).split(",")],
        "answer_type": query["answer_type"],
        "category": query["category"],
        "pred_answer": answer.get("answer"),
        "pred_quote": answer.get("quote"),
        "pred_source_pages": answer.get("source_pages"),
        "agent_metrics": metrics,
        "error": error,
    }


def run_benchmark(run_id: str, limit: int | None = None, notes: str = "") -> None:
    """Run every pending query of one benchmark iteration, one JSONL file per document.

    Args:
        run_id: Name of this iteration; results land in eval/runs/{run_id}/.
        limit: Stop after N queries in total (pilot runs); None runs them all.
        notes: What this run changes, recorded in run_metadata.json on creation.
    """
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    run_dir = _init_run(run_id, notes)
    done_total = 0

    # doc_ocr.json is the real prerequisite: the DOM is derived from it on first use.
    for ocr_path in sorted(ARTIFACTS.glob("*/doc_ocr.json")):
        doc_dir = ocr_path.parent
        results = run_dir / f"{doc_dir.name}.jsonl"

        done = set()
        if results.exists():
            done = {json.loads(line)["query_id"]
                    for line in results.read_text(encoding="utf-8").splitlines() if line.strip()}

        pending = [q for q in queries
                   if q["query-id"].startswith(doc_dir.name + "_") and q["query-id"] not in done]
        if limit is not None:
            pending = pending[:limit - done_total]
        if not pending:
            continue

        # Loaded or built once per document, not once per query. The PDF is resolved here
        # too: its presence is a property of the document, not of the query.
        dom, skeleton = load_or_build_dom(doc_dir)
        pdf_path = pdf_for(doc_dir.name)

        print(f"[{run_id}] [{doc_dir.name}] {len(pending)} queries, {len(done)} déjà faites")

        # Appended and flushed per query: a crash keeps every answer already paid for.
        with results.open("a", encoding="utf-8") as out:
            bar = tqdm(pending, desc=doc_dir.name, unit="q")
            for query in bar:
                bar.set_postfix_str(query["query-id"])
                out.write(json.dumps(run_one(dom, skeleton, query, pdf_path),
                                     ensure_ascii=False) + "\n")
                out.flush()

        done_total += len(pending)
        if limit is not None and done_total >= limit:
            break


def aggregate_run(run_id: str) -> Path:
    """Concatenate every per-document JSONL of a run into eval/runs/{run_id}/all.jsonl.

    Args:
        run_id: The iteration to aggregate.

    Returns:
        Path of the written aggregate.
    """
    run_dir = RUNS / run_id
    out = run_dir / AGGREGATE_NAME

    # Skipping the run's own outputs: they live among the per-document files, so a second
    # call would otherwise fold the previous aggregate -- or the previous scored.jsonl,
    # which is a superset of it -- back in and duplicate every row.
    derived = {AGGREGATE_NAME, SCORED_NAME}
    lines = [line
             for path in sorted(run_dir.glob("*.jsonl")) if path.name not in derived
             for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{run_id}] {len(lines)} lignes -> {out}")
    return out
