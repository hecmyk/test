"""MMLongBench-Doc QA runner — the FinRAGBench-V runner, repointed.

Same contract as eval/finragbench/run_benchmark.py: one JSONL line per question under
eval/mmlongbench/runs/{run_id}/{stem}.jsonl, one file per document, a query already
present in the file is skipped. That is a crash net, not a resume mode: to start over,
pass a new run_id.

Two deliberate differences with the FinRAGBench runner. The row carries NO gold — answer,
answer_format, evidence_pages and evidence_sources stay in samples.json and are joined by
query_id in score_mm.py, so the runner knows nothing about how it will be scored. And the
question-to-document match is an equality on an explicit `stem` field rather than a prefix
test on the query id.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))   # repo root, for the imports below
sys.path.insert(0, str(_HERE))              # eval/mmlongbench/, for dataset

from agent_dom import run_dom_agent          # noqa: E402
from dataset import MM_DOCS, load_samples    # noqa: E402
from dom_build import build_dom              # noqa: E402
from dom_render import build_outline_xml     # noqa: E402
from pdf_utils import page_dims_from_pdf     # noqa: E402

logger = logging.getLogger(__name__)

ARTIFACTS = _HERE / "artifacts"
RUNS = _HERE / "runs"
AGGREGATE_NAME = "all.jsonl"
EXTRACTED_NAME = "extracted.jsonl"   # produit par extract.py, dans le meme dossier
SCORED_NAME = "scored.jsonl"         # produit par score_mm.py, dans le meme dossier

# The benchmark's own prompt bank: same wording as FinRAGBench's minus the financial
# framing, plus the numeric-format rule the rule-based scoring needs. Loaded here and
# handed to run_dom_agent, which otherwise reads the FinRAGBench one at import.
PROMPTS_PATH = _HERE / "dom_prompts_mm.yaml"
with PROMPTS_PATH.open(encoding="utf-8") as f:
    PROMPTS_MM = yaml.safe_load(f)

FIELD_DEFINITIONS = {"qa": "Answer the question grounded in the document."}
MAX_STEPS = 10


def _init_run(run_id: str, notes: str) -> Path:
    """The run directory, created with its metadata on first use.

    run_metadata.json is written ONCE per run_id: a later call on the same run is a
    continuation, and rewriting it would move created_at away from the code that actually
    produced the earliest rows. Since nothing else identifies the code version, `notes` is
    the only record of what this run changes.

    Args:
        run_id: Name of this benchmark iteration.
        notes: Free text describing what this run changes.

    Returns:
        The eval/mmlongbench/runs/{run_id} directory.
    """
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = run_dir / "run_metadata.json"
    if not metadata.exists():
        metadata.write_text(json.dumps({
            "run_id": run_id,
            "benchmark": "MMLongBench-Doc",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "notes": notes,
            "MAX_STEPS": MAX_STEPS,
            "FIELD_DEFINITIONS": FIELD_DEFINITIONS,
            "prompts": PROMPTS_PATH.name,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def load_or_build_dom(doc_dir: Path) -> tuple[dict, str]:
    """DOM and XML skeleton of a document, built from doc_ocr.json on first use.

    Both are persisted next to doc_ocr.json (doc_dom.json, skeleton.xml) and reused as-is
    afterwards. Building is NOT free or reproducible: build_dom's default title_leveler
    calls an LLM, so a rebuilt DOM can nest its sections differently and shift every
    section_id. Freezing them is what keeps a continued run comparable to the one before.

    Args:
        doc_dir: The eval/mmlongbench/artifacts/{stem} directory.

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
        dom = build_dom(doc_ocr, page_dims_from_pdf(MM_DOCS / f"{stem}.pdf"),
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
    exposes the text-only tool set, and the run goes on.

    Args:
        stem: The document folder name, which is also the PDF's stem.

    Returns:
        The path, or None when the file is missing.
    """
    pdf_path = MM_DOCS / f"{stem}.pdf"
    if pdf_path.exists():
        return pdf_path
    logger.warning("%s: %s not found, get_visual disabled for this document",
                   stem, pdf_path)
    return None


def run_one(dom: dict, skeleton: str, sample: dict,
            pdf_path: Path | None = None) -> dict:
    """Answer one question and build its result row; an agent failure becomes `error`.

    `pred_source_pages` is written EXACTLY as the model sent it, with no coercion: the
    schema is advisory on this endpoint, so a scalar or a malformed entry is a fact about
    the run and metrics.as_pages normalizes it at read time.

    Args:
        dom: The document's DOM.
        skeleton: Its XML skeleton.
        sample: One entry of dataset.load_samples.
        pdf_path: The source PDF, or None to keep the call text-only.

    Returns:
        The result row, gold-free.
    """
    started = time.perf_counter()
    try:
        draft, metrics = run_dom_agent(dom, skeleton, FIELD_DEFINITIONS,
                                       max_steps=MAX_STEPS, mode="qa",
                                       question=sample["question"],
                                       pdf_path=pdf_path, prompts=PROMPTS_MM)
        answer, error = draft["qa"], None
    except Exception as e:
        answer, metrics, error = {}, {}, f"{type(e).__name__}: {e}"
    metrics["wall_clock_s"] = round(time.perf_counter() - started, 2)

    return {
        "query_id": sample["query_id"],
        "doc_id": sample["doc_id"],
        "stem": sample["stem"],
        "question": sample["question"],
        "pred_answer": answer.get("answer"),
        "pred_quote": answer.get("quote"),
        "pred_source_pages": answer.get("source_pages"),
        "agent_metrics": metrics,
        "error": error,
    }


def run_mm(run_id: str, limit: int | None = None, notes: str = "") -> None:
    """Run every pending question of one benchmark iteration, one JSONL file per document.

    Args:
        run_id: Name of this iteration; results land in eval/mmlongbench/runs/{run_id}/.
        limit: Stop after N questions in total (pilot runs); None runs them all.
        notes: What this run changes, recorded in run_metadata.json on creation.
    """
    samples = load_samples()
    run_dir = _init_run(run_id, notes)
    done_total = 0

    # doc_ocr.json is the real prerequisite: the DOM is derived from it on first use.
    for ocr_path in sorted(ARTIFACTS.glob("*/doc_ocr.json")):
        doc_dir = ocr_path.parent
        results = run_dir / f"{doc_dir.name}.jsonl"

        done = set()
        if results.exists():
            done = {json.loads(line)["query_id"]
                    for line in results.read_text(encoding="utf-8").splitlines()
                    if line.strip()}

        pending = [s for s in samples
                   if s["stem"] == doc_dir.name and s["query_id"] not in done]
        if limit is not None:
            pending = pending[:limit - done_total]
        if not pending:
            continue

        # Loaded or built once per document, not once per question. The PDF is resolved
        # here too: its presence is a property of the document, not of the question.
        dom, skeleton = load_or_build_dom(doc_dir)
        pdf_path = pdf_for(doc_dir.name)

        print(f"[{run_id}] [{doc_dir.name}] {len(pending)} questions, {len(done)} déjà faites")

        # Appended and flushed per question: a crash keeps every answer already paid for.
        with results.open("a", encoding="utf-8") as out:
            bar = tqdm(pending, desc=doc_dir.name[:30], unit="q")
            for sample in bar:
                bar.set_postfix_str(sample["query_id"][-8:])
                out.write(json.dumps(run_one(dom, skeleton, sample, pdf_path),
                                     ensure_ascii=False) + "\n")
                out.flush()

        done_total += len(pending)
        if limit is not None and done_total >= limit:
            break


def aggregate_run(run_id: str) -> Path:
    """Concatenate every per-document JSONL of a run into runs/{run_id}/all.jsonl.

    Args:
        run_id: The iteration to aggregate.

    Returns:
        Path of the written aggregate.
    """
    run_dir = RUNS / run_id
    out = run_dir / AGGREGATE_NAME

    # Skipping the run's own outputs: they live among the per-document files, so a second
    # call would otherwise fold the previous aggregate — or extracted/scored, supersets of
    # it — back in and duplicate every row.
    derived = {AGGREGATE_NAME, EXTRACTED_NAME, SCORED_NAME}
    lines = [line
             for path in sorted(run_dir.glob("*.jsonl")) if path.name not in derived
             for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{run_id}] {len(lines)} lignes -> {out}")
    return out
