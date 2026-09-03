"""Oracle baselines: the gold pages are handed to the reader, in one of three forms.

This is the ceiling, not a system: retrieval is removed from the problem entirely, so what
is left is the reader's ability to read. The three modes answer the question the pipeline
actually raises — is the OCR markdown enough, does the model need the pixels, or does it
need both:

    md      the OCR markdown of the gold pages, same page markers as everywhere else
    image   the gold pages rasterized at 150 DPI, no text at all
    both    the markdown AND the images of the same pages

One condition is one call and one record. The three are NOT merged into a single row: a
mode is a separate experimental arm and has to be scored as one, which is what
`agent_metrics["mode"]` is for.

page_recall is meaningless against these rows — the pages were given, not found.
pred_source_page and pred_cited_pages are kept for schema parity with the other baselines,
and filtering the oracle_* arms out of the recall aggregates is analyze.py's job.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for the imports below

from config import DPI                                        # noqa: E402
from eval.baselines.reader import _call_reader, _render_pages  # noqa: E402
from pdf_utils import encode_image_to_base64, rasterize_page   # noqa: E402

logger = logging.getLogger(__name__)

MODES = ("md", "image", "both")


def _render_page_images(pdf_path: str, gold_pages: list[int],
                        dpi: int = DPI) -> list[dict[str, Any]]:
    """Rasterize the gold pages into labelled multimodal content blocks.

    Each image is preceded by its own '### Start Page N ###' text block — the same marker
    the markdown modes use. Without it the reader is shown pixels it cannot name, and
    cited_pages / source_page become unanswerable in image mode.

    Args:
        pdf_path: Path to the source PDF.
        gold_pages: 0-indexed pages to render.
        dpi: Rasterization resolution.

    Returns:
        Alternating text-marker and image_url blocks, in the order of `gold_pages`.
    """
    blocks: list[dict[str, Any]] = []
    doc = fitz.open(pdf_path)
    try:
        for page_idx in gold_pages:
            b64 = encode_image_to_base64(rasterize_page(doc[page_idx], dpi))
            blocks.append({"type": "text", "text": f"### Start Page {page_idx} ###"})
            blocks.append({"type": "image_url",
                           "image_url": {"url": f"data:image/png;base64,{b64}"}})
    finally:
        doc.close()
    return blocks


def answer_oracle(question: str, gold_pages: list[int], mode: str,
                  pages_md: dict[int, str], pdf_path: str,
                  dpi: int = DPI) -> dict[str, Any]:
    """Answer one question from its gold pages only, in one reader call.

    No batching, no map, no reduce: the context is small by construction, so the whole
    arbitration machinery of the full-context baseline has nothing to arbitrate.

    Args:
        question: The question to answer.
        gold_pages: The supporting pages, **0-indexed**. The dataset's `from_pages` is
            1-indexed (see metrics.page_recall) — the caller converts, and this function
            trusts what it is given.
        mode: One of "md", "image", "both".
        pages_md: `{page_idx: markdown}` from reader.ocr_pages, already OCR'd once for the
            whole document. Unused in "image" mode.
        pdf_path: Path to the source PDF. Only read in "image" and "both".
        dpi: Rasterization resolution for the image modes.

    Returns:
        A record with the exact shape fullctx.answer_batched returns, minus the
        fullctx-only map/reduce counters and plus `agent_metrics["mode"]`.

    Raises:
        ValueError: On an unknown `mode`, or when a gold page is absent from `pages_md` in
            a mode that needs the markdown.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    started = time.perf_counter()
    needs_md = mode in ("md", "both")

    # Checked up front rather than through a KeyError from the renderer: a gold page missing
    # from pages_md is either an out-of-range index or a page OCR never returned, and both
    # are worth naming. A page that OCR'd to nothing is PRESENT with an empty string and is
    # deliberately not caught here — an empty gold page is a finding, not a crash.
    if needs_md:
        missing = [p for p in gold_pages if p not in pages_md]
        if missing:
            raise ValueError(f"gold pages absent from pages_md: {missing} "
                             f"(pages_md covers 0..{max(pages_md) if pages_md else -1})")

    context_md = _render_pages(gold_pages, pages_md) if needs_md else ""
    images = _render_page_images(pdf_path, gold_pages, dpi) if mode in ("image", "both") else None

    try:
        result = _call_reader(question, context_md=context_md, images=images)
    except Exception as e:
        answer, cited, source_page, abstain = "", [], None, True
        reader_tokens, error = 0, f"{type(e).__name__}: {e}"
    else:
        answer, cited, source_page, abstain = (result["answer"], result["cited_pages"],
                                               result["source_page"], result["abstain"])
        reader_tokens, error = result["usage_tokens"], None

    return {
        "query_id": None,           # caller fills the five query-side fields below
        "question": question,
        "gold_answer": None,
        "gold_pages": None,
        "answer_type": None,
        "category": None,
        # None, not "": analyze.load_scored reads a null pred_answer as an abstention, and
        # judges.judge_answer scores it 0 without spending a call.
        "pred_answer": None if abstain else answer,
        "pred_quote": None,         # the reader contract has no quote field
        "pred_source_page": source_page,
        "agent_metrics": {
            "reader_tokens": reader_tokens,
            # What the row actually consumed of the page-level OCR: nothing in image mode,
            # where the reader is shown pixels and no OCR text is spent at all.
            "n_ocr_pages": 0 if mode == "image" else len(gold_pages),
            "pred_cited_pages": cited,
            "duration_s": round(time.perf_counter() - started, 2),
            "mode": mode,
        },
        "error": error,
    }
