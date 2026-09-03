"""Full-context baseline: map over page batches, reduce only when batches disagree.

The pipeline's claim is that a DOM lets the model reach the right page without reading the
whole document. The control is this: hand the reader every page of the OCR markdown, in
raw order, and see what it costs and how well it does.

Batching is by page index, not by tokens — a token-packed batch would make the batch
boundaries depend on OCR verbosity, so two documents of the same length would not be cut
the same way and the cost figures would stop being comparable. 50 pages with 3 of overlap,
the overlap being there so an answer straddling a boundary is whole in at least one batch.

The reduce is fired on disagreement COUNT alone (>= 2 batches answered), never on the
content of the answers: deciding whether two answers say the same thing is exactly the
judge's job, and doing it here would bake the judge's tolerance into the baseline it is
supposed to score.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for the import below

from eval.baselines.reader import _call_reader, _render_pages   # noqa: E402

logger = logging.getLogger(__name__)

# Input-side overflow guard. The footgun is that an over-long prompt does not raise: it is
# silently truncated, and the reader then abstains on evidence it was never shown. v0 warns
# and sends anyway — splitting automatically would change the batch geometry mid-run and
# make the cost columns incomparable between documents.
CONTEXT_WINDOW = 128_000
OVERFLOW_RATIO = 0.8
_CHARS_PER_TOKEN = 4        # rough, and deliberately so: the guard is an alarm, not a budget

_REDUCE_INSTRUCTION = """The document was read in several passes, each covering a different range of pages. Below is each pass that produced an answer, with the pages it cited and the single page it read the evidence from. They may agree, disagree, or answer from different parts of the document.

Pick the candidate best supported by the pages it cites, and answer the question with it. Carry over that candidate's pages and its source page. Abstain only if no candidate actually answers the question."""


def _estimate_tokens(text: str) -> int:
    """Rough token count of a prompt, at ~4 characters per token."""
    return len(text) // _CHARS_PER_TOKEN


def _batch_page_keys(page_keys: list[int], batch_size: int,
                     overlap: int) -> list[list[int]]:
    """Cut the page keys into overlapping batches, in the order given.

    Args:
        page_keys: Page indices, in the order they must be read.
        batch_size: Pages per batch.
        overlap: Pages shared by two consecutive batches.

    Returns:
        One list of page keys per batch.

    Raises:
        ValueError: When `overlap` is not smaller than `batch_size` (the stride would be
            zero or negative and the batches would never advance).
    """
    stride = batch_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than batch_size ({batch_size})")

    batches = []
    start = 0
    while start < len(page_keys):
        batches.append(page_keys[start:start + batch_size])
        # Stop on the batch that reaches the end, otherwise the stride would emit a last
        # batch entirely contained in the previous one — a paid call and a duplicate
        # candidate that would fire the reduce on a document that never disagreed.
        if start + batch_size >= len(page_keys):
            break
        start += stride
    return batches


def _render_candidates(candidates: list[dict[str, Any]]) -> str:
    """Render the map answers as the reduce call's context.

    Each candidate carries its source_page as well as its cited_pages: the reduce has to
    emit a source_page of its own, and without the candidates' own scalars it would have to
    re-derive one from a list it never read the pages of.

    Answers are passed through verbatim — no normalization, no dedup, no ordering by
    frequency: any of those would be the harness deciding which answers are "the same",
    which is the judge's call, not the baseline's.

    Args:
        candidates: The map results that did not abstain.

    Returns:
        The context markdown for the reduce call.
    """
    blocks = [f"Candidate {n} (cited pages: {c['cited_pages']}, "
              f"source page: {c['source_page']}): {c['answer']}"
              for n, c in enumerate(candidates, 1)]
    return _REDUCE_INSTRUCTION + "\n\n" + "\n".join(blocks)


def answer_batched(question: str, pages_md: dict[int, str],
                   batch_size: int = 50, overlap: int = 3) -> dict[str, Any]:
    """Answer one question over a whole document, by batched map then reduce-on-conflict.

    Every batch is read (the abstaining ones included) — that is the cost being measured.
    The reduce then fires only when at least two batches answered; a single answer is passed
    through with no second call, and no answer at all is recorded as an abstention.

    The returned record has the shape run_benchmark.run_one writes and score_results reads.
    The five query-side fields it cannot know (query_id, gold_answer, gold_pages,
    answer_type, category) are present and set to None for the caller to fill from the query
    dict; `question` is already filled.

    Args:
        question: The question to answer.
        pages_md: `{page_idx: markdown}` from reader.ocr_pages, 0-indexed. Iterated in the
            order given, with no defensive re-sort: the page order IS the reading order.
        batch_size: Pages per batch.
        overlap: Pages shared by two consecutive batches.

    Returns:
        The result row. `pred_source_page` is the reader's own scalar source_page — the
        field metrics.page_recall expects — not a page picked out of cited_pages by this
        module; the full citation list survives in `agent_metrics["pred_cited_pages"]`.
    """
    started = time.perf_counter()
    batches = _batch_page_keys(list(pages_md), batch_size, overlap)

    candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    reader_tokens = 0
    n_map_calls = 0

    for batch in batches:
        context = _render_pages(batch, pages_md)
        estimated = _estimate_tokens(context)
        if estimated > OVERFLOW_RATIO * CONTEXT_WINDOW:
            logger.warning(
                "batch pages %d-%d is ~%d tokens, over %.0f%% of the %d window: the prompt "
                "will be truncated silently, not rejected",
                batch[0], batch[-1], estimated, OVERFLOW_RATIO * 100, CONTEXT_WINDOW)

        n_map_calls += 1
        try:
            result = _call_reader(question, context_md=context)
        except Exception as e:
            # The batch is lost, the ones already paid for are not: a document is several
            # calls, and re-running the whole map for one failed batch is real money.
            errors.append(f"map[{batch[0]}-{batch[-1]}]: {type(e).__name__}: {e}")
            continue
        reader_tokens += result["usage_tokens"]
        if not result["abstain"]:
            candidates.append(result)

    n_reduce_calls = 0
    if not candidates:
        answer, cited, source_page, abstain = "", [], None, True
    elif len(candidates) == 1:
        # Passthrough: one batch answered, there is nothing to arbitrate and a reduce call
        # here would only add a paraphrase step between the reader and the judge.
        only = candidates[0]
        answer, cited, source_page, abstain = (only["answer"], only["cited_pages"],
                                               only["source_page"], False)
    else:
        n_reduce_calls = 1
        try:
            result = _call_reader(question, context_md=_render_candidates(candidates))
            reader_tokens += result["usage_tokens"]
            answer, cited, source_page, abstain = (result["answer"], result["cited_pages"],
                                                   result["source_page"], result["abstain"])
        except Exception as e:
            # Recorded as an abstention rather than falling back to a candidate: picking one
            # would report a consensus the reduce never reached.
            errors.append(f"reduce: {type(e).__name__}: {e}")
            answer, cited, source_page, abstain = "", [], None, True

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
            "n_ocr_pages": len(pages_md),
            "n_map_calls": n_map_calls,
            "n_reduce_calls": n_reduce_calls,
            "reduce_fired": n_reduce_calls > 0,
            "pred_cited_pages": cited,
            "duration_s": round(time.perf_counter() - started, 2),
        },
        "error": "; ".join(errors) if errors else None,
    }
