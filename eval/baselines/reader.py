"""Shared reader contract for every baseline, plus the page-level OCR substrate.

Every baseline (no-context, page-level OCR markdown, oracle-md / oracle-img / oracle-both)
answers through the SAME _call_reader: same system prompt, same abstention policy, same
model, same temperature. Only the context handed to it changes — which is the whole point
of the comparison, so none of the three must be allowed to drift per baseline.

The reader is deliberately tool-free: response_format JSON only, no tools exposed. The DOM
agent's hybrid failure mode (a tool call emitted as prose in `content`) cannot happen here,
so the baselines measure reading, not tool-calling.
"""

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import fitz
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for the imports below

from config import (DPI, MAX_WORKERS, MISTRAL_AGENT_MODEL,  # noqa: E402
                    MISTRAL_API_KEY, MISTRAL_BASE_URL)
from ocr import _MISTRAL_SEM, call_mistral_ocr               # noqa: E402
from pdf_utils import encode_image_to_base64, rasterize_page  # noqa: E402

logger = logging.getLogger(__name__)

client = OpenAI(base_url=MISTRAL_BASE_URL, api_key=MISTRAL_API_KEY)

# Imported, not re-declared: "same model as the agent" is the invariant being tested, so it
# has to break at the config level or not at all.
MODEL = MISTRAL_AGENT_MODEL

ARTIFACTS = Path(__file__).resolve().parents[1] / "finragbench" / "artifacts"
PAGES_OCR_NAME = "baseline_pages_ocr.json"

READER_SYSTEM = """You are answering a question about a financial document, using ONLY the context provided below (document text and/or page images).

Output format: reply with a single JSON object and nothing else. No prose before or after, no markdown code fences. Exact structure:
{"answer": "...", "cited_pages": [0], "source_page": 0, "abstain": false}

- answer: the answer to the question, as short as the question allows. Empty string when you abstain.
- cited_pages: the absolute page numbers of the document that support your answer, as integers. Pages are 0-indexed: the first page of the document is page 0. Take them from the '### Start Page N ###' markers in the context, or from the page label given with each image. NEVER take them from a page number printed inside the page content itself (footers, headers, 'Page X of Y'). Empty list when you abstain.
- source_page: the single page number that best supports your answer, as one integer on the same 0-indexed scale as cited_pages, and normally one of them. When several pages contribute, give the one carrying the evidence you actually answered from. null when you abstain.
- abstain: true if the evidence needed to answer is NOT in the context provided, false otherwise.

Abstention policy. You have no knowledge beyond the context provided. If that context does not contain the evidence needed to answer the question, set abstain to true, answer to "", cited_pages to [] and source_page to null. Do not guess, do not fall back on prior knowledge about the company, the market or the period. Otherwise set abstain to false, answer the question, and cite every page that supports your answer."""


def _render_pages(page_keys: list[int], pages_md: dict[int, str]) -> str:
    """Concatenate pages between their 0-indexed page markers.

    Lives here, next to READER_SYSTEM, because that prompt is what tells the model these
    markers are where page numbers come from: the format and its documentation have to move
    together or the reader starts citing pages it was never shown the numbers of. Shared by
    every baseline that sends text — one definition, so a marker change cannot reach one
    arm and not another.

    Args:
        page_keys: Page indices to render, in the order they must be read.
        pages_md: `{page_idx: markdown}` covering at least `page_keys`.

    Returns:
        The markdown of those pages, each between its Start/End markers.
    """
    return "\n".join(f"### Start Page {i} ###\n{pages_md[i]}\n### End Page {i} ###"
                     for i in page_keys)


def _parse_reader_json(raw: str) -> dict[str, Any] | None:
    """Parse the reader's reply into a dict, or None when nothing valid comes out.

    Defensive on purpose: response_format is advisory on this endpoint, so the reply can
    still arrive wrapped in prose or in a markdown fence. The object is taken from the
    first '{' to the last '}'.

    Args:
        raw: The assistant message content.

    Returns:
        The parsed object, or None.
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _call_reader(question: str, context_md: str = "",
                 images: list | None = None) -> dict[str, Any]:
    """Ask the reader one question over the given context and normalize its answer.

    Args:
        question: The question to answer.
        context_md: Document text handed to the reader; empty for the no-context baseline.
        images: Multimodal content blocks (data URLs) appended to the user message; None
            for the text-only baselines.

    Returns:
        `{"answer", "cited_pages", "source_page", "abstain", "usage_tokens"}`. An
        unparseable reply is logged and returned as an abstention, so a broken reply never
        scores as an answer.
    """
    text = f"Question: {question}"
    if context_md:
        text = f"Document context:\n{context_md}\n\n{text}"
    # String content when there is no image: the list-of-blocks form is only needed for the
    # multimodal baselines, and some shims reject it on a text-only request.
    content: Any = text if not images else [{"type": "text", "text": text}, *images]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": READER_SYSTEM},
                  {"role": "user", "content": content}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    usage = getattr(response, "usage", None)
    usage_tokens = ((getattr(usage, "prompt_tokens", 0) or 0)
                    + (getattr(usage, "completion_tokens", 0) or 0))

    raw = (response.choices[0].message.content or "").strip()
    data = _parse_reader_json(raw)
    if data is None:
        logger.warning("reader output not parseable: %s", raw[:200])
        return {"answer": "", "cited_pages": [], "source_page": None, "abstain": True,
                "usage_tokens": usage_tokens}

    answer = data.get("answer")
    # Pages kept only when they are already integers: a "page 4" string or a float is a
    # reader that did not follow the contract, and coercing it would hide that in the
    # citation metrics. bool is excluded explicitly — it is an int subclass, so True would
    # otherwise pass through as page 1.
    cited = data.get("cited_pages")
    cited = [p for p in cited if isinstance(p, int) and not isinstance(p, bool)] \
        if isinstance(cited, list) else []
    # Same rule for the scalar: this one feeds metrics.page_recall, which does `page + 1`.
    source_page = data.get("source_page")
    if not isinstance(source_page, int) or isinstance(source_page, bool):
        source_page = None
    return {
        "answer": answer if isinstance(answer, str) else "",
        "cited_pages": cited,
        "source_page": source_page,
        "abstain": bool(data.get("abstain")),
        "usage_tokens": usage_tokens,
    }


def _ocr_one_page(pdf_path: str, page_idx: int, dpi: int) -> tuple[int, str | None]:
    """OCR one full page: rasterize, encode, one Mistral OCR call.

    Args:
        pdf_path: Path to the source PDF.
        page_idx: 0-indexed page to read.
        dpi: Rasterization resolution.

    Returns:
        `(page_idx, markdown)`, `markdown` None when the call failed or came back malformed.
    """
    doc = fitz.open(pdf_path)
    page_img = rasterize_page(doc[page_idx], dpi)
    doc.close()

    with _MISTRAL_SEM:
        result, _, error = call_mistral_ocr(encode_image_to_base64(page_img))
    if error is None and not (isinstance(result, dict) and result.get("pages")):
        error = "malformed response: missing/empty 'pages'"
    if error is not None:
        logger.warning("baseline OCR failed on page %d of %s: %s",
                       page_idx, Path(pdf_path).name, error)
        return page_idx, None
    return page_idx, result["pages"][0]["markdown"]


def ocr_pages(pdf_path: str, dpi: int = DPI) -> dict[int, str]:
    """Page-level OCR markdown of a whole document, 0-indexed, cached on disk.

    One OCR call per full page — no layout, no stacking: this is the substrate of the
    page-level baseline and of the oracle variants, i.e. what the pipeline is compared
    against. Cached in eval/artifacts/{stem}/baseline_pages_ocr.json; a second call only
    pays for the pages missing from that file.

    Args:
        pdf_path: Path to the source PDF.
        dpi: Rasterization resolution.

    Returns:
        `{page_idx: markdown}` for every page of the PDF, ordered. A page whose OCR failed
        comes back as `""` and is left OUT of the cache, so the next call retries it.
    """
    cache_path = ARTIFACTS / Path(pdf_path).stem / PAGES_OCR_NAME
    pages: dict[int, str] = {}
    if cache_path.exists():
        pages = {int(k): v for k, v in
                 json.loads(cache_path.read_text(encoding="utf-8")).items()}

    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    doc.close()

    missing = [i for i in range(n_pages) if i not in pages]
    if not missing:
        return dict(sorted(pages.items()))

    failed = set()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_ocr_one_page, pdf_path, i, dpi) for i in missing]
        for fut in as_completed(futures):
            page_idx, markdown = fut.result()
            if markdown is None:
                failed.add(page_idx)
            pages[page_idx] = markdown or ""

    pages = dict(sorted(pages.items()))
    # Failures excluded from what is written: caching one as "" would make every later call
    # consider that page done and never read it again.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({i: md for i, md in pages.items() if i not in failed},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    return pages
