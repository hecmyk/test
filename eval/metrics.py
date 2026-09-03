"""Phase 1 metrics — pure functions over strings and page numbers.

No I/O, no fitz, no DOM knowledge: every input is a plain value supplied by the caller.
page_recall owns the only index-convention boundary of the harness (agent 0-indexed vs
dataset 1-indexed), so no other module ever holds a shifted page number.
"""

import re
import unicodedata
from collections import Counter

# A comma is a thousands separator only when followed by EXACTLY three digits. Without the
# (?!\d) guard this would also eat the European decimal comma in "2,5" -> "25", a factor 10
# on a numeric answer, silent and scored as a model error.
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_ARTICLES = re.compile(r"\b(?:a|an|the)\b")
_LEADING_CURRENCY = re.compile(r"^[$€£]\s*")
_WHITESPACE = re.compile(r"\s+")
_PERIPHERAL = " .,;:!?"


def normalize_text(s: str | None) -> str:
    """Canonical form of an answer for comparison.

    NFKC (which also folds non-breaking and thin spaces, common in financial amounts),
    casefold, thousands separators, percent and leading currency signs removed, peripheral
    punctuation trimmed, isolated articles dropped, whitespace collapsed.

    Deliberately NOT handled: scale equivalence ($2.5M vs 2,500,000), date formats, and
    aliases (FY19 vs fiscal 2019). Those change what the benchmark measures.

    Args:
        s: Raw text, or None.

    Returns:
        The normalized text; "" when `s` is None.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = _THOUSANDS.sub("", s)
    s = s.replace("%", "")
    s = s.strip(_PERIPHERAL)
    s = _LEADING_CURRENCY.sub("", s)
    s = _ARTICLES.sub(" ", s)
    return _WHITESPACE.sub(" ", s).strip()


def token_f1(pred: str | None, gold: str) -> float:
    """Token-level F1 between a prediction and the gold answer, after normalization.

    Args:
        pred: Predicted answer, or None (abstention).
        gold: Gold answer.

    Returns:
        F1 in [0, 1]; 0.0 when either side normalizes to nothing.
    """
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def as_pages(value: int | list | None) -> list[int]:
    """Page citations in canonical form: a list of ints, possibly empty.

    Absorbs the three shapes a run can carry — the scalar `pred_source_page` of the runs
    predating the list schema, the list the agent emits now, and the None or [] of an
    abstention — so no caller has to know which one it is holding. Non-integer entries are
    dropped rather than coerced: a "page 4" string is a contract the agent broke, and
    coercing it would hide that in the citation metrics. bool is excluded explicitly — it
    is an int subclass, so True would otherwise pass through as page 1.

    Args:
        value: A page number, a list of them, or None.

    Returns:
        The pages, in the order given.
    """
    if isinstance(value, list):
        return [p for p in value if isinstance(p, int) and not isinstance(p, bool)]
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    return []


def page_recall(pred_pages: int | list[int] | None, gold_pages: list[int]) -> float:
    """1.0 when at least one cited page is among the gold pages.

    THE index boundary of the harness: cited pages are 0-indexed (agent and DOM
    convention), `gold_pages` is 1-indexed (dataset convention), and the +1 is applied here
    and nowhere else. An abstention scores 0.0 rather than being excluded: on this
    intra-doc oracle the answer is always in the document, so not citing is a failure to
    locate — and excluding it would reward abstaining on the hard queries.

    A bare int is accepted as well as a list (see as_pages), and on a single citation the
    two give the same score — the numbers of a run made before the list schema stay
    comparable with those made after it.

    Args:
        pred_pages: Pages cited by the agent, 0-indexed: a list, a bare int, or None.
        gold_pages: Gold pages, 1-indexed.

    Returns:
        1.0 or 0.0.
    """
    return 1.0 if any(p + 1 in gold_pages for p in as_pages(pred_pages)) else 0.0


def test_metrics() -> None:
    """Run the metric self-checks; raises AssertionError on the first failure."""
    assert normalize_text(None) == ""
    assert normalize_text("The United States.") == "united states"
    assert normalize_text("$1,234,567") == "1234567"
    assert normalize_text("2,5 million") == "2,5 million"   # decimal comma preserved
    assert normalize_text("  12.5%  ") == "12.5"
    print("OK: normalize_text")

    assert token_f1("the united states", "United States") == 1.0
    assert token_f1(None, "united states") == 0.0
    assert token_f1("united states", "") == 0.0
    assert token_f1("canada", "united states") == 0.0
    assert 0.0 < token_f1("the united states of america", "united states") < 1.0
    print("OK: token_f1")

    assert as_pages(None) == []
    assert as_pages(5) == [5]
    assert as_pages([]) == []
    assert as_pages([0, 3]) == [0, 3]
    assert as_pages(True) == []                  # bool is an int subclass
    assert as_pages(["p3", 4, None]) == [4]      # non-ints dropped, never coerced
    print("OK: as_pages")

    assert page_recall(15, [16]) == 1.0          # 0-indexed 15 == 1-indexed 16
    assert page_recall(16, [16]) == 0.0
    assert page_recall(None, [16]) == 0.0
    assert page_recall(156, [157, 158, 159]) == 1.0
    assert page_recall([15], [16]) == 1.0        # same score as the scalar it replaces
    assert page_recall([], [16]) == 0.0
    assert page_recall([3, 15], [16]) == 1.0     # one hit is enough
    assert page_recall([3, 7], [16]) == 0.0
    print("OK: page_recall")
