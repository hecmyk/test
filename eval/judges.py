"""LLM-as-judge, binary, aligned with FinRAGBench-V Table 9.

One call per query, cached on query_id so a re-scored run costs nothing. The cache is a
plain dict shared across the scoring threads: dict get/set is atomic under the GIL, and a
duplicate judgement of the same query would be harmless anyway.
"""

import json
import logging
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config

from config import MISTRAL_API_KEY, MISTRAL_BASE_URL   # noqa: E402

logger = logging.getLogger(__name__)

MODEL = "mistral-medium-3.5-ITG"

client = OpenAI(base_url=MISTRAL_BASE_URL, api_key=MISTRAL_API_KEY)

JUDGE_SYSTEM = """You are evaluating whether a model's answer matches a ground truth answer to a question about a financial document.

Reply with a brief reasoning (1-2 sentences) followed by a final line containing only "Score: 0" or "Score: 1".

Score 1 (correct) if the model's answer conveys the same information as the ground truth:

- Numerical values: format and units do not matter. 1.5M, 1,500,000, and "1.5 million" are equivalent. Currency symbols and thousands separators are ignored. Parenthesized numbers denote negatives ((1,234) = -1234). Accept the model answer if it matches the ground truth at the precision the ground truth is written (e.g. gold "45.2" accepts model "45.15" through "45.25"; gold "1.5M" accepts model values that round to 1.5M).
- Percentages and rates: tolerance is expressed in percentage points, not relative. "5.2%" and "6.2%" differ by 1 point and are NOT equivalent. Accept small differences (≤0.1 point) due to rounding of intermediate values.
- If the question is ambiguous about units (e.g. "revenue growth" could be a raw amount or a percentage) and the model answer and ground truth are numerically compatible under a reasonable interpretation, accept them as equivalent.
- Dates: accept equivalent formats (2023-03-15, March 15 2023, 15/03/2023). Accept a less precise date if the ground truth is less precise.
- Short factual answers (names, entities, single values): must match the ground truth's core content. Minor wording differences and additional correct context are fine.
- Long or descriptive answers: the model's answer must contain all key facts from the ground truth without contradicting any. It may be more verbose or add correct context — that is fine. It may not omit a key fact or introduce a false one.
- Lists: all items from the ground truth must be present. Extra correct items are fine; extra incorrect items make the answer wrong.

Score 0 (wrong) if:
- A numerical value is off by more than the tolerance above.
- A key fact from the ground truth is missing or contradicted.
- The answer introduces false information, even alongside correct content.
- The answer is on the wrong subject entirely.

Output format: reply with a single JSON object and nothing else. No prose before or after, no markdown code fences. Exact structure:
{"reasoning": "one or two sentences", "score": 0 or 1}"""


def load_cache(path: Path) -> dict:
    """Judgements cached so far, keyed by query_id; empty dict when the file is absent."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_cache(cache: dict, path: Path) -> None:
    """Persist the judgement cache."""
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def judge_answer(question: str, gold: str, pred: str | None,
                 query_id: str, cache: dict) -> dict:
    """Judge one answer, reusing the cache when possible.

    An abstention is scored 0 without any call: there is nothing to evaluate, and caching
    it would freeze a non-judgement under a key a later run may want to fill.

    Args:
        question: The question asked.
        gold: Ground truth answer.
        pred: The model's answer, or None (abstention).
        query_id: Cache key.
        cache: Mutated in place with the new judgement, malformed ones excepted.

    Returns:
        `{"score": 0 | 1 | None, "reasoning": str}` — None when the judge's output could
        not be parsed, which is a broken judge and not a wrong answer.
    """
    if pred is None:
        return {"score": 0, "reasoning": "abstention"}
    if query_id in cache:
        return cache[query_id]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n"
                                        f"Ground truth answer: {gold}\n"
                                        f"Model's answer: {pred}"},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()

    # A judge that does not return the asked-for JSON is a broken judge, not a wrong answer:
    # the raw output is kept in `reasoning` under a tagged prefix, and the judgement is left
    # out of the cache so the next scoring retries it.
    try:
        data = json.loads(raw)
        score = data["score"] if data["score"] in (0, 1) else None
        reasoning = data["reasoning"]
        if score is None:
            reasoning = f"invalid_score_value: {raw[:200]}"
            logger.warning(f"Judge returned invalid score for {query_id}: {raw[:200]}")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        score = None
        reasoning = f"json_parse_failed: {raw[:200]}"
        logger.warning(f"Judge output not parseable for {query_id} ({type(e).__name__}): {raw[:200]}")

    result = {"score": score, "reasoning": reasoning}
    if score is not None:
        cache[query_id] = result
    return result
