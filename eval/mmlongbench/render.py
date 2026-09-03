"""Turn a submit_draft entry into the free-form "Analysis" the official extractor reads.

The official protocol is extract-then-eval: a model writes prose, an extractor reduces it
to a short answer, and only that short answer is scored. Our agent already answers in
structured form, so this is the adapter — and it stays HERE, at extraction time, so the
run itself keeps the structured shape the run log is built on.

Deliberately minimal: only `answer` crosses over. `quote` and `source_pages` stay in the
JSONL for analysis and take no part in the scoring, because anything added to the
Analysis is something the extractor can misread.
"""

from typing import Any

# The literal the official scoring matches, case-sensitive and without trailing period:
# eval_acc_and_f1 builds its precision denominator on `pred != "Not answerable"`, so a
# variant would silently count an abstention as a wrong affirmative answer.
NOT_ANSWERABLE = "Not answerable"


def draft_to_analysis(draft: dict[str, Any]) -> str:
    """The Analysis string for one draft entry.

    Args:
        draft: A submit_draft entry, i.e. `{answer, quote, source_pages}`. Only `answer`
            is read.

    Returns:
        The answer as written by the agent, or the exact `Not answerable` literal when it
        abstained. An empty or blank answer counts as an abstention: the tool schema is
        advisory on this endpoint, so the agent can return "" where the prompt asks for
        null, and an empty Analysis would make the extractor invent something.
    """
    answer = draft.get("answer")
    if not str(answer or "").strip():
        return NOT_ANSWERABLE
    return str(answer)
