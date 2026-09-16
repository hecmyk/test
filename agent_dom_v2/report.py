"""The report — the subagent's finish-tool, and the shape the orchestrator reads.

build_report_tool is what the model is told; _normalize_report is what is read back. The
two are deliberately NOT one validator: the schema is advisory on this endpoint, and a
value the model sent wrong (a page that is not an int, a figure that is not a string) has
to reach the checks as sent, so it becomes a verdict rather than a coercion or a rejected
report. _descriptive_fallback is the report of an invocation that produced none.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .protocol import tool_schema

# Subagent tool. Its sections arrive pre-loaded, so get_visual and report are all it has.
REPORT = "report"

class Evidence(BaseModel):
    """One passage the subagent answered from.

    Schéma annoncé au modèle, jamais appliqué à sa réponse — voir _normalize_report pour la
    lecture. `pages` is the only channel a page number has to the final answer: the
    subagent copies it from the `### Start Page N ###` marker it alone can see.
    """
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(description="id of the <section> the quote came from.")
    atom_id: str | None = Field(
        default=None,
        description="pN_eM of the table or figure it came from, or null for running text.")
    quote: str = Field(
        description="Exact substring of the section, copied character for character.")
    pages: list[int] = Field(
        description=("Pages the quote sits on, copied from the nearest ### Start Page N ### "
                     "marker above it. 0-indexed."))


class SectionContent(BaseModel):
    """What one delivered section holds — the navigational signal, filled even on a miss.

    Schéma annoncé au modèle, jamais appliqué à sa réponse — voir _normalize_report pour la
    lecture. key_figures is the verbatim channel; check_key_figures measures it.
    """
    model_config = ConfigDict(extra="forbid")

    section_id: str
    topics: list[str] = Field(description="What the section covers, in your words.")
    key_figures: list[str] = Field(
        description=("Figures copied VERBATIM with their units, never rounded or "
                     "paraphrased."))
    visuals_present: list[str] = Field(
        default_factory=list,
        description=("Tables and figures seen, each named by its atom id plus a short "
                     "caption."))


class Report(BaseModel):
    """The subagent's report — the arguments of its finish-tool.

    Schéma annoncé au modèle, jamais appliqué à sa réponse — voir _normalize_report pour la
    lecture. section_contents is required while candidate_answer is not: the report
    describes what was read even when nothing answered, because that description is what
    the orchestrator navigates on. A bare "not found" would give it nothing.
    """
    model_config = ConfigDict(extra="forbid")

    candidate_answer: str | None = Field(
        default=None,
        description=("Your answer to the sub-question, or null when what you were given "
                     "does not answer it."))
    confidence: Literal["high", "medium", "low"] = Field(
        description="How sure you are of candidate_answer.")
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="The passages you answered from; empty when candidate_answer is null.")
    section_contents: list[SectionContent] = Field(
        description="One entry per section you were given. Never empty.")
    recommendation: Literal["answer_ready", "search_elsewhere", "try_sections"] = Field(
        description="What the navigator should do next.")
    try_section_ids: list[str] = Field(
        default_factory=list,
        description=("Section ids something you read points at; only with recommendation "
                     "try_sections."))


def build_report_tool() -> dict[str, Any]:
    """The subagent's finish-tool definition: Report's schema under the tool's description.

    Returns:
        The tool definition, OpenAI function-calling shape.
    """
    return tool_schema(Report, REPORT, (
        "End your work and send everything back. Call it exactly once. "
        "section_contents carries one entry per section you were given, whether or "
        "not it answered."))


def _normalize_report(payload: Any) -> dict[str, Any]:
    """Coerce raw report arguments into the canonical report shape.

    The tool schema is advisory on this endpoint, so every field is rebuilt rather than
    trusted: a missing one becomes its neutral value and a list that came back as something
    else becomes empty. Values themselves are kept exactly as the model sent them — no
    coercion, so a malformed page stays visible to the checks instead of being silently
    dropped here.

    Args:
        payload: The raw arguments object of the report tool call.

    Returns:
        The report, every key present.
    """
    payload = payload if isinstance(payload, dict) else {}

    def _dicts(key: str) -> list[dict[str, Any]]:
        value = payload.get(key)
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    try_ids = payload.get("try_section_ids")
    return {
        "candidate_answer": payload.get("candidate_answer"),
        "confidence": payload.get("confidence"),
        "evidence": _dicts("evidence"),
        "section_contents": _dicts("section_contents"),
        "recommendation": payload.get("recommendation"),
        "try_section_ids": [i for i in try_ids if isinstance(i, str)] if isinstance(try_ids, list) else [],
    }


def _descriptive_fallback(section_ids: list[str]) -> dict[str, Any]:
    """The report of an invocation that produced none, naming the sections it was given.

    Used when the model never called report and the forced call did not comply either. It
    carries no description — there is nothing to describe without a model — but it still
    tells the orchestrator which sections are now covered, which is the one thing that
    keeps it from delegating them again.

    Args:
        section_ids: The sections that were delivered.

    Returns:
        A report in canonical shape.
    """
    return {
        "candidate_answer": None,
        "confidence": "low",
        "evidence": [],
        "section_contents": [{"section_id": section_id, "topics": [], "key_figures": [],
                              "visuals_present": []} for section_id in section_ids],
        "recommendation": "search_elsewhere",
        "try_section_ids": [],
    }
