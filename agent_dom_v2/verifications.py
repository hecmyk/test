"""Non-repairing checks on a report, and the verdict records they produce.

check_key_figures and check_page_ranges test what the subagent reported against what it
was given, and RECORD the outcome — they never correct a figure or a page. That is the
discipline inherited from the single-phase agent's invalid_tool_args: a hallucination is a
measurement. The records go to a caller-owned sink (the runners write them to
runs/{run_id}/verifications.jsonl, one line per verdict); only the four aggregates of
_aggregate_verifications enter the metrics.

The two record shapes are Pydantic models, and the file format is their discriminated
union: a line of verifications.jsonl is `VerdictAdapter.validate_json(line)`. They are
built by THIS code, never from the model's output, so their validation cannot fail — the
values the model sent (section_id, figure, page) are typed Any on purpose and stored
exactly as sent, because a page that is not an int is a `not_an_int` verdict, not a
coercion.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from eval.metrics import normalize_text


class KeyFigureVerdict(BaseModel):
    """One key_figure of a report, tested against the markdown of its claimed section."""
    model_config = ConfigDict(extra="forbid")

    query_id: str | None = None   # stamped by the runner; the agent does not know it
    check: Literal["key_figure"] = "key_figure"
    round: int
    section_id: Any
    figure: Any
    verdict: Literal["exact", "normalized", "absent", "unknown_section"]


class PageRangeVerdict(BaseModel):
    """One cited page of a report, tested against [page_start, page_end] of its section."""
    model_config = ConfigDict(extra="forbid")

    query_id: str | None = None   # stamped by the runner; the agent does not know it
    check: Literal["page_range"] = "page_range"
    round: int
    section_id: Any
    page: Any
    verdict: Literal["in_range", "out_of_range", "not_an_int", "no_page_range",
                     "unknown_section"]


# A line of verifications.jsonl, told apart on `check`.
Verdict = Annotated[KeyFigureVerdict | PageRangeVerdict, Field(discriminator="check")]
VerdictAdapter: TypeAdapter[KeyFigureVerdict | PageRangeVerdict] = TypeAdapter(Verdict)


def check_key_figures(report: dict[str, Any], loaded: list[dict[str, Any]],
                      verifications: list[Verdict], round_index: int) -> None:
    """Test every key_figure against the markdown it was read from; record, never repair.

    key_figures verbatim is the linchpin of a numeric cross-page answer — "EUR 4.2bn" can
    be combined with a figure from another report, "about four billion" cannot — and it is
    a prompt constraint with nothing enforcing it. So it is measured: an exact substring
    first, then the same test through metrics.normalize_text, which absorbs currency marks,
    thousands separators and case without touching the digits.

    A "normalized" verdict is not a failure; it is a figure the model re-spelled. Only
    "absent" says the figure is not in what it read.

    Args:
        report: The canonical report.
        loaded: build_subagent_context's per-section records.
        verifications: Sink of verdict records, appended in place, one KeyFigureVerdict
            per figure.
        round_index: Orchestrator round the invocation belongs to.
    """
    markdown = {record["section_id"]: record["markdown"] for record in loaded}
    normalized: dict[str, str] = {}

    for entry in report["section_contents"]:
        section_id = entry.get("section_id")
        source = markdown.get(section_id)
        figures = entry.get("key_figures")
        for figure in figures if isinstance(figures, list) else []:
            if not isinstance(figure, str) or not figure.strip():
                verdict = "absent"
            elif source is None:
                verdict = "unknown_section"
            elif figure in source:
                verdict = "exact"
            else:
                if section_id not in normalized:
                    normalized[section_id] = normalize_text(source)
                needle = normalize_text(figure)
                verdict = "normalized" if needle and needle in normalized[section_id] else "absent"
            verifications.append(KeyFigureVerdict(
                round=round_index, section_id=section_id, figure=figure, verdict=verdict))


def check_page_ranges(report: dict[str, Any], section_ids: list[str], maps: dict[str, Any],
                      verifications: list[Verdict], round_index: int) -> None:
    """Test every cited page against the territory of the section it is claimed from.

    The subagent is the only tier that sees the `### Start Page N ###` markers, so its
    pages are the only ones that can reach the final answer. Deriving them here instead —
    from the section's own page_start/page_end, which this code knows — would be the easy
    path and the wrong one: page_recall is pure recall, so a derived range would score
    close to 1.0 mechanically and destroy comparability with the single-phase run.

    page_start/page_end already cover a section's FULL territory, subsections included, so
    the range is the right test for a delivered ancestor too. Recorded, never repaired.

    Args:
        report: The canonical report.
        section_ids: The sections that were delivered.
        maps: build_section_maps output.
        verifications: Sink of verdict records, appended in place, one PageRangeVerdict
            per cited page.
        round_index: Orchestrator round the invocation belongs to.
    """
    delivered = set(section_ids)

    for entry in report["evidence"]:
        section_id = entry.get("section_id")
        pages = entry.get("pages")
        for page in pages if isinstance(pages, list) else []:
            if section_id not in delivered:
                verdict = "unknown_section"
            elif not isinstance(page, int) or isinstance(page, bool):
                verdict = "not_an_int"
            else:
                section = maps["section"][section_id]
                start, end = section["page_start"], section["page_end"]
                verdict = ("no_page_range" if start is None else
                           "in_range" if start <= page <= end else "out_of_range")
            verifications.append(PageRangeVerdict(
                round=round_index, section_id=section_id, page=page, verdict=verdict))


def _aggregate_verifications(verifications: list[Verdict],
                             metrics: dict[str, Any]) -> None:
    """Fold the verdict records into the four flat keys of the metrics.

    "grounded" is exact OR normalized: a re-spelled figure is still the document's figure.
    Every other verdict of a check counts in its denominator, unknown_section included —
    a figure claimed from a section that was never delivered is not grounded.

    Args:
        verifications: The run's verdict records.
        metrics: Shared metrics dict, mutated in place.
    """
    figures = [v for v in verifications if v.check == "key_figure"]
    pages = [v for v in verifications if v.check == "page_range"]
    metrics["n_key_figures_checked"] = len(figures)
    metrics["key_figures_grounded_rate"] = (
        sum(v.verdict in ("exact", "normalized") for v in figures) / len(figures)
        if figures else None)
    metrics["n_pages_checked"] = len(pages)
    metrics["pages_in_range_rate"] = (
        sum(v.verdict == "in_range" for v in pages) / len(pages) if pages else None)
