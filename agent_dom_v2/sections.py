"""Section id resolution and the pre-loaded context of a subagent invocation.

build_section_maps walks the dom tree once per document; resolve_section_ids turns what
the orchestrator or the search named into deliverable ids (unknown ids recorded, never
raised; ancestors absorb their descendants); build_subagent_context renders them whole,
each under its breadcrumb. Pure functions over the dom — no client, no prompt.
"""

import logging
from typing import Any

from dom_tools import _heading_text, render_section

logger = logging.getLogger(__name__)


def build_section_maps(dom: dict) -> dict[str, Any]:
    """Section objects, ancestor chains and breadcrumbs, in one pre-order walk.

    Built once per document. The section tree carries no parent pointer, so the chain is
    accumulated on the way down instead of being searched for on demand: a delegation then
    resolves in O(k x depth) rather than one full tree scan per candidate.

    The root is `tree` itself — section_id "root", children under "sections" and not
    "subsections" — and is left out of every chain: it names the document, not a position
    inside it.

    Args:
        dom: build_dom output ({header, index, chrome, tree}).

    Returns:
        {"section", "ancestors", "crumb"}, each keyed by section_id: the section object,
        its ancestor ids outermost-first, and its titles joined into a breadcrumb.
    """
    sections: dict[str, dict] = {}
    ancestors: dict[str, tuple[str, ...]] = {}
    crumb: dict[str, str] = {}

    # Explicit stack, reversed so siblings come out in document order. A section is always
    # popped after its ancestors, so their labels are in `sections` when the crumb is built.
    stack = [(section, ()) for section in reversed(dom["tree"]["sections"])]
    while stack:
        section, chain = stack.pop()
        section_id = section["section_id"]
        sections[section_id] = section
        ancestors[section_id] = chain
        crumb[section_id] = " > ".join(
            [_heading_text(sections[ancestor]) for ancestor in chain]
            + [_heading_text(section)])
        for sub in reversed(section["subsections"]):
            stack.append((sub, chain + (section_id,)))

    return {"section": sections, "ancestors": ancestors, "crumb": crumb}


def resolve_section_ids(section_ids: Any, maps: dict[str, Any], metrics: dict[str, Any],
                        step: int, tool: str) -> list[str]:
    """Deliverable section ids: known, deduplicated by ancestor, in the order given.

    Three passes, in this order:
      1. unknown ids are dropped and recorded in invalid_tool_args — a hallucinated id is a
         measurement, not a crash, and render_section would only turn it into an error dict
         nobody reads;
      2. duplicates collapse onto their first occurrence;
      3. a candidate having an ancestor in the same set is dropped: render_section renders
         a section AND its whole subtree, so keeping the ancestor loses nothing the
         descendant carried. Disjoint siblings both survive.

    The order handed in is preserved throughout — it is the BM25 rank on the search path,
    and it decides what the subagent reads first. It is never re-sorted.

    Note that step 3 reduces the NUMBER of sections while it can raise their token weight:
    an ancestor two levels up expands to everything under it. That is why the invocation's
    real prompt tokens (subagent_stats[i]["tokens"]) are logged rather than capped.

    Args:
        section_ids: The ids as the model or the search sent them; any type.
        maps: build_section_maps output.
        metrics: Shared metrics dict, mutated in place.
        step: Current orchestrator round, for the refusal record.
        tool: Tool the ids came from, for the refusal record.

    Returns:
        The ids to pre-load, in delivery order; possibly empty.
    """
    candidates = section_ids if isinstance(section_ids, list) else []

    known: list[str] = []
    for section_id in candidates:
        if isinstance(section_id, str) and section_id in maps["section"]:
            if section_id not in known:
                known.append(section_id)
        else:
            metrics["invalid_tool_args"].append(
                {"tool": tool, "args": {"section_id": section_id},
                 "reason": "unknown_section_id", "step": step, "tier": "orchestrator"})

    kept = set(known)
    return [section_id for section_id in known
            if not set(maps["ancestors"][section_id]) & kept]


def build_subagent_context(section_ids: list[str], dom: dict,
                           maps: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """The pre-loaded sections, each under the breadcrumb of its position.

    The subagent never sees the skeleton, so the breadcrumb is what restores the one
    navigational fact the markdown alone cannot carry: where in the document this text
    sits. The sections are wrapped in id-carrying tags so a report can name the section a
    quote came from instead of having to infer it.

    Nothing is truncated and nothing is capped: the sections go in whole. An oversized
    context has to become visible in the invocation's real prompt tokens, on a real
    document, rather than be hidden by a cut — and a cut would sever a closing page band and leave a quote
    with no `### Start Page N ###` above it, which is the one thing the page transport
    cannot survive.

    Args:
        section_ids: Ids to deliver, already resolved, in delivery order.
        dom: build_dom output.
        maps: build_section_maps output.

    Returns:
        `(context_markdown, loaded)` — the blob handed to the subagent, and one record per
        section {"section_id", "crumb", "markdown"} kept for the key_figures check and the
        debug dump.
    """
    blocks: list[str] = []
    loaded: list[dict[str, Any]] = []

    for section_id in section_ids:
        markdown = render_section(section_id, dom)
        if not isinstance(markdown, str):
            # resolve_section_ids already filtered unknown ids, so this is unreachable
            # unless the dom and the maps came from two different builds.
            logger.warning("%s resolved but did not render; dom and maps disagree",
                           section_id)
            continue
        crumb = maps["crumb"][section_id]
        blocks.append(f'<section id="{section_id}" path="{crumb}">\n{markdown}\n</section>')
        loaded.append({"section_id": section_id, "crumb": crumb, "markdown": markdown})

    return "\n\n".join(blocks), loaded
