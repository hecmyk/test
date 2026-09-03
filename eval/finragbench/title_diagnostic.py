"""Read-only diagnostic: recurring and duplicated titles across the FinRAGBench-V DOMs.

Measures how many title atoms are repetitions of one another — running document headers
("Annual Report 2023" on every page) and running section titles (repeated at the top of
each page of a section) — so thresholds can be calibrated before any fix is written.

Reads the schema build_dom actually produces: {header, index, chrome, tree}, where index
maps atom-id to {id, page, bbox, label, text, ...} and header carries page_count, the
per-page dimensions and the section count. Chrome atoms (header/footer/number) are NOT in
the index, so what is measured here is exactly what creates sections.

Nothing modifies a DOM.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DOM_FILENAME = "doc_dom.json"

HEADER_KEY = "header"
ATOMS_KEY = "index"
TEXT_KEY = "text"
PAGE_KEY = "page"
LABEL_KEY = "label"
BBOX_KEY = "bbox"          # [x1, y1, x2, y2] in raster pixels, same space as header.pages
TITLE_LABELS = {"doc_title", "paragraph_title"}   # = config.DOM_TITLE_LABELS

MIN_OCCURRENCES = 2
TRUNCATE = 60


def normalize_title(text: str) -> str:
    """Group key for a title: lowercased, digit-free, whitespace-collapsed.

    Digits are dropped so "Report 2023 page 12" and "Report 2023 page 13" land on the same
    key. That same rule also merges "Note 12" with "Note 13", which must NOT be deduplicated
    — the n_variants column is what exposes those over-merges.

    Args:
        text: Raw title text.

    Returns:
        The normalized key, possibly empty (a page number normalizes to "").
    """
    return re.sub(r"\s+", " ", re.sub(r"\d+", "", text.lower())).strip()


def diagnose_doc(dom_path: Path) -> pd.DataFrame:
    """One row per distinct normalized title of one document.

    Every title is returned, singletons included: the unique-vs-total ratio and the saving
    estimate both need them. Filter with `df[df.n_occurrences >= 2]` for the recurring ones.

    Args:
        dom_path: Path of a doc_dom.json.

    Returns:
        Columns doc_stem, title, n_chars, n_occurrences, n_pages, first_page, last_page,
        span, span_ratio, density, y_rel_median, n_variants, labels, n_pages_doc,
        n_sections_doc. Empty when the document has no page or no title.
    """
    dom = json.loads(dom_path.read_text(encoding="utf-8"))
    header = dom.get(HEADER_KEY, {})
    n_pages_doc = header.get("page_count", 0)
    if not n_pages_doc:
        return pd.DataFrame(columns=_COLUMNS)

    heights = {p["index"]: p["height"] for p in header.get("pages", [])}
    n_sections_doc = header.get("stats", {}).get("n_sections", 0)

    # Empty keys are dropped: a title normalizing to "" (a bare number) would otherwise
    # collapse every such atom into one meaningless group.
    groups: dict[str, list[tuple]] = defaultdict(list)
    for atom in dom.get(ATOMS_KEY, {}).values():
        if atom.get(LABEL_KEY) not in TITLE_LABELS:
            continue
        raw = str(atom.get(TEXT_KEY) or "").strip()
        key = normalize_title(raw)
        if key:
            groups[key].append((atom[PAGE_KEY], raw, _y_rel(atom, heights), atom[LABEL_KEY]))

    rows = [_row(dom_path.parent.name, key, items, n_pages_doc, n_sections_doc)
            for key, items in groups.items()]
    return pd.DataFrame(rows, columns=_COLUMNS)


def diagnose_all(artifacts_dir: Path = ARTIFACTS_DIR) -> pd.DataFrame:
    """Run diagnose_doc over every document and concatenate.

    Args:
        artifacts_dir: Directory holding one subfolder per document.

    Returns:
        The per-document tables stacked, sorted by n_occurrences descending.
    """
    # Empty frames are dropped rather than concatenated: a document with no title would
    # otherwise drag every column's dtype to object.
    frames = [df for path in sorted(artifacts_dir.glob(f"*/{DOM_FILENAME}"))
              if not (df := diagnose_doc(path)).empty]
    if not frames:
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("n_occurrences", ascending=False, ignore_index=True)


def summarize(df: pd.DataFrame) -> None:
    """Print the aggregate report: duplication ratio, saving estimate, distributions.

    Args:
        df: diagnose_all output, singletons included.
    """
    rec = df[df["n_occurrences"] >= MIN_OCCURRENCES]
    total_atoms = int(df["n_occurrences"].sum())
    total_sections = int(df.groupby("doc_stem")["n_sections_doc"].first().sum())
    saving = int((rec["n_occurrences"] - 1).sum())

    print(f"{df['doc_stem'].nunique()} documents, {total_sections} sections")
    print(f"{total_atoms} title atoms -> {len(df)} distinct titles "
          f"({1 - len(df) / total_atoms:.1%} are repetitions)")
    print(f"{len(rec)} recurring titles ({len(rec) / len(df):.1%} of distinct titles)")
    print(f"\nSaving: {total_sections} -> {total_sections - saving} sections "
          f"(-{saving}, {saving / total_sections:.1%}) if one occurrence is kept per title\n")

    print("span_ratio quantiles on recurring titles (1.0 = spans the whole document)")
    print(rec["span_ratio"].quantile([.1, .25, .5, .75, .9, 1.0]).round(3).to_string())

    buckets = pd.cut(rec["span_ratio"], [-.01, .1, .3, .6, .9, 1.01],
                     labels=["<0.1", "0.1-0.3", "0.3-0.6", "0.6-0.9", ">0.9"])
    print("\nsaving contributed by each span_ratio bucket")
    print(rec.assign(bucket=buckets, dup=rec["n_occurrences"] - 1)
             .groupby("bucket", observed=False)
             .agg(n_titles=("dup", "size"), saved=("dup", "sum"),
                  density_med=("density", "median"), y_rel_med=("y_rel_median", "median"))
             .to_string())

    print("\nn_occurrences distribution on recurring titles")
    print(pd.cut(rec["n_occurrences"], [1, 2, 3, 5, 10, 25, 10 ** 6],
                 labels=["2", "3", "4-5", "6-10", "11-25", ">25"])
          .value_counts().sort_index().to_string())


_COLUMNS = ["doc_stem", "title", "n_chars", "n_occurrences", "n_pages", "first_page",
            "last_page", "span", "span_ratio", "density", "y_rel_median", "n_variants",
            "labels", "n_pages_doc", "n_sections_doc"]


def _y_rel(atom: dict, heights: dict[int, float]) -> float | None:
    """Top y of an atom as a fraction of its page height; None without a usable bbox."""
    bbox = atom.get(BBOX_KEY)
    height = heights.get(atom.get(PAGE_KEY))
    return float(bbox[1]) / height if bbox and height else None


def _row(stem: str, key: str, items: list[tuple], n_pages_doc: int,
         n_sections_doc: int) -> dict:
    """Feature row for one normalized title of one document."""
    pages = sorted({page for page, _, _, _ in items})
    ys = [y for _, _, y, _ in items if y is not None]
    span = pages[-1] - pages[0]
    return {
        "doc_stem": stem,
        "title": key[:TRUNCATE],
        "n_chars": len(key),
        "n_occurrences": len(items),
        "n_pages": len(pages),
        "first_page": pages[0],
        "last_page": pages[-1],
        "span": span,
        "span_ratio": round(span / n_pages_doc, 4),
        "density": round(len(pages) / (span + 1), 4),
        "y_rel_median": round(pd.Series(ys).median(), 4) if ys else float("nan"),
        "n_variants": len({raw for _, raw, _, _ in items}),
        "labels": "|".join(sorted({label for _, _, _, label in items})),
        "n_pages_doc": n_pages_doc,
        "n_sections_doc": n_sections_doc,
    }
