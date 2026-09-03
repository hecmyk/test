"""DOM agent tools — the read tools and the OpenAI schemas exposed to the model.

render_section turns a section of a build_dom output into markdown, build_search_index
builds the BM25 keyword index over its atoms, and the three build_*_tool functions declare
the corresponding OpenAI tool definitions. No client and no prompt: agent_dom.py owns the
loop, this module owns what the loop hands to the model.

The Python function is render_section, but the name advertised to the model stays
GET_SECTION ("get_section"): the model asks for a section, it gets it rendered.

Search is deliberately lexical (BM25, local, deterministic) rather than dense: the corpus
is a single already-selected document, so there is no cross-document vocabulary drift for
embeddings to bridge, and a dense index would put a network call and a non-reproducible
ranking inside the tool path. It complements get_section, it does not replace it — a hit
names the SECTION to read, never the answer.
"""

import logging
import string
from typing import Any

from rank_bm25 import BM25Okapi

from config import DOM_TITLE_LABELS
from dom_render import clean_text, truncate

logger = logging.getLogger(__name__)

# Tool names as the model sees them. Shared by the schemas and by the loop's dispatch so
# the two cannot drift apart.
GET_SECTION = "get_section"
SEARCH = "search"
SUBMIT_DRAFT = "submit_draft"

# Per-field property schema advertised inside submit_draft's `fields`. DRAFT_SCHEMA is the
# KYC extraction shape (the historical, and still default, one); QA_SCHEMA is the
# question-answering shape. agent_dom picks one from its `mode` and hands the SAME dict to
# build_submit_draft_tool and _normalize_draft, so what is advertised and what is read back
# cannot drift apart.
DRAFT_SCHEMA: dict[str, Any] = {
    "value": {
        "type": ["string", "null"],
        "description": "The extracted value, or null if absent from the document.",
    },
    "source_section_id": {
        "type": "string",
        "description": "Id of the section the value was read in.",
    },
}

QA_SCHEMA: dict[str, Any] = {
    "answer": {
        "type": ["string", "null"],
        "description": "The answer to the question, or null if the document does not answer it.",
    },
    "quote": {
        "type": ["string", "null"],
        "description": "Verbatim substring of the retrieved section supporting the answer, "
                       "or null when there is no answer.",
    },
    "source_pages": {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Numbers N of the '### Start Page N ###' markers covering the text "
                       "the answer was read from (0-indexed), in reading order. Empty "
                       "list when there is no answer.",
    },
}


# --------------------------------------------------------------------------- #
# Section walk                                                                 #
# --------------------------------------------------------------------------- #
def _iter_sections(sections: list[dict], depth: int = 0):
    """Yield (section, depth) over a section list in pre-order; depth is 0-based."""
    for section in sections:
        yield section, depth
        yield from _iter_sections(section["subsections"], depth + 1)


def _collect_descendant_sections(section_id: str,
                                 tree: dict) -> list[tuple[dict, int]] | None:
    """Return (section, relative_depth) for the subtree rooted at section_id, or None.

    Relative depth is 0 for the requested section itself, so it maps straight onto the
    markdown heading level. Order is the pre-order walk order, i.e. reading order.
    """
    root = next((s for s, _ in _iter_sections(tree["sections"])
                 if s["section_id"] == section_id), None)
    if root is None:
        return None
    return list(_iter_sections([root]))


def all_section_ids(tree: dict) -> set[str]:
    """Every section id of a dom tree, s_toc included.

    Args:
        tree: The `tree` part of a build_dom output.

    Returns:
        The set of addressable section ids.
    """
    return {s["section_id"] for s, _ in _iter_sections(tree["sections"])}


# --------------------------------------------------------------------------- #
# Markdown rendering                                                           #
# --------------------------------------------------------------------------- #
def _heading_text(section: dict) -> str:
    """Markdown heading label; title_id (not title) tells a real opener from s_toc."""
    if section["section_id"] == "s_toc":
        return "(front matter)"
    title = (section["title"] or "").strip() if section["title_id"] else ""
    return title or "(untitled section)"


def _atom_markdown(atom: dict) -> str:
    """One atom as a markdown block, or "" when it carries nothing to show.

    Visual atoms keep the [TABLE id=...] / [FIGURE id=...] convention of markdown_build,
    so the ids the agent quotes stay the ones get_crop and the skeleton use. Role, not
    label, decides: dom_build routes "figure" as visual too, which config.FIGURE_LABELS
    (a markdown-only set) does not cover.
    """
    text = (atom.get("text") or "").strip()
    if atom["role"] != "visual":
        return text
    marker = "TABLE" if atom["label"] == "table" else "FIGURE"
    block = f"[{marker} id={atom['id']}]"
    return f"{block}\n{text}" if text else block


def render_section(section_id: str, dom: dict) -> str | dict:
    """Render a section and its whole subtree as markdown, with page markers.

    The atoms are wrapped in page bands (N 0-indexed, like the skeleton) so the agent can
    report the page a quote sits on by copying the marker above it instead of computing
    it. The opening and closing markers say whether the section is flush with the page or
    starts/ends inside it:

        ### Start Page N ###                  the section's first atom opens page N
        ### Section starts within Page N ###  page N already had content above it
        ### End Page N ###                    the section's last atom closes page N
        ### Section ends within Page N ###    page N still has content below it

    A page change inside the section emits the bare "End Page N" / "Start Page N+1" pair.

    Args:
        section_id: Section id as advertised in the skeleton (e.g. "s3", "s_toc").
        dom: build_dom output ({header, index, chrome, tree}).

    Returns:
        The markdown string, or an error dict {"error", "id_requested", "message"}
        when the id is not in the section tree.
    """
    tree, index = dom["tree"], dom["index"]

    subtree = _collect_descendant_sections(section_id, tree)
    if subtree is None:
        logger.warning("%s: unknown id %r", GET_SECTION, section_id)
        return {
            "error": "unknown_section_id",
            "id_requested": section_id,
            "message": (
                "get_section only accepts section ids: 's' followed by a number "
                "(s0, s12) or 's_toc'. Ids of the form pN_eM or sN_eM are atom "
                "ids — they name a table or figure inside a section, not a "
                "section — and this tool does not take them. Go back to the "
                "skeleton and use the id of the <section> that contains it."
            ),
        }

    wanted = {s["section_id"] for s, _ in subtree}
    # The opening title atom carries its section_id but lives outside "children"
    # (dom_build references it via title_id), so the flat scan would render it twice.
    titles = {s["title_id"] for s, _ in subtree if s["title_id"]}

    # dom["index"] is in insertion = global reading order, and chrome never enters it,
    # so filtering preserves reading order without any re-sort. dim7 is a PER-PAGE key:
    # sorting on it would scramble a multi-page section.
    by_section: dict[str, list[dict]] = {}
    for atom_id, atom in index.items():
        if atom["section_id"] in wanted and atom_id not in titles:
            by_section.setdefault(atom["section_id"], []).append(atom)

    # Page extremities, over the WHOLE document: "flush with the page" is a property of
    # the page, not of the requested subtree. Counted on the atoms that actually render —
    # a section title comes out as a markdown heading, outside the bands, and an atom
    # rendering "" emits nothing, so neither can hold a page end. Including them would
    # make every section opening at a page top report "starts within".
    all_titles = {s["title_id"] for s, _ in _iter_sections(tree["sections"]) if s["title_id"]}
    page_first: dict[int, str] = {}
    page_last: dict[int, str] = {}
    for atom_id, atom in index.items():
        if atom_id not in all_titles and _atom_markdown(atom):
            page_first.setdefault(atom["page"], atom_id)
            page_last[atom["page"]] = atom_id

    blocks: list[str] = []
    for section, depth in subtree:
        # The heading stays OUTSIDE the page bands: it is never the evidence a quote is
        # taken from, and keeping it out spares an empty band on a section with no atom.
        blocks.append(f"{'#' * (depth + 1)} {_heading_text(section)}")

        # Materialized before anything is emitted: both boundary markers depend on whether
        # the section's first/last RENDERED atom is also the page's, so the end of the
        # section must be known before its opening marker is written.
        rendered = []
        for atom in by_section.get(section["section_id"], ()):
            block = _atom_markdown(atom)
            if block:
                rendered.append((atom, block))
        if not rendered:
            continue

        first, last = rendered[0][0], rendered[-1][0]
        page = first["page"]
        blocks.append(f"### Start Page {page} ###" if page_first.get(page) == first["id"]
                      else f"### Section starts within Page {page} ###")
        for atom, block in rendered:
            # Internal page change: bare pair, the section owns both sides of the break.
            if atom["page"] != page:
                blocks.append(f"### End Page {page} ###")
                page = atom["page"]
                blocks.append(f"### Start Page {page} ###")
            blocks.append(block)
        blocks.append(f"### End Page {page} ###" if page_last.get(page) == last["id"]
                      else f"### Section ends within Page {page} ###")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Keyword search                                                               #
# --------------------------------------------------------------------------- #
DEFAULT_TOP_K = 10
SNIPPET_FULL_MAX_CHARS = 500    # at or below, the snippet is the whole atom text
SNIPPET_WINDOW_CHARS = 400      # above, width of the window kept around the match
_SNIPPET_HARD_MARGIN = 40       # blunt-cut allowance handed to truncate

_VISUAL_KINDS = {"table", "chart", "figure", "image"}

# Stripped from both ends of every token, on the query side and on the corpus side — the
# only property that matters is that ONE tokenizer serves both. The unicode additions are
# what string.punctuation misses on OCRed reports: typographic apostrophe, en/em dashes,
# guillemets, ellipsis.
_STRIP_CHARS = string.punctuation + "’—–«»…"


def _tokenize(text: str | None) -> list[str]:
    """Lowercased whitespace split, punctuation stripped off both ends of each token.

    No stemming and no stopword list: dropping "not"/"no" flips the meaning of a financial
    sentence, and a stemmer would conflate "operating" with "operations". A token that
    strips down to nothing is dropped, so punctuation cannot inflate a document's length
    and skew the BM25 normalization.

    Args:
        text: Raw text, from an atom or from the query.

    Returns:
        The token list, possibly empty.
    """
    tokens = []
    for raw in (text or "").lower().split():
        token = raw.strip(_STRIP_CHARS)
        if token:
            tokens.append(token)
    return tokens


def _kind(atom: dict, index: dict) -> str:
    """Kind of one atom, in the vocabulary the skeleton already shows the model.

    A caption reports the kind of the visual it captions, not "content": chart, figure and
    image atoms are never OCRed, so their text is empty and BM25 can never reach them
    directly — their caption is the only searchable handle they have, and it is useless if
    it does not say what it points at.

    Args:
        atom: An atom of dom["index"].
        index: The whole dom["index"], to resolve a caption onto its visual.

    Returns:
        "table", "chart", "figure", "image", "heading" or "content".

    Raises:
        ValueError: On a visual whose visual_kind is outside the four known labels.
    """
    caption_of = atom.get("caption_of")
    if caption_of or atom["role"] == "visual":
        source = index[caption_of] if caption_of else atom
        kind = source["visual_kind"]
        if kind not in _VISUAL_KINDS:
            raise ValueError(f"atom {source['id']!r}: unexpected visual_kind {kind!r}")
        return kind
    return "heading" if atom["label"] in DOM_TITLE_LABELS else "content"


def _snippet(text: str | None, kind: str, tokens: list[str]) -> str:
    """Snippet for one atom: the whole text when short, else a window around the match.

    A visual never gets a centred window: 400 chars taken from the middle of an OCRed
    table are figures without the header row that names them, which is worse than useless.
    Its snippet is always the head of the text, where the header sits.

    Args:
        text: Raw atom text.
        kind: _kind output for that atom.
        tokens: Tokenized query, used to locate the window.

    Returns:
        The snippet, elided with "..." on whichever side was cut.
    """
    text = clean_text(text)
    hard = SNIPPET_WINDOW_CHARS + _SNIPPET_HARD_MARGIN
    if kind in _VISUAL_KINDS:
        return truncate(text, SNIPPET_WINDOW_CHARS, hard)
    if len(text) <= SNIPPET_FULL_MAX_CHARS:
        return text

    lowered = text.lower()
    found = [pos for pos in (lowered.find(token) for token in tokens) if pos >= 0]
    start = max(0, min(found, default=0) - SNIPPET_WINDOW_CHARS // 2)
    # truncate owns the tail: it snaps to a word boundary, appends the "..." itself, and
    # leaves the text alone when the window already runs to the end of the atom.
    window = truncate(text[start:], SNIPPET_WINDOW_CHARS, hard)
    return f"...{window}" if start else window


class _SearchIndex:
    """BM25 over the atoms of one dom, plus the per-call cache. See build_search_index.

    The corpus is dom["index"] walked in its own iteration order, which is insertion =
    global reading order. That position doubles as the tie-break key, so two atoms with
    the same BM25 score always rank in reading order and a run is reproducible.
    """

    def __init__(self, dom: dict) -> None:
        self._index = dom["index"]
        self._atoms = list(self._index.values())
        corpus = [_tokenize(atom.get("text")) for atom in self._atoms]
        # BM25Okapi divides by the mean document length, so a corpus whose atoms are ALL
        # empty (a dom built without OCR) would raise on the first query. The index is
        # then simply not built and every search comes back empty.
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None
        self._cache: dict[tuple[str, int], list[dict]] = {}
        if self._bm25 is None:
            logger.warning("%s: no searchable text in this dom, every call returns []",
                           SEARCH)

    def search(self, keyword: str, top_k: int = DEFAULT_TOP_K) -> list[dict] | dict:
        """Locate the sections holding a keyword, best first.

        Args:
            keyword: The words to look for; tokenized like the corpus.
            top_k: Maximum number of SECTIONS to return. The tool schema is advisory on
                this endpoint, so anything that is not a positive int falls back to
                DEFAULT_TOP_K rather than being refused.

        Returns:
            One hit per section — {"section_id", "page", "kind", "snippet", "n_matches",
            "score"} — capped at top_k, or an error dict {"error", "message"} when the
            keyword is empty. No match is an empty list, not an error.
        """
        if not isinstance(keyword, str) or not keyword.strip():
            logger.warning("%s: empty keyword", SEARCH)
            return {
                "error": "empty_keyword",
                "message": ("search needs something to look for. Give 2 to 5 content "
                            "words taken from the document's own vocabulary."),
            }
        if not isinstance(top_k, int) or top_k <= 0:
            top_k = DEFAULT_TOP_K

        # Cache keyed on the served arguments: the same call twice returns the same list
        # instead of re-ranking, and the model does not pay a step for a re-query it could
        # have read above. It lives here, not in the loop, so every caller inherits it.
        key = (keyword, top_k)
        if key not in self._cache:
            self._cache[key] = self._rank(keyword, top_k)
        return self._cache[key]

    def _rank(self, keyword: str, top_k: int) -> list[dict]:
        """Score the corpus, collapse the atoms onto their sections, cut at top_k."""
        tokens = _tokenize(keyword)
        if self._bm25 is None or not tokens:
            return []

        # score > 0 IS the definition of a match: a term absent from an atom contributes a
        # zero numerator whatever its idf. Without this filter BM25 ranks the WHOLE corpus,
        # and a keyword the document does not contain still comes back with top_k
        # arbitrary sections for the model to chase.
        best: dict[str, tuple[float, int, dict]] = {}
        n_matches: dict[str, int] = {}
        scores = self._bm25.get_scores(tokens)
        for position, (atom, score) in enumerate(zip(self._atoms, scores)):
            if score <= 0:
                continue
            section_id = atom["section_id"]
            n_matches[section_id] = n_matches.get(section_id, 0) + 1
            # Strictly greater: the corpus is walked in reading order, so on a tie inside
            # a section the earliest atom is the one that represents it.
            if section_id not in best or score > best[section_id][0]:
                best[section_id] = (float(score), position, atom)

        ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[1][1]))[:top_k]
        return [self._hit(section_id, score, atom, n_matches[section_id], tokens)
                for section_id, (score, _, atom) in ranked]

    def _hit(self, section_id: str, score: float, atom: dict, n_matches: int,
             tokens: list[str]) -> dict:
        """One result row. No atom_id: no tool consumes one, and handing the model an id
        it cannot use is what feeds the atom-id/section-id confusion get_section refuses."""
        kind = _kind(atom, self._index)
        return {
            "section_id": section_id,
            "page": atom["page"],
            "kind": kind,
            "snippet": _snippet(atom.get("text"), kind, tokens),
            "n_matches": n_matches,
            # Rounded for display only — the ranking above sorts on the raw float. The
            # value is a within-document score, never comparable across documents.
            "score": round(score, 2),
        }


def build_search_index(dom: dict) -> _SearchIndex:
    """Build the keyword index of one dom.

    Args:
        dom: build_dom output ({header, index, chrome, tree}).

    Returns:
        An index exposing .search(keyword, top_k). Hold it for the lifetime of the
        document: its per-call cache lives there.
    """
    return _SearchIndex(dom)


# --------------------------------------------------------------------------- #
# Tool definitions                                                             #
# --------------------------------------------------------------------------- #
def build_get_section_tool() -> dict[str, Any]:
    """Build the section-reading tool definition.

    Returns:
        An OpenAI tool definition named GET_SECTION.
    """
    return {
        "type": "function",
        "function": {
            "name": GET_SECTION,
            "description": (
                "Retrieve the full text of a section and all of its subsections, as "
                "markdown. Use the ids shown in the skeleton."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "section_id": {
                        "type": "string",
                        "description": "Section id from the skeleton, e.g. 's3' or 's_toc'.",
                    },
                },
                "required": ["section_id"],
            },
        },
    }


def build_search_tool() -> dict[str, Any]:
    """Build the keyword-search tool definition.

    Returns:
        An OpenAI tool definition named SEARCH.
    """
    return {
        "type": "function",
        "function": {
            "name": SEARCH,
            "description": (
                "Locate where a term appears in the document. Returns up to top_k hits, "
                "one per section, each with the id of the SECTION holding the match, its "
                "page, its kind and a snippet. Matching is lexical, not semantic: give 2 "
                "to 5 content words, not a question. section_id is the only field you "
                "may pass to another tool. A snippet tells you where to look, never what "
                "the answer is — read the section with get_section before answering."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Words to look for, e.g. 'net income 2023'.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": ("Maximum number of sections to return; defaults "
                                        f"to {DEFAULT_TOP_K}."),
                    },
                },
                "required": ["keyword"],
            },
        },
    }


def build_submit_draft_tool(field_definitions: dict[str, str],
                            schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the finish-tool for the expected fields.

    Args:
        field_definitions: Mapping from field name to its definition text.
        schema: Per-field property schema; defaults to DRAFT_SCHEMA (KYC shape). Every
            key is required, so the model always answers on all of them.

    Returns:
        An OpenAI tool definition named SUBMIT_DRAFT, whose `fields` object holds one
        entry per field.
    """
    schema = DRAFT_SCHEMA if schema is None else schema
    return {
        "type": "function",
        "function": {
            "name": SUBMIT_DRAFT,
            "description": (
                "Submit the extracted values for all fields. Call this exactly once, "
                "when you have read everything you need."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "fields": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            name: {
                                "type": "object",
                                "description": definition,
                                "additionalProperties": False,
                                "properties": schema,
                                "required": list(schema),
                            }
                            for name, definition in field_definitions.items()
                        },
                        "required": list(field_definitions),
                    },
                },
                "required": ["fields"],
            },
        },
    }
