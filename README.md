# DOM agents and evaluation harness

This PR adds two question-answering agents that read a document through a DOM: a tree of sections, text blocks, tables and figures built from the OCR output. It also adds the harness that runs and scores them on FinRAGBench-V and MMLongBench-Doc. Layout detection and OCR from PR 1 are still underneath and still produce the input; the DOM layer and the agents sit on top. v1 is a single agent that navigates the DOM with tools. v2 splits the work: an orchestrator that sees only the document outline, and short-lived subagents that read the sections. Two baselines (full-context, and oracle, where the gold pages are handed to the reader) are included for comparison.

For the underlying layout → OCR → markdown pipeline (PR 1), see README_pipeline.md.

## Architecture at a glance

```
PDF
 └─ Layout.py → ocr.py ........................ PR 1: doc_ocr  ({"page_0": [item, ...], ...})
     └─ dom_build.build_dom ................... DOM dict {header, index, chrome, tree}
         └─ dom_render.build_outline_xml ...... XML skeleton of the DOM (no body text)
             └─ agent v1 or v2 ................ answer {answer, quote, source_pages}
                 └─ eval: runner → all.jsonl → score → scored.jsonl
```

DOM layer:
- `dom_build.py` — builds the DOM from `doc_ocr` and page sizes.
- `reading_order.py` — fixes the reading order of the items on a page (called by `build_dom`).
- `dom_build_levels.py` — one LLM call that assigns heading levels 1-4 (called by `build_dom`).
- `dom_render.py` — renders the DOM as the XML skeleton the agent reads first.
- `dom_tools.py` — the agent's tools: `get_section`, `search`, `submit_draft`.
- `dom_visual.py` — the `get_visual` tool: returns the image crop of a table or figure.

Agents:
- `dom_agent.py` — v1, `run_dom_agent`: one agent, one tool-calling loop.
- `agent_dom_v2/` — v2, `run_dom_agent_v2`: orchestrator plus ephemeral subagent.
- `dom_prompts.yaml`, `prompts_dom_v2.yaml` — the prompts of v1 and v2.

Evaluation:
- `eval/rate_limit.py` — paced, retried LLM calls (`call_with_retry`) and the infra-failure log.
- `eval/metrics.py` — `token_f1`, `page_recall`, text normalization.
- `eval/judges.py` — binary LLM judge for FinRAGBench-V, cached per run.
- `eval/baselines/` — `reader.py` (shared reader + page-level OCR), `fullctx.py`, `oracle.py`.
- `eval/finragbench/` — `run_benchmark.py` (v1), `run_v2.py` (v2), `score_results.py`.
- `eval/mmlongbench/` — `dataset.py`, `run_mm.py` (v1), `run_mm_v2.py`, `run_oracle_mm.py`,
  `extract.py`, `render.py`, `score_mm.py`, `official/` (the benchmark's own scorer, vendored),
  `dom_prompts_mm*.yaml` (prompts for this benchmark).

## Quickstart

Prerequisites:
- Python 3.10.
- Environment variables `MISTRAL_BASE_URL` and `MISTRAL_API_KEY` (read in `config.py`). The
  agents, OCR, judge and answer extractor all call that OpenAI-compatible endpoint.
- The benchmark data paths are hard-coded: `ROOT` in `eval/finragbench/run_benchmark.py` and
  `MM_ROOT` in `eval/mmlongbench/dataset.py` (both `/home/datacamp/data/...`). Edit them on another machine.
- Run everything from the repo root. There are no command-line entry points: each command below is Python.

**1. Build a DOM for one document**

```python
from pathlib import Path
from pipeline import process_pipeline
from pdf_utils import page_dims_from_pdf
from dom_build import build_dom
from dom_render import build_outline_xml

pdf = Path("my_doc.pdf")
_, _, doc_ocr, _ = process_pipeline(pdf, "out")    # PR 1: layout + OCR, ~1 Mistral call per page
dom = build_dom(doc_ocr, page_dims_from_pdf(pdf), doc_id=pdf.stem, pdf_name=pdf.name)
skeleton = build_outline_xml(dom)
print(dom["header"]["stats"])                      # n_atoms, n_sections, n_visual, ...
```

`build_dom` makes one LLM call to level the headings. Pass `title_leveler=None` to skip it
(every section then stays at level 1).

**2. Run agent v1**

```python
from dom_agent import run_dom_agent

FIELDS = {"qa": "Answer the question grounded in the document."}
draft, metrics = run_dom_agent(dom, skeleton, FIELDS, mode="qa",
                               question="What was the 2023 net revenue?",
                               pdf_path=pdf)       # pdf_path=None: text only, no get_visual
print(draft["qa"])                                 # {"answer": ..., "quote": ..., "source_pages": [...]}
```

`mode` defaults to `"kyc"` (multi-field extraction); pass `mode="qa"` for a question.

**3. Run agent v2**

```python
from agent_dom_v2 import run_dom_agent_v2

draft, metrics, debug = run_dom_agent_v2(dom, skeleton, FIELDS, question="What was the 2023 net revenue?",
                                         pdf_path=pdf)
print(draft["qa"])                                 # same shape as v1; `debug` is [] unless log_subagent_details=True
```

Budgets (`ORCHESTRATOR_MAX_ROUNDS`, `SEARCH_TOP_K`, `SUBAGENT_MAX_TOOL_CALLS`) and model ids are
in `agent_dom_v2/settings.py`.

**4. Run a baseline**

```python
from eval.baselines.reader import ocr_pages
from eval.baselines.fullctx import answer_batched
from eval.baselines.oracle import answer_oracle

pages_md = ocr_pages(str(pdf))                     # page-level OCR, cached under eval/finragbench/artifacts/<stem>/
full = answer_batched("What was the 2023 net revenue?", pages_md)                    # full-context
orac = answer_oracle("What was the 2023 net revenue?", [3, 4], "both", pages_md, str(pdf))  # oracle
print(full["pred_answer"], orac["pred_answer"])
```

Oracle `mode` is `"md"`, `"image"` or `"both"`, and gold pages are **0-indexed**. Both functions
return one result row with the query-side fields (`query_id`, `gold_answer`, ...) set to `None`
for the caller to fill. For a benchmark-scale oracle run on MMLongBench-Doc, see below.

## Running a benchmark

Per document, a runner needs `eval/<benchmark>/artifacts/<stem>/doc_ocr.json`, with the shape of
`doc_ocr` above. The files that produce it in bulk are not part of this PR; for one document:

```python
import json
d = Path("eval/finragbench/artifacts") / pdf.stem
d.mkdir(parents=True, exist_ok=True)
(d / "doc_ocr.json").write_text(json.dumps(doc_ocr, ensure_ascii=False), encoding="utf-8")
```

The runner builds and caches `doc_dom.json` and `skeleton.xml` next to it on first use. Each run
writes to `eval/<benchmark>/runs/<run_id>/`. Re-running the same `run_id` skips queries already
answered (crash recovery, not a restart); pass a new `run_id` to start over. `limit` caps the
total number of queries (pilot runs).

### FinRAGBench-V

```python
from eval.finragbench.run_benchmark import run_benchmark   # v1
from eval.finragbench.score_results import score_results

run_benchmark("fin_v1_pilot", limit=5, notes="v1 pilot")   # → runs/fin_v1_pilot/<stem>.jsonl
score_results("fin_v1_pilot")                              # → all.jsonl, scored.jsonl, judge_cache.json
```

v2, run + score in one call:

```python
from eval.finragbench.run_v2 import run_and_score
run_and_score("fin_v2_pilot", limit=5, notes="v2 pilot")   # returns the path of scored.jsonl
```

`score_results` prints `[fin_v1_pilot] <N> lignes -> .../scored.jsonl`. Each line of
`scored.jsonl` is the run row plus four metrics (placeholder values):

```json
{"query_id": "<stem>_<n>", "question": "...", "gold_answer": "...", "gold_pages": [4],
 "pred_answer": "...", "pred_quote": "...", "pred_source_pages": [3], "agent_metrics": {...},
 "error": null, "token_f1": 0.0-1.0, "page_recall": 0.0-1.0, "judge_score": 0-1, "judge_reasoning": "..."}
```

`gold_pages` is 1-indexed and agent citations are 0-indexed; `page_recall` handles the shift.
`judge_score` is `null` when the judge reply could not be parsed. v2 runs also write
`verifications.jsonl` (and `debug_v2/` with `log_subagent_details=True`) in the run directory.

### MMLongBench-Doc

Scoring has three stages: run, extract a short answer from each reply (one LLM call per
row), then score with the official rules.

```python
from eval.mmlongbench.run_mm import run_mm                 # v1
from eval.mmlongbench.extract import extract_run
from eval.mmlongbench.score_mm import score_run, sanity_counters

run_mm("mm_v1_pilot", limit=20, notes="v1 pilot")          # → runs/mm_v1_pilot/<stem>.jsonl (no gold in rows)
extract_run("mm_v1_pilot")                                 # → all.jsonl, extracted.jsonl
score_run("mm_v1_pilot")                                   # → scored.jsonl, results.txt
sanity_counters("mm_v1_pilot")                             # read before trusting the F1
```

v2 and the oracle arm take a stratified sample (`p`, `seed`); use the same pair on both to
compare them on the same questions:

```python
from eval.mmlongbench import run_mm_v2, run_oracle_mm

run_mm_v2.run_and_score("mm_v2_pilot", p=0.1, seed=42, notes="v2 pilot")
run_oracle_mm.run_and_score("mm_oracle_both", mode="both", p=0.1, seed=42)   # mode: md | image | both
```

`run_mm` (v1) has no `p`/`seed`: it runs every question. The oracle arm skips unanswerable
questions, so compare it with an agent run on the answerable subset only.

`score_run` prints `[mm_v1_pilot] Acc <0.0-1.0> | F1 <0.0-1.0>`; the full breakdown goes to
`results.txt`. A `scored.jsonl` line is the run row plus the gold and the score (placeholders):

```json
{"query_id": "<stem>__<nnnn>", "doc_id": "<stem>.pdf", "pred_answer": "...", "agent_metrics": {...},
 "error": null, "answer": "...", "answer_format": "...", "evidence_pages": "[3, 5]",
 "pred": "...", "parse_ok": true, "score": 0.0-1.0}
```

A row whose extraction could not be parsed has `parse_ok: false`, no `score`, and is left out of
the Acc/F1; `score_run` reports the count.

## What's NOT included in this PR

Kept out of this PR, following the PR 1 convention. 

- `eval/analyze.py`, `eval/mmlongbench/analyze_mm.py` — analysis and plotting tools.
- `eval/finragbench/isolated_visual.py`, `eval/finragbench/title_diagnostic.py` — ad hoc diagnostics.
- The notebooks .

## Known cleanup debt

Tracked for a follow-up cleanup pass; none of it affects functional correctness.

- Some comments, docstrings and printed messages in the eval modules are still in French
  (`eval/rate_limit.py`, `eval/mmlongbench/dataset.py`, the runners' progress output).
- A few helpers are duplicated between the two benchmarks' runners (`load_or_build_dom`,
  `pdf_for`, `_init_run`, `aggregate_run` in `run_benchmark.py` / `run_mm.py`).
- Some magic numbers are not in `config.py`: `TPM_BUDGET` in `rate_limit.py`, `MAX_STEPS`
  in the runners, batch size and context window in `fullctx.py`.

