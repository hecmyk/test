# agent_dom_v2

This package implements DocAgent v2: an orchestrator that navigates the document skeleton
and delegates every read to an ephemeral subagent. The architecture, the metrics contract
and the evaluation runners are described in `README_dom_agent_v2.md` at the repository
root; this file only describes what the folder contains.

## What each file holds

**`settings.py`** — the OpenAI client, the two model ids (`ORCHESTRATOR_MODEL`,
`SUBAGENT_MODEL`), the three budgets (`ORCHESTRATOR_MAX_ROUNDS`, `SEARCH_TOP_K`,
`SUBAGENT_MAX_TOOL_CALLS`), `LOG_SUBAGENT_DETAILS`, and the prompt bank `PROMPTS_V2`
loaded from `prompts_dom_v2.yaml` at the repository root.

**`metrics.py`** — `_init_metrics_v2`, the function that declares every key of the
`agent_metrics` dict: the twenty v1 keys in their v1 order, then the v2 keys, flat.

**`sections.py`** — `build_section_maps` (section objects, ancestor chains and
breadcrumbs of one DOM), `resolve_section_ids` (unknown ids dropped and recorded,
duplicates and descendants of a delivered ancestor collapsed) and `build_subagent_context`
(the delivered sections rendered whole, each under its breadcrumb).

**`report.py`** — the `report` tool of the subagent: the Pydantic models `Evidence`,
`SectionContent` and `Report` from which its JSON Schema is generated, `build_report_tool`,
`_normalize_report` (the lenient reading of what the model sent back) and
`_descriptive_fallback` (the report of an invocation that produced none).

**`protocol.py`** — `chat()`, the single call to the endpoint, wrapped in
`call_with_retry`; `_classify_protocol` (the four protocol buckets of an assistant turn);
`parse_tool_args` and `refuse_unknown_tool` (recording of malformed and unknown tool
calls); `force_finish_tool` (the last forced call on a finish-tool); `tool_schema` and its
`_ToolSchema` generator (a Pydantic model rendered as an OpenAI tool definition);
`_RETRY_PROTOCOL_MSG`.

**`verifications.py`** — the non-repairing checks `check_key_figures` and
`check_page_ranges`, the `KeyFigureVerdict` / `PageRangeVerdict` models of the records they
produce (a discriminated union on `check`, `VerdictAdapter` to read one back), and
`_aggregate_verifications`, which folds the records into the four aggregate metric keys.

**`subagent.py`** — `run_subagent`: one invocation per delegation, sections pre-loaded in
the opening message, `get_visual` and `report` as its only tools, context dropped on return.
Also `_force_report` and `_SUBAGENT_TOOL_NAMES`.

**`orchestrator.py`** — `run_dom_agent_v2`, the entry point; the `delegate_read` and
`search_and_read` tools with their `DelegateReadArgs` / `SearchAndReadArgs` models;
`_delegate`, the shared body of both delegation paths; `_force_submit_draft_v2`;
`_MODES_V2` and `_ORCHESTRATOR_TOOL_NAMES`.

**`__init__.py`** — the facade: re-exports the public names of the modules above
(`run_dom_agent_v2`, `run_subagent`, the budgets, the model ids, the client, the tool
builders, the checks) and defines `__all__`. Private names are not re-exported.

## Dependency tree inside the package

Arrows read "imports". Four modules import nothing from the package: `settings`,
`metrics`, `sections`, `verifications`.

```
__init__ ──► orchestrator ──► subagent ──► report ──► protocol ──► settings
                 │                │                       ▲
                 │                ├───────────────────────┘
                 │                ├──► sections
                 │                ├──► verifications
                 │                └──► settings
                 ├──► protocol
                 ├──► sections
                 ├──► verifications
                 ├──► metrics
                 └──► settings
```

| Module | Imports from the package |
|---|---|
| `settings` | — |
| `metrics` | — |
| `sections` | — |
| `verifications` | — |
| `protocol` | `settings` |
| `report` | `protocol` |
| `subagent` | `protocol`, `report`, `sections`, `settings`, `verifications` |
| `orchestrator` | `metrics`, `protocol`, `sections`, `settings`, `subagent`, `verifications` |
| `__init__` | `orchestrator`, `protocol`, `report`, `sections`, `settings`, `subagent`, `verifications` |

## Dependencies outside the package

| Source | Symbols | Used by |
|---|---|---|
| `agent_dom` (v1 agent) | `_add_usage`, `_serve_visual`, `_check_pdf_matches_dom`, `_normalize_draft` | `protocol`, `subagent`, `orchestrator` |
| `dom_tools` | `render_section`, `_heading_text`, `build_search_index`, `build_submit_draft_tool`, `QA_SCHEMA`, `SUBMIT_DRAFT` | `sections`, `orchestrator` |
| `dom_visual` | `GET_VISUAL`, `build_get_visual_tool` | `subagent` |
| `eval.metrics` | `normalize_text` | `verifications` |
| `eval.rate_limit` | `call_with_retry` | `protocol` |
| `config` | `MISTRAL_AGENT_MODEL`, `MISTRAL_API_KEY`, `MISTRAL_BASE_URL` | `settings` |
| `openai` | `OpenAI` | `settings` |
| `pydantic` | `BaseModel`, `ConfigDict`, `Field`, `TypeAdapter`, `json_schema.GenerateJsonSchema` | `protocol`, `report`, `verifications`, `orchestrator` |
| `yaml` | `safe_load` | `settings` |
