"""DOM agent v2 — navigation and reading split across two tiers.

run_dom_agent_v2 keeps the ORCHESTRATOR on the XML skeleton and delegates every read to an
EPHEMERAL subagent: one invocation per delegation, its context dropped once it has
reported. Only a compressed report reaches the orchestrator, so the tool results that made
the single-phase loop silt up never accumulate at the navigation tier.

The subagent has NO get_section. Its sections are PRE-LOADED into the message that opens
the invocation. That is what keeps the cost linear — the single-phase loop re-sends a
context that grows with every read, so reading k sections costs O(k^2) tokens — and what
puts every crop in a context already holding the section text, the favourable oracle_img
case of E1/E5, obtained by construction rather than asked for in a prompt.

Nothing here modifies agent_dom: _add_usage and _serve_visual are imported from it, the
read tools come from dom_tools/dom_visual unchanged, and _init_metrics_v2 re-emits the v1
metric keys under their v1 names, in their v1 order, so score_results and eval/analyze.py
run on a v2 run untouched. Every v2 metric is ADDITIVE, in flat keys beside them.
"""

# The package in one look:
#   settings.py       client, model ids, budgets, prompt bank — what a notebook sets before a run
#   metrics.py        _init_metrics_v2, the key contract with eval/analyze.py
#   sections.py       section id resolution and the pre-loaded context
#   report.py         the subagent's finish-tool and the report shape read back
#   protocol.py       chat() — the ONE paced call — and the tool-calling instrumentation
#   verifications.py  the non-repairing checks and their verdict records
#   subagent.py       run_subagent, one ephemeral read per delegation
#   orchestrator.py   run_dom_agent_v2 and the delegation tools
#
# This facade re-exports the public names, so `import agent_dom_v2` and
# `from agent_dom_v2 import run_dom_agent_v2` are what the runners and notebooks use, and
# `agent_dom_v2.SEARCH_TOP_K = 8` in a cell before a run still takes effect (see
# settings.py for why that channel works and no other). Private names are not re-exported:
# import them from their module.

from .orchestrator import (DELEGATE_READ, SEARCH_AND_READ, build_delegate_read_tool,
                           build_search_and_read_tool, run_dom_agent_v2)
from .protocol import chat, force_finish_tool, parse_tool_args, refuse_unknown_tool
from .report import REPORT, build_report_tool
from .sections import build_section_maps, build_subagent_context, resolve_section_ids
from .settings import (LOG_SUBAGENT_DETAILS, ORCHESTRATOR_MAX_ROUNDS, ORCHESTRATOR_MODEL,
                       PROMPTS_V2, SEARCH_TOP_K, SUBAGENT_MAX_TOOL_CALLS, SUBAGENT_MODEL,
                       client)
from .subagent import run_subagent
from .verifications import check_key_figures, check_page_ranges

__all__ = [
    "run_dom_agent_v2", "run_subagent",
    "DELEGATE_READ", "SEARCH_AND_READ", "REPORT",
    "build_delegate_read_tool", "build_search_and_read_tool", "build_report_tool",
    "build_section_maps", "build_subagent_context", "resolve_section_ids",
    "check_key_figures", "check_page_ranges",
    "chat", "force_finish_tool", "parse_tool_args", "refuse_unknown_tool",
    "client", "ORCHESTRATOR_MODEL", "SUBAGENT_MODEL", "PROMPTS_V2",
    "ORCHESTRATOR_MAX_ROUNDS", "SEARCH_TOP_K", "SUBAGENT_MAX_TOOL_CALLS",
    "LOG_SUBAGENT_DETAILS",
]
