"""Client, model ids, budgets and prompt bank of the v2 agent — the run-level knobs.

Held apart from the package's __init__ so the submodules can import them without a
circular import: subagent.py needs the client, its model id and the prompt bank, and
__init__ imports subagent.py.

HOW A NOTEBOOK SETS A BUDGET. The runners import the package as a module and pass every
value explicitly at call time — `run_dom_agent_v2(..., search_top_k=agent_dom_v2.SEARCH_TOP_K)`
— so `agent_dom_v2.SEARCH_TOP_K = 8` in a cell before the run takes effect and lands in
run_metadata.json. That works because it rebinds the name on the facade the runners read,
and it is the ONLY channel that works: function defaults are evaluated at definition time,
so a default in a signature below is the value at import, whatever the cell did after.
Same as before the split; the facade merely re-exports what is defined here.
"""

from pathlib import Path

# Requires PyYAML (external runtime dependency, not declared in this repo).
import yaml
from openai import OpenAI

from config import MISTRAL_AGENT_MODEL, MISTRAL_API_KEY, MISTRAL_BASE_URL

client = OpenAI(
    base_url=MISTRAL_BASE_URL,
    api_key=MISTRAL_API_KEY,
)

# The two tiers are configured separately on purpose. They hold the same id in v2.0 so the
# architecture is measured alone; SUBAGENT_MODEL is the single line to change when a VLM
# is confirmed on the endpoint (v2.2), and no other code has to move.
ORCHESTRATOR_MODEL = MISTRAL_AGENT_MODEL
SUBAGENT_MODEL = MISTRAL_AGENT_MODEL

# The bank stays at the repository root, next to prompts_dom.yaml; parents[1] is that root
# from inside the package.
with open(Path(__file__).resolve().parents[1] / "prompts_dom_v2.yaml", encoding="utf-8") as f:
    PROMPTS_V2 = yaml.safe_load(f)

# Budgets. v2.0 holds top_k at the value the single-phase run used: raising k is v2.1, and
# changing it here would confound it with the split this run is meant to measure.
ORCHESTRATOR_MAX_ROUNDS = 3
SEARCH_TOP_K = 5
# Ceiling, not a target: the subagent reads nothing (its sections are already there), so
# this only bounds get_visual plus the closing report.
SUBAGENT_MAX_TOOL_CALLS = 6

# Off by default: when True the run writes the full report and the pre-loaded context under
# runs/{run_id}/debug_v2/, OUTSIDE the scored rows.
LOG_SUBAGENT_DETAILS = False
