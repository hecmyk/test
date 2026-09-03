"""Stage 2 of the official protocol: the free-form answer reduced to a short one.

One GPT-OSS-120B call per question, with the official extraction prompt and the official
role layout — the prompt as the user turn, the Question/Analysis pair as an assistant
turn, so the conversation ends on an assistant message. That layout is unusual and may
behave differently here than on the OpenAI API; it is kept verbatim because the published
numbers were produced with it, and a pivot is one edit away if the extractions come back
wrong.

Results are appended to runs/{run_id}/extracted.jsonl, one line per query_id, and a
question already in that file is skipped — the same crash net as the runner. The cache
lives IN the run directory and is never shared: an extraction is a function of the answer,
which changes from one run to the next.

A reply the official parser cannot read is recorded as `pred: null` with `parse_ok: false`
and counted, never as a zero. A silent zero would be indistinguishable from a wrong
answer, and would move the score without leaving a trace.
"""

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))   # repo root, for config
sys.path.insert(0, str(_HERE))              # eval/mmlongbench/, for the imports below

from config import MISTRAL_API_KEY, MISTRAL_BASE_URL   # noqa: E402
from render import draft_to_analysis                   # noqa: E402
from run_mm import AGGREGATE_NAME, EXTRACTED_NAME, RUNS, aggregate_run   # noqa: E402

logger = logging.getLogger(__name__)

# The extractor is NOT the agent's model: the official protocol uses a separate reader to
# turn prose into a short answer, and mixing the two would let the agent grade its own
# phrasing. The endpoint is the same, only the model id differs.
MODEL = "GPT-OSS-120B"
MAX_TOKENS = 2048

PROMPT_PATH = _HERE / "official" / "prompt_for_answer_extraction.md"

client = OpenAI(base_url=MISTRAL_BASE_URL, api_key=MISTRAL_API_KEY)


def _call_extractor(question: str, analysis: str, prompt: str) -> str:
    """One extractor call, official role layout and sampling parameters.

    Args:
        question: The benchmark question.
        analysis: The Analysis string from render.draft_to_analysis.
        prompt: The official extraction prompt.

    Returns:
        The reply text, or "Failed" when the call raised — the sentinel the official
        implementation uses, kept so the parse failure is counted the same way.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant",
                 "content": "\n\nQuestion:{}\nAnalysis:{}\n".format(question, analysis)},
            ],
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("extractor call failed: %s: %s", type(e).__name__, e)
        return "Failed"


def parse_extraction(raw: str) -> str | None:
    """The short answer inside an extractor reply, or None when it is not there.

    The split expression is the official one, with two additions. The marker must be
    present, otherwise the reply is a failure and not an answer — without that guard the
    "Failed" sentinel of a dead call would come back as the prediction. And `[-1]`, not
    `[1]`: a model that reasons before answering can write the marker twice, and the LAST
    occurrence is the one that carries the answer.

    Args:
        raw: The extractor's reply.

    Returns:
        The extracted answer, or None when the reply carries no readable one.
    """
    if not raw or "Extracted answer:" not in raw:
        return None
    return raw.split("Answer format:")[0].split("Extracted answer:")[-1].strip() or None


def extract_run(run_id: str, max_workers: int = 8) -> Path:
    """Extract a short answer for every row of a run.

    Args:
        run_id: The iteration to extract.
        max_workers: Concurrent extractor calls.

    Returns:
        Path of runs/{run_id}/extracted.jsonl.
    """
    aggregate_run(run_id)   # always re-aggregated: a stale all.jsonl would extract a subset

    run_dir = RUNS / run_id
    out_path = run_dir / EXTRACTED_NAME
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    rows = [json.loads(line)
            for line in (run_dir / AGGREGATE_NAME).read_text(encoding="utf-8").splitlines()
            if line.strip()]

    done = set()
    if out_path.exists():
        done = {json.loads(line)["query_id"]
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()}
    pending = [row for row in rows if row["query_id"] not in done]
    if not pending:
        print(f"[{run_id}] {len(done)} extractions déjà faites, rien à faire")
        return out_path

    def extract(row: dict) -> dict:
        # A dict, not the bare answer: draft_to_analysis takes a submit_draft entry, and
        # the JSONL row carries that entry flattened into pred_* columns.
        analysis = draft_to_analysis({"answer": row["pred_answer"]})
        raw = _call_extractor(row["question"], analysis, prompt)
        pred = parse_extraction(raw)
        return {"query_id": row["query_id"], "analysis": analysis,
                "extracted_res": raw, "pred": pred, "parse_ok": pred is not None}

    n_parse_failures = 0
    # Written as the pool yields, in order: a crash keeps every extraction already paid for.
    with out_path.open("a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=max_workers) as pool:
        for result in tqdm(pool.map(extract, pending), total=len(pending), unit="q"):
            n_parse_failures += not result["parse_ok"]
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

    print(f"[{run_id}] {len(pending)} extraites ({len(done)} déjà faites) -> {out_path}")
    print(f"[{run_id}] n_parse_failures = {n_parse_failures}")
    return out_path
