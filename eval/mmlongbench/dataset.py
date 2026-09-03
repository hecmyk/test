"""MMLongBench-Doc dataset adapter: the questions, keyed so a run can be resumed.

samples.json carries no identifier of its own, and (doc_id, question) is unique only
1081 times out of 1082 — one pair is duplicated. The position in the file is therefore
the only safe key, and `query_id` is built from it. Everything the agent must NOT see
(answer, answer_format, evidence_pages, evidence_sources) stays here and is joined back
by query_id at scoring time.
"""

import json
from pathlib import Path

MM_ROOT = Path("/home/datacamp/data/MMLongBench-Doc")
MM_DOCS = MM_ROOT / "data" / "documents"
SAMPLES = MM_ROOT / "data" / "samples.json"


def load_samples() -> list[dict]:
    """Every question of the benchmark, enriched with `stem` and `query_id`.

    Returns:
        The 1082 samples in file order, each with the original fields plus `stem` (the
        PDF's basename without extension, which is also the artifacts folder name) and
        `query_id` (`{stem}__{index:04d}`).
    """
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    for i, sample in enumerate(samples):
        doc_id = sample["doc_id"]
        sample["stem"] = doc_id[:-4] if doc_id.endswith(".pdf") else doc_id
        sample["query_id"] = f"{sample['stem']}__{i:04d}"
    return samples


def samples_by_query_id() -> dict[str, dict]:
    """The same samples, indexed by query_id for the scoring join.

    Returns:
        `{query_id: sample}`.
    """
    return {sample["query_id"]: sample for sample in load_samples()}
