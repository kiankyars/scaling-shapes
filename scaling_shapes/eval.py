"""Run lm-evaluation-harness over a single Pythia (size, checkpoint) pair.

One call loads the model once (via HFLM), then runs each task at its
configured few-shot count. Designed to be called once per Modal function
invocation so the grid parallelizes naturally across (size, revision) pairs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .tasks import TASKS, Task


def evaluate_checkpoint(
    *,
    hf_id: str,
    revision: str,
    out_path: Path,
    batch_size: str | int = "auto",
    limit: int | float | None = None,
    dtype: str = "bfloat16",
    device: str = "cuda:0",
    tasks: list[Task] | None = None,
) -> dict:
    """Run the full task suite on one (hf_id, revision); write a results row.

    The output JSON contains, per task, all metrics returned by lm-eval-harness
    (typically `acc`, `acc_stderr`, and where applicable `acc_norm` /
    `acc_norm_stderr`). MMLU expands to 57 subtask entries (`mmlu_*`).

    Returns the parsed result row.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tasks = tasks or TASKS
    t0 = time.time()

    lm = HFLM(
        pretrained=hf_id,
        revision=revision,
        dtype=dtype,
        device=device,
        batch_size=batch_size,
        trust_remote_code=False,
    )
    load_s = time.time() - t0

    all_results: dict[str, dict] = {}
    n_shot: dict[str, int] = {}
    versions: dict[str, str] = {}
    per_task_s: dict[str, float] = {}

    for task in tasks:
        task_t0 = time.time()
        raw = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task.name],
            num_fewshot=task.num_fewshot,
            batch_size=batch_size,
            limit=limit,
            log_samples=False,
        )
        for subtask, metrics in raw.get("results", {}).items():
            all_results[subtask] = {
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in metrics.items()
                if isinstance(v, (int, float, str))
            }
        for k, v in raw.get("n-shot", {}).items():
            n_shot[k] = int(v) if isinstance(v, (int, float)) else v
        for k, v in raw.get("versions", {}).items():
            versions[k] = str(v)
        per_task_s[task.name] = time.time() - task_t0

    row = {
        "hf_id": hf_id,
        "revision": revision,
        "dtype": dtype,
        "batch_size": str(batch_size),
        "results": all_results,
        "n-shot": n_shot,
        "versions": versions,
        "timing": {
            "load_s": load_s,
            "per_task_s": per_task_s,
            "total_s": time.time() - t0,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(row, indent=2))
    return row
