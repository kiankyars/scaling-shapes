"""Run lm-evaluation-harness over a single Pythia (size, checkpoint) pair.

One call loads the model once (via HFLM), then runs each task at its
configured few-shot count. Designed to be called once per Modal function
invocation so the grid parallelizes naturally across (size, revision) pairs.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

from .tasks import TASKS, Task


def _patch_safetensors_metadata_for_transformers_446() -> None:
    """Some Pythia 2.8B revisions upload safetensors shards with a None
    metadata block. transformers 4.46.x then raises either AttributeError
    (None.get) or ValueError ("File metadata is not ['pt','tf',...] but
    None"). Wrap safe_open inside transformers.modeling_utils so .metadata()
    always returns a dict with format='pt' filled in.

    Idempotent — re-import inside a hot worker is a no-op.
    """
    import transformers.modeling_utils as _mu

    if getattr(_mu, "_pythia_metadata_patch_applied", False):
        return

    real_safe_open = _mu.safe_open

    class _SafeOpenWrap:
        def __init__(self, *args, **kwargs):
            self._cm = real_safe_open(*args, **kwargs)
            self._inner = None

        def __enter__(self):
            self._inner = self._cm.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._cm.__exit__(exc_type, exc, tb)

        def metadata(self):
            md = self._inner.metadata()
            if md is None:
                return {"format": "pt"}
            if md.get("format") is None:
                md = dict(md)
                md["format"] = "pt"
            return md

        def __getattr__(self, name):
            return getattr(self._inner, name)

    _mu.safe_open = _SafeOpenWrap
    _mu._pythia_metadata_patch_applied = True


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

    _patch_safetensors_metadata_for_transformers_446()

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

    import torch  # local import to avoid hard dep at module level

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
        # lm-eval-harness retains per-request tensors on the LM instance —
        # without an explicit cache flush each task accumulates ~GB on the
        # GPU. Clear between tasks so memory stays bounded.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
