"""Task list + few-shot settings.

We use the lm-evaluation-harness task names directly. The pilot list is the
intersection of (a) the tasks reported in the Pythia paper and (b) the Open LLM
Leaderboard v1 set, plus MMLU which gives 57 subtasks of inherent diversity.

Few-shot counts match the Open LLM Leaderboard / Pythia paper conventions so
our small-model numbers can be cross-checked against published values.

Notes:
- MMLU expands to 57 subtasks when lm-eval-harness sees `tasks=["mmlu"]`; each
  subtask returns its own metrics in the results dict.
- GSM8K is deliberately omitted from the pilot — it's a generation task, much
  slower per example, and its sub-billion-parameter accuracy is uninformative.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str             # lm-eval-harness task name
    num_fewshot: int
    primary_metric: str   # "acc" or "acc_norm" — which value to use as headline


TASKS: list[Task] = [
    Task("arc_easy",       0,  "acc_norm"),
    Task("arc_challenge",  25, "acc_norm"),
    Task("hellaswag",      10, "acc_norm"),
    Task("piqa",           0,  "acc_norm"),
    Task("winogrande",     5,  "acc"),
    Task("sciq",           0,  "acc_norm"),
    Task("boolq",          0,  "acc"),
    Task("openbookqa",     0,  "acc_norm"),
    Task("lambada_openai", 0,  "acc"),
    Task("mmlu",           5,  "acc"),
]


def task_names() -> list[str]:
    return [t.name for t in TASKS]


def fewshot_by_task() -> dict[str, int]:
    return {t.name: t.num_fewshot for t in TASKS}
