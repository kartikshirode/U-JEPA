"""TRACE benchmark sub-sequence loaders.

Phase 1 uses two contrasting tasks to make catastrophic forgetting visible:
  - fomc: financial policy stance classification (3 classes)
  - scienceqa_text: ScienceQA multiple-choice questions, text-only items
    (vision items skipped until Phase 3)

Each loader returns a list of dicts with keys 'prompt' and 'target', ready
for the PromptTargetDataset used by the continual training loop.
"""
from __future__ import annotations
from typing import Iterable

from datasets import load_dataset


def load_fomc(split: str = "train", n: int | None = None) -> list[dict]:
    """FOMC monetary policy stance: dovish / hawkish / neutral.

    Dataset: gtfintechlab/fomc_communication on HuggingFace.
    Fields: sentence (str), label (int 0=dovish, 1=hawkish, 2=neutral).
    """
    ds = load_dataset("gtfintechlab/fomc_communication", split=split)
    label_map = {0: "dovish", 1: "hawkish", 2: "neutral"}
    items: list[dict] = []
    for ex in ds:
        sentence = ex.get("sentence") or ex.get("text") or ""
        label = label_map.get(ex.get("label", 2), "neutral")
        items.append({
            "prompt": (
                "Classify the monetary policy stance as dovish, hawkish, or neutral.\n"
                f"Statement: {sentence}\nStance:"
            ),
            "target": label,
        })
        if n is not None and len(items) >= n:
            break
    return items


def load_scienceqa_text(split: str = "train", n: int | None = None) -> list[dict]:
    """ScienceQA text-only items (no image attached).

    Dataset: derek-thomas/ScienceQA on HuggingFace.
    Fields: question, choices (list), answer (int index), image (PIL or None).
    """
    ds = load_dataset("derek-thomas/ScienceQA", split=split)
    letters = "ABCDEFGH"
    items: list[dict] = []
    for ex in ds:
        if ex.get("image") is not None:
            continue  # skip multimodal items for Phase 1
        choices = ex.get("choices") or []
        if not choices:
            continue
        n_choices = min(len(choices), len(letters))
        choice_str = "\n".join(
            f"{letters[i]}. {choices[i]}" for i in range(n_choices)
        )
        answer_idx = ex.get("answer", 0)
        if not (0 <= answer_idx < n_choices):
            continue
        items.append({
            "prompt": (
                f"Question: {ex['question']}\n{choice_str}\nAnswer:"
            ),
            "target": letters[answer_idx],
        })
        if n is not None and len(items) >= n:
            break
    return items


TASK_LOADERS = {
    "fomc": load_fomc,
    "scienceqa_text": load_scienceqa_text,
}


def load_trace_task(name: str, split: str = "train", n: int | None = None) -> list[dict]:
    """Dispatch to the right loader by task name."""
    if name not in TASK_LOADERS:
        raise KeyError(f"unknown TRACE task: {name}. Known: {sorted(TASK_LOADERS)}")
    return TASK_LOADERS[name](split=split, n=n)
