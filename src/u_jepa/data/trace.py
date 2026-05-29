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


def _log_first_keys(name: str, ds) -> None:
    """Print the keys of the first row so a schema mismatch is obvious in the
    Kaggle log instead of surfacing as a cryptic KeyError mid-training."""
    try:
        first = next(iter(ds))
        print(f"[data:{name}] first row keys: {sorted(first.keys())}")
    except Exception as e:  # pragma: no cover - logging only
        print(f"[data:{name}] could not inspect first row: {e}")


def _load_dataset_resilient(repo: str, split: str, configs: Iterable[str | None]):
    """Load a dataset trying several config names and a split fallback.

    Some HF datasets (e.g. FOMC) require a config name and only ship
    train/test, not validation. Try each config; if the requested split is
    missing, fall back to 'test' then 'train' so a missing-split error after
    10 minutes of model loading does not kill the whole run.
    """
    last_err: Exception | None = None
    split_fallbacks = [split]
    if split == "validation":
        split_fallbacks += ["test", "train"]
    elif split == "test":
        split_fallbacks += ["validation", "train"]
    for cfg in configs:
        for sp in split_fallbacks:
            try:
                if cfg is None:
                    return load_dataset(repo, split=sp)
                return load_dataset(repo, cfg, split=sp)
            except Exception as e:  # try the next combination
                last_err = e
    raise RuntimeError(
        f"could not load {repo} for split {split!r} with any of configs "
        f"{list(configs)}: {last_err}"
    )


# FOMC label can arrive as an int index or as a string already.
_FOMC_INT_MAP = {0: "dovish", 1: "hawkish", 2: "neutral"}
_FOMC_STR_SET = {"dovish", "hawkish", "neutral"}


def _fomc_label(raw) -> str:
    if isinstance(raw, str):
        low = raw.strip().lower()
        return low if low in _FOMC_STR_SET else "neutral"
    if isinstance(raw, bool):  # guard: bool is an int subclass
        return "neutral"
    if isinstance(raw, int):
        return _FOMC_INT_MAP.get(raw, "neutral")
    return "neutral"


def load_fomc(split: str = "train", n: int | None = None) -> list[dict]:
    """FOMC monetary policy stance: dovish / hawkish / neutral.

    Dataset: gtfintechlab/fomc_communication on HuggingFace. The repo is
    config-gated, so we try the common config names before the bare load.
    Fields seen: sentence/text (str), label (int 0=dovish, 1=hawkish,
    2=neutral) or a label string.
    """
    ds = _load_dataset_resilient(
        "gtfintechlab/fomc_communication",
        split,
        configs=("split_num_5_seed_5768", "fomc_communication", None),
    )
    _log_first_keys("fomc", ds)
    items: list[dict] = []
    for ex in ds:
        sentence = ex.get("sentence") or ex.get("text") or ex.get("sentences") or ""
        label = _fomc_label(ex.get("label", ex.get("labels", 2)))
        if not str(sentence).strip():
            continue
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
    ds = _load_dataset_resilient(
        "derek-thomas/ScienceQA", split, configs=(None,)
    )
    _log_first_keys("scienceqa_text", ds)
    letters = "ABCDEFGH"
    items: list[dict] = []
    for ex in ds:
        if ex.get("image") is not None:
            continue  # skip multimodal items for Phase 1
        choices = ex.get("choices") or ex.get("options") or []
        if not choices:
            continue
        question = ex.get("question") or ex.get("text") or ""
        if not str(question).strip():
            continue
        n_choices = min(len(choices), len(letters))
        choice_str = "\n".join(
            f"{letters[i]}. {choices[i]}" for i in range(n_choices)
        )
        answer_idx = ex.get("answer", ex.get("label", 0))
        if not isinstance(answer_idx, int) or isinstance(answer_idx, bool):
            continue
        if not (0 <= answer_idx < n_choices):
            continue
        items.append({
            "prompt": (
                f"Question: {question}\n{choice_str}\nAnswer:"
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
