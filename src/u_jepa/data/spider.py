"""Spider text-to-SQL loader producing NL and SQL paired views.

The Spider benchmark ships natural-language questions and the SQL that answers
them; these are two distinct views of the same underlying intent, which is the
exact shape the LLM-JEPA loss wants. We emit a 'view_a' (the NL prompt the
model sees during forward) and a 'view_b' (the SQL string used to compute the
target hidden state), plus the standard 'prompt'/'target' pair that the
PromptTargetDataset already understands.

Spider's HF repo has been moved around (the bare 'spider' name was deprecated
and replaced by org-prefixed mirrors). We try several known repo names so a
silent rename upstream does not break the run.
"""
from __future__ import annotations

from typing import Iterable

from datasets import load_dataset


SPIDER_REPOS: tuple[str, ...] = (
    "xlangai/spider",
    "spider",
    "Salesforce/spider",
)


def _load_spider_resilient(split: str, repos: Iterable[str] = SPIDER_REPOS):
    """Try several known Spider repo names and a split fallback chain.

    Spider does NOT publish a public test split; only train and validation
    are available. If the caller asks for test, fall back to validation
    rather than failing after a long load.
    """
    last_err: Exception | None = None
    split_chain = [split]
    if split == "test":
        split_chain.append("validation")
    if split == "validation":
        split_chain.append("train")
    for repo in repos:
        for sp in split_chain:
            try:
                return load_dataset(repo, split=sp, trust_remote_code=True)
            except Exception as e:
                last_err = e
    raise RuntimeError(
        f"could not load Spider for split {split!r} from any of {list(repos)}: "
        f"{last_err}"
    )


def _build_prompt(question: str) -> str:
    return f"Translate to SQL:\n{question}\nSQL:"


def load_spider_pairs(split: str = "train", n: int | None = None) -> list[dict]:
    """Return Spider items shaped for both PromptTargetDataset and the JEPA loop.

    Each item:
      prompt:  the NL question wrapped in a translate-to-SQL instruction
      target:  the gold SQL string (used by CE training and by exact-match eval)
      view_a:  the NL prompt (the side the model trains on)
      view_b:  the gold SQL string (the side we pool for the JEPA target)
    """
    ds = _load_spider_resilient(split)
    first_printed = False
    items: list[dict] = []
    for ex in ds:
        if not first_printed:
            print(f"[data:spider] first row keys: {sorted(ex.keys())}")
            first_printed = True
        # Spider's canonical fields are 'question' and 'query'. Some mirrors
        # rename them; cover the common variants without crashing the run.
        q = ex.get("question") or ex.get("question_text") or ex.get("nl")
        s = ex.get("query") or ex.get("sql") or ex.get("query_text")
        if not q or not s:
            continue
        q = str(q).strip()
        s = str(s).strip()
        prompt = _build_prompt(q)
        items.append({
            "prompt": prompt,
            "target": s,
            "view_a": prompt,
            "view_b": s,
        })
        if n is not None and len(items) >= n:
            break
    return items
