"""Build the pinned probe sets. Login node only, since it downloads.

Four general ability suites matching UltraEdit's evaluation set, plus a locality
set drawn from WikiBigEdit's own unrelated-neighbour columns. Written once,
checksummed, and then never regenerated: a probe set that shifts between cells
makes every cross-cell comparison meaningless and does it silently.

    python v3/scripts/01_build_probes.py --out ~/probes --n 200
    export U_JEPA_V3_PROBE_DIR=~/probes

Compute nodes on Baramati have no outbound network, so this runs on
aicoeserver01 and the output goes somewhere on /home that the jobs can read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

LETTERS = ("A", "B", "C", "D")

# Pin the revision so a rebuild six months from now produces the same file. The
# default is main, which moves.
SOURCES = {
    "sst": ("nyu-mll/glue", "sst2", "validation"),
    "mmlu": ("cais/mmlu", "all", "test"),
    "mrpc": ("nyu-mll/glue", "mrpc", "validation"),
    "nli": ("nyu-mll/glue", "rte", "validation"),
}


def _load(repo: str, config: str, split: str, revision: str | None):
    from datasets import load_dataset

    return load_dataset(repo, config, split=split, revision=revision)


def build_sst(rows) -> list[tuple[str, str]]:
    labels = ("negative", "positive")
    return [(f"Sentence: {r['sentence'].strip()}\n"
             "Question: Is this sentence positive or negative?\nAnswer:",
             labels[r["label"]]) for r in rows]


def build_mmlu(rows) -> list[tuple[str, str]]:
    out = []
    for r in rows:
        choices = "\n".join(f"{letter}. {choice}"
                            for letter, choice in zip(LETTERS, r["choices"]))
        out.append((f"{r['question'].strip()}\n{choices}\nAnswer:",
                    LETTERS[r["answer"]]))
    return out


def build_mrpc(rows) -> list[tuple[str, str]]:
    labels = ("no", "yes")
    return [(f"Sentence 1: {r['sentence1'].strip()}\n"
             f"Sentence 2: {r['sentence2'].strip()}\n"
             "Question: Do these two sentences mean the same thing?\nAnswer:",
             labels[r["label"]]) for r in rows]


def build_nli(rows) -> list[tuple[str, str]]:
    # RTE labels 0 as entailment and 1 as not entailment, which reads backwards.
    labels = ("yes", "no")
    return [(f"Premise: {r['sentence1'].strip()}\n"
             f"Hypothesis: {r['sentence2'].strip()}\n"
             "Question: Does the premise entail the hypothesis?\nAnswer:",
             labels[r["label"]]) for r in rows]


BUILDERS = {"sst": build_sst, "mmlu": build_mmlu, "mrpc": build_mrpc, "nli": build_nli}


def build_locality(n: int, seed: int) -> list[tuple[str, str]]:
    """Unrelated neighbours from WikiBigEdit's own loc and loc_ans columns.

    Using the corpus's own locality pairs rather than a generic set, because
    they were chosen to be near the edited facts without being them, which is
    what makes locality a real test instead of a general ability rerun.
    """
    from u_jepa_v3.data.wikibigedit import load_raw

    frame = load_raw()
    if "loc" not in frame.columns or "loc_ans" not in frame.columns:
        raise RuntimeError(
            f"WikiBigEdit has no loc/loc_ans columns; found {sorted(frame.columns)}. "
            "Locality has to come from somewhere else."
        )
    pairs = [(str(row["loc"]).strip(), str(row["loc_ans"]).strip())
             for row in frame.to_dict("records")
             if isinstance(row.get("loc"), str) and isinstance(row.get("loc_ans"), str)
             and row["loc"].strip() and row["loc_ans"].strip()]
    seen, unique = set(), []
    for prompt, answer in pairs:
        if prompt not in seen:
            seen.add(prompt)
            unique.append((prompt, answer))
    return random.Random(seed).sample(unique, min(n, len(unique)))


def sample(pairs: list[tuple[str, str]], n: int, seed: int) -> list[tuple[str, str]]:
    if n >= len(pairs):
        return pairs
    return random.Random(seed).sample(pairs, n)


def write_set(out_dir: Path, name: str, pairs: list[tuple[str, str]]) -> dict:
    path = out_dir / f"{name}.json"
    payload = json.dumps([list(p) for p in pairs], indent=1, ensure_ascii=False)
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(f"  {name:9} {len(pairs):5} pairs  sha256 {digest[:16]}")
    return {"n": len(pairs), "sha256": digest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="directory to write the sets into")
    parser.add_argument("--n", type=int, default=200, help="pairs per suite")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--revision", default=None,
                        help="dataset revision to pin; leave unset to take main and "
                             "record whatever that resolved to")
    parser.add_argument("--skip-locality", action="store_true",
                        help="skip the WikiBigEdit download, which is the slow part")
    args = parser.parse_args(argv)

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"n_requested": args.n, "seed": args.seed,
                "revision": args.revision, "sets": {}}

    print(f"writing to {out_dir}")
    for name, (repo, config, split) in SOURCES.items():
        rows = _load(repo, config, split, args.revision)
        pairs = sample(BUILDERS[name](rows), args.n, args.seed)
        manifest["sets"][name] = {
            **write_set(out_dir, name, pairs),
            "repo": repo, "config": config, "split": split,
        }

    if args.skip_locality:
        print("  locality  skipped")
    else:
        pairs = build_locality(args.n, args.seed)
        manifest["sets"]["locality"] = {
            **write_set(out_dir, "locality", pairs),
            "repo": "lukasthede/WikiBigEdit", "config": "loc/loc_ans", "split": "all",
        }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest written. Now:  export U_JEPA_V3_PROBE_DIR={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
