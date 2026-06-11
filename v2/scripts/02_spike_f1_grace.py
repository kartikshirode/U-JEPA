"""F1 spike: can EasyEdit's GRACE edit GPT-2-XL cleanly on a Kaggle T4?

This is the timeboxed spike that resolves open fork F1 in the build plan
(EasyEdit-GRACE vs a hand-rolled codebook). Its deliverable is a measurement
and a verdict, not a finished editing layer.

What it does, defensively (no step is allowed to hard-crash the run):
  1. Load GPT-2-XL through our own loader and tokenizer.
  2. Score a small inline set of CounterFact-style edits BEFORE editing
     (edit-success should be near zero) and the held-out perplexity baseline.
  3. Try to run EasyEdit GRACE on the same edits. If the import, the hparams,
     or the edit call fights us, capture exactly where and why.
  4. If GRACE ran, re-score the edited model: did the facts flip, did the
     neighborhood facts survive, did held-out perplexity stay put?
  5. Write a verdict JSON (KEEP_EASYEDIT / HANDROLL / INVESTIGATE) plus all the
     numbers and the captured error, so the decision is made from evidence.

The edit set is inline and tiny on purpose. The spike answers "does the
mechanism work at all," not "what is the GRACE baseline on zsRE" (that is a
separate Phase 0 deliverable with the real benchmark loaders).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from u_jepa_v2 import env, persistence  # noqa: E402
from u_jepa_v2.eval.edit_metrics import Edit, mean_heldout_perplexity, score_edits  # noqa: E402
from u_jepa_v2.models import core  # noqa: E402


# CounterFact-style edits: rewrite a true fact to a false target, then check the
# model now states the false target (success), nearby facts stay put (locality),
# and a paraphrase also flips (generalization).
EDITS = [
    Edit(
        prompt="The Eiffel Tower is located in the city of",
        subject="Eiffel Tower",
        target_new="Rome",
        ground_truth="Paris",
        paraphrases=["You can visit the Eiffel Tower in the city of"],
        neighborhood=[{"prompt": "The Louvre is located in the city of", "ground_truth": "Paris"}],
    ),
    Edit(
        prompt="The capital of Japan is",
        subject="Japan",
        target_new="Seoul",
        ground_truth="Tokyo",
        paraphrases=["Japan's capital city is"],
        neighborhood=[{"prompt": "The capital of China is", "ground_truth": "Beijing"}],
    ),
    Edit(
        prompt="The author of Romeo and Juliet is",
        subject="Romeo and Juliet",
        target_new="Charles Dickens",
        ground_truth="William Shakespeare",
        neighborhood=[{"prompt": "The author of Hamlet is", "ground_truth": "William Shakespeare"}],
    ),
    Edit(
        prompt="The chemical symbol for gold is",
        subject="gold",
        target_new="Pb",
        ground_truth="Au",
        neighborhood=[{"prompt": "The chemical symbol for silver is", "ground_truth": "Ag"}],
    ),
    Edit(
        prompt="The largest planet in our solar system is",
        subject="largest planet",
        target_new="Mars",
        ground_truth="Jupiter",
        neighborhood=[{"prompt": "The smallest planet in our solar system is", "ground_truth": "Mercury"}],
    ),
    Edit(
        prompt="The inventor of the telephone is",
        subject="telephone",
        target_new="Thomas Edison",
        ground_truth="Alexander Graham Bell",
        neighborhood=[{"prompt": "The inventor of the radio is", "ground_truth": "Guglielmo Marconi"}],
    ),
]

# Unrelated, stable text. Perplexity here should barely move after editing.
HELDOUT = [
    "The sun rises in the east and sets in the west each day.",
    "Water is made of hydrogen and oxygen, two atoms of the first and one of the second.",
    "A triangle has three sides, and its interior angles add up to one hundred eighty degrees.",
    "Reading books regularly tends to broaden a person's vocabulary over time.",
]


def run_easyedit_grace(edits: list[Edit], hparams_path: str):
    """Run GRACE through EasyEdit. Returns (edited_model, native_metrics).

    Raises on any failure; the caller captures the traceback. GRACE is a
    sequential editor by design, so sequential_edit=True.
    """
    from easyeditor import BaseEditor, GraceHyperParams

    hparams = GraceHyperParams.from_hparams(hparams_path)

    prompts = [e.prompt for e in edits]
    target_new = [e.target_new for e in edits]
    ground_truth = [e.ground_truth for e in edits]
    subject = [e.subject or e.prompt for e in edits]

    loc_prompts, loc_gt = [], []
    for e in edits:
        if e.neighborhood:
            loc_prompts.append(e.neighborhood[0]["prompt"])
            loc_gt.append(e.neighborhood[0]["ground_truth"])
        else:
            loc_prompts.append(e.prompt)
            loc_gt.append(e.ground_truth)
    locality_inputs = {"neighborhood": {"prompt": loc_prompts, "ground_truth": loc_gt}}

    editor = BaseEditor.from_hparams(hparams)
    native_metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        locality_inputs=locality_inputs,
        sequential_edit=True,
    )
    return edited_model, native_metrics


def _verdict(easyedit_ok: bool, baseline: dict, post: dict | None,
             base_ppl: float, post_ppl: float | None) -> tuple[str, str]:
    if not easyedit_ok:
        return ("HANDROLL", "EasyEdit did not import or run; hand-roll the codebook per the F1 fallback.")
    if post is None:
        return ("INVESTIGATE", "EasyEdit ran but post-edit scoring failed; read the error.")
    flipped = post["edit_success"] - baseline["edit_success"]
    loc = post.get("locality_preserved")
    ppl_ratio = (post_ppl / base_ppl) if (base_ppl and post_ppl and base_ppl == base_ppl) else None
    ok_flip = flipped >= 0.5
    ok_loc = (loc is None) or (loc >= 0.6)
    ok_ppl = (ppl_ratio is None) or (ppl_ratio <= 1.5)
    if ok_flip and ok_loc and ok_ppl:
        return ("KEEP_EASYEDIT", "GRACE flipped the facts, kept locality, and left perplexity stable.")
    reasons = []
    if not ok_flip:
        reasons.append(f"weak edit-success lift ({flipped:+.2f})")
    if not ok_loc:
        reasons.append(f"locality damaged ({loc})")
    if not ok_ppl:
        reasons.append(f"perplexity rose {ppl_ratio:.2f}x")
    return ("INVESTIGATE", "EasyEdit ran but: " + "; ".join(reasons))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-xl", choices=list(core.CORE_MODELS))
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument(
        "--hparams",
        default="/kaggle/working/EasyEdit/hparams/GRACE/gpt2-xl.yaml",
        help="path to EasyEdit's GRACE gpt2-xl hparams yaml (from a cloned EasyEdit repo)",
    )
    ap.add_argument("--out", default=None, help="results JSON path (defaults under the checkpoint dir)")
    args = ap.parse_args()

    result: dict = {"spike": "f1_grace", "env": env.summary().as_dict()}
    print(json.dumps(result["env"], indent=2, sort_keys=True), flush=True)

    # --- baseline: our loader + our metrics, before any edit -------------
    print(f"\n[load] {args.model} (dtype={args.dtype}, device={args.device})", flush=True)
    model, tok = core.load_core(args.model, dtype=args.dtype, device=args.device, cache_dir=args.cache_dir)

    print("[baseline] scoring edits + held-out perplexity before editing", flush=True)
    baseline = score_edits(model, tok, EDITS)
    base_ppl = mean_heldout_perplexity(model, tok, HELDOUT)
    result["baseline"] = {k: v for k, v in baseline.items() if k != "details"}
    result["baseline_perplexity"] = base_ppl
    result["baseline_details"] = baseline["details"]
    print(f"  baseline edit_success={baseline['edit_success']:.2f} "
          f"locality={baseline['locality_preserved']} ppl={base_ppl:.2f}", flush=True)

    # --- EasyEdit GRACE attempt -----------------------------------------
    easyedit_ok = False
    post = None
    post_ppl = None
    native_metrics = None
    error = None

    hparams_exists = Path(args.hparams).exists()
    result["hparams_path"] = args.hparams
    result["hparams_exists"] = hparams_exists
    if not hparams_exists:
        error = f"hparams yaml not found at {args.hparams}; did the notebook clone EasyEdit?"
        print(f"[easyedit] SKIP: {error}", flush=True)
    else:
        print("[easyedit] running GRACE...", flush=True)
        try:
            edited_model, native_metrics = run_easyedit_grace(EDITS, args.hparams)
            easyedit_ok = True
            print("[easyedit] edit() returned; scoring edited model with our metrics", flush=True)
            post = score_edits(edited_model, tok, EDITS)
            post_ppl = mean_heldout_perplexity(edited_model, tok, HELDOUT)
            print(f"  post edit_success={post['edit_success']:.2f} "
                  f"locality={post['locality_preserved']} ppl={post_ppl:.2f}", flush=True)
        except Exception as e:  # the spike's whole point is to discover this
            error = f"{type(e).__name__}: {e}"
            print(f"[easyedit] FAILED: {error}", flush=True)
            traceback.print_exc()

    result["easyedit_ran"] = easyedit_ok
    result["easyedit_error"] = error
    # EasyEdit's native metrics can be large/nested; keep them but guard JSON.
    try:
        json.dumps(native_metrics)
        result["easyedit_native_metrics"] = native_metrics
    except (TypeError, ValueError):
        result["easyedit_native_metrics"] = str(native_metrics) if native_metrics is not None else None
    if post is not None:
        result["post"] = {k: v for k, v in post.items() if k != "details"}
        result["post_perplexity"] = post_ppl
        result["post_details"] = post["details"]

    verdict, rationale = _verdict(easyedit_ok, baseline, post, base_ppl, post_ppl)
    result["verdict"] = verdict
    result["verdict_rationale"] = rationale
    print(f"\n[verdict] {verdict}: {rationale}", flush=True)

    out_path = args.out or str(persistence.kaggle_working_dir() / "results" / "f1_spike.json")
    persistence.save_json(out_path, result)
    print(f"[ok] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
