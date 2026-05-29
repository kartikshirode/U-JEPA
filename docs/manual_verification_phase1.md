# Phase 1 manual verification

Run these checks after the Kaggle kernel finishes, either inside the same kernel
or in a fresh notebook that loads `results/phase1_continual.json`. The CPU test
suite already proves the math: gradient flow through the bank, the N-LoRA
penalty, label masking, the accuracy-row layout, and the BWT/forgetting formulas.
What it cannot prove is that the real NF4 Qwen3-14B actually learned each task and
that the orthogonality penalty kept the first task alive. That is what this doc is
for.

The result file is written by `scripts/02_train_continual_phase1.py` and printed
at the end of the phase1 notebook. Expected shape:

```python
{
  "model": "...",
  "task_order": ["fomc", "scienceqa_text"],
  "accuracy_matrix": [[...], [...]],   # row = stage, col = task by task_order
  "backward_transfer": float,
  "average_forgetting": float,
  "n_lora_lambda": float,              # the orth_weight actually used
  "collision_weight": float
}
```

Sections 1 to 3 and 5 use only the matrix and the metric scalars, which always
exist. Section 4 decodes a few prompts live, so it needs the model and bank still
in memory (run it in the same kernel right after training).

## 1. Load the result and print the matrix

```python
import json
from pathlib import Path

r = json.loads(Path("/kaggle/working/results/phase1_continual.json").read_text())
order = r["task_order"]
A = r["accuracy_matrix"]

print("task order:", order)
for i, row in enumerate(A):
    cells = "  ".join(f"{order[j]}={row[j]:.3f}" for j in range(len(row)))
    print(f"after stage {i} (trained up to {order[i]}): {cells}")
print("BWT:", r["backward_transfer"], " forgetting:", r["average_forgetting"])
print("n_lora_lambda:", r.get("n_lora_lambda"))
```

How to read it. `A[i][j]` is accuracy on task `order[j]` measured after training
through stage `i`. The diagonal `A[i][i]` is how well the model did on a task
right after learning it. The last row is final accuracy on everything. Column 0
is FOMC, column 1 is ScienceQA-text, always, regardless of anything else.

## 2. What the numbers should look like if the run is healthy

For the two-task FOMC then ScienceQA setup:

- **A[0][0] (FOMC right after learning it)** should be clearly above chance. FOMC
  sentiment is 3-class, so chance is about 0.33. Expect roughly 0.55 to 0.85. If
  it sits near 0.33 the adapter did not train at all.
- **A[0][1]** is the unseen-task placeholder, hard-coded to 0.0. Ignore it.
- **A[1][1] (ScienceQA right after learning it)** should beat its chance level
  too (multiple choice, chance ~0.25 depending on option count). Expect roughly
  0.45 to 0.80.
- **A[1][0] (FOMC after also learning ScienceQA)** is the whole point. With the
  N-LoRA penalty working it should stay close to A[0][0]. A small drop is normal.
  A drop of more than ~0.15 absolute is real forgetting.

Derived metrics:

- **backward_transfer** = A[1][0] - A[0][0]. Slightly negative is expected and
  fine. Near 0 or positive means the orthogonality penalty did its job.
- **average_forgetting** = max past FOMC accuracy minus final FOMC accuracy. With
  two tasks that is A[0][0] - A[1][0]. Small is good.

## 3. Failure signatures to watch for

- **Both rows of the matrix are nearly identical.** Task 1 training changed
  nothing measurable, i.e. the adapters are not training. Check that the active
  task was switched before stage 1, that hooks were installed, and that the LoRA
  params (not the frozen base) were handed to the optimizer. The CPU test
  `test_bank_gradient_flow.py` covers that contract; if the real matrix is flat
  despite that test passing, suspect the optimizer wiring in the script, not the
  bank.

- **A[1][0] drops to near 0 after task 1 (catastrophic forgetting).** FOMC went
  from, say, 0.75 down to ~0.05. The orthogonality penalty is not holding. This
  is the pivot-trigger condition: if forgetting is catastrophic with N-LoRA on,
  the shared-frozen-base plus orthogonal-adapters story is not buying retention
  and the approach needs rethinking. Before pivoting, confirm `n_lora_lambda` was
  non-zero and the penalty was actually added to the loss. A zero lambda reduces
  this to plain sequential LoRA, which is expected to forget.

- **Any diagonal entry is exactly 0.0 or exactly 1.0.** Almost always a broken
  matcher or decode, not a real score. Cross-check with the live decode in
  section 4 before believing it.

- **BWT is a large positive number.** Suspicious. Either the task is trivial or
  the eval set leaked into training. Decode a few examples (section 4).

## 4. Eyeball generations (catch a degenerate constant-answer model)

A model that always emits one label can still post a decent-looking accuracy
(always guessing the majority FOMC class scores ~0.4 to 0.5). The number alone
does not catch that; the prediction spread does. Run this with the model and bank
still loaded:

```python
from collections import Counter
from u_jepa.eval.continual import _match_generation

def spot_check(bank, tokenizer, task, items, n=8, device="cuda:0"):
    bank.activate(task)
    handles = bank.install_hooks()
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    preds = []
    try:
        for ex in items[:n]:
            enc = tokenizer(ex["prompt"], return_tensors="pt").to(device)
            out = bank.base.generate(**enc, max_new_tokens=8, do_sample=False,
                                     pad_token_id=pad)
            gen = tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            ok = _match_generation(gen, ex["target"])
            preds.append(gen.strip().split()[0] if gen.strip() else "<empty>")
            print(f"[{task}] gold={ex['target']!r}  gen={gen.strip()[:40]!r}  ok={ok}")
    finally:
        for h in handles:
            h.remove()
    print(f"[{task}] first-token spread:", Counter(preds))

# spot_check(bank, tokenizer, "fomc", eval_sets["fomc"])
# spot_check(bank, tokenizer, "scienceqa_text", eval_sets["scienceqa_text"])
```

What to look for:

- The first-token spread should have variety. If every prediction is the same
  label, the model collapsed to a constant answer and the accuracy is meaningless
  even if it looks decent.
- `gold` and `gen` should share a label vocabulary. A format mismatch means the
  matcher compares apples to oranges and every example scores wrong.
- A couple of `ok=True` rows where the prompt clearly supports the answer is the
  cheapest confirmation the model is reasoning, not pattern-matching the eval.

## 5. Cross-check FOMC retention directly

FOMC retention is the headline result, so read the two relevant matrix cells side
by side. These always exist, no live decode needed:

```python
A = r["accuracy_matrix"]
fomc_col = r["task_order"].index("fomc")
print("FOMC acc right after learning it   :", A[0][fomc_col])
print("FOMC acc after also learning task 1:", A[-1][fomc_col])
drop = A[0][fomc_col] - A[-1][fomc_col]
print("absolute FOMC drop:", round(drop, 3))
print("n_lora_lambda used:", r.get("n_lora_lambda"))
```

Read it like this:

- drop near 0: the orthogonality penalty held; FOMC was retained.
- drop up to ~0.15: normal, mild forgetting.
- drop large with FOMC ending near chance (~0.33): catastrophic forgetting. If
  `n_lora_lambda` was non-zero and this still happens, that is the pivot trigger.

## Note on GPU-only coverage

The following can only be checked on the real Kaggle run, which is why they live
here and not in the unit tests:

- That NF4 quantization plus the LoRA hooks produce sane logits on the 14B model
  (the CPU stub uses a tiny float Linear).
- That the base stays frozen end to end on GPU through bitsandbytes. The unit
  test asserts `base.grad is None` on a stub; the flat-matrix and forgetting
  checks here are the on-GPU proxy.
- That decode plus the answer matcher line up for real FOMC and ScienceQA
  outputs. Section 4's spread and gold/gen format checks cover this manually.
- That `n_lora_lambda` at its chosen value is strong enough to matter on a real
  14B adapter. Sections 2, 3, and 5's BWT and forgetting bands are the read.
