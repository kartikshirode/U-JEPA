# U-JEPA project: external-audit context and Phase 2 logic audit

Date: 2026-06-03
Author: Kartik Shirode (solo researcher)
Repo: github.com/kartikshirode/U-JEPA
Hardware: Kaggle free-tier T4 (primary, all training) + RTX 4060 laptop (local dev only)
Archive notice (2026-06-05): the v1 architecture and all Phase 0/1/2 code has been frozen under `legacy/v1/`. The forward-looking v2 plan lives in `U-JEPA v2/pipeline-1-build-plan.md`. Path references in this doc (e.g. `src/u_jepa/...`, `scripts/...`, `kaggle/...`) were accurate at the time of writing; the same files now live under `legacy/v1/src/...`, `legacy/v1/scripts/...`, etc. `results/` stays at the top level since the v1 experimental numbers are still useful data points. The earlier "central JEPA brain + sub-agents" spec at `docs/superpowers/specs/2026-06-03-u-jepa-v2-architecture.md` (now `legacy/v1/docs/superpowers/specs/...`) is also superseded by the new pipeline-1 plan and is kept for history only.

This document gives an external auditor everything they need to evaluate the project so far. It covers the research idea, the phase plan, what actually happened in each phase, the code layout, and a deliberate logic audit of Phase 2 (which failed both gates). Read top to bottom; later sections assume the earlier ones.

---

## 1. What U-JEPA is trying to do

**One-line pitch.** Build a continual-learning system on top of a frozen quantized LLM where new tasks are absorbed by orthogonal LoRA adapters, paired-view auxiliary losses (LLM-JEPA) align embeddings of related text views, a sliced-statistic regularizer (SIGReg, from LeJEPA) keeps representations from collapsing, and an optional V-JEPA vision bridge feeds visual prefixes into the same latent space.

**Why this combination.** Each piece exists separately in the literature. The bet is that they compose:

- N-LoRA / O-LoRA gives near-zero forgetting because adapters are orthogonal in parameter space.
- LLM-JEPA gives a self-supervised structural signal beyond next-token prediction.
- SIGReg gives an explicit anti-collapse pressure on the embedding distribution.
- V-JEPA + a small Q-Former projects vision into the LLM's latent space without re-training the LLM.

If they compose, you get a small base model that learns new tasks without forgetting, builds richer representations than CE alone, doesn't collapse, and can take vision input. That's a credible NeurIPS workshop or ICLR target on a free-tier cloud budget (Kaggle T4).

**Why this might not work.** Composition is the risky step. Each piece was validated in isolation, often on different model sizes and data. Whether the auxiliary losses still help when the base is frozen and the only trainable parameters are a tiny LoRA is an open empirical question. Phase 2 was the first test of that and the answer was no, at least under the configuration we tried. See section 8.

---

## 2. Approach choice

Original brainstorm covered multiple paths. The pivot decision on 2026-05-26 picked "Approach 1": fork LatentMAS (a vLLM-backed multi-agent latent-reasoning system from a recent paper), keep its agents and KV-cache plumbing, swap its base to Qwen3-14B, add a V-JEPA bridge, add N-LoRA on top, and use LeJEPA-flavored regularizers during training.

LatentMAS contributes the multi-agent latent-CoT runtime; we contribute the continual + multimodal stack.

The original plan briefly considered keeping everything on the RTX 4060 laptop with a 4B model. That framing was abandoned early in favor of running all heavy training on Kaggle's free T4 with Qwen3-14B (better headline numbers, no local VRAM ceiling). The laptop is for unit tests, design exploration, and writing; the training infrastructure has always been Kaggle. The pivot note lives at `docs/decisions/2026-05-26-kaggle-pivot.md`.

---

## 3. Phase plan (with status)

The plan document is `docs/superpowers/plans/2026-05-26-ujepa-prototype.md`. Summary with current status:

| Phase | Goal | Gate | Status |
|---|---|---|---|
| 0 | LatentMAS GSM8K baseline on Qwen3-14B-AWQ via vLLM | acc within 5pt of paper | **PASS**, 60.8% |
| 1 | Sequential LoRA on FOMC then ScienceQA, N-LoRA orth penalty | avg forgetting < 5% | **PASS**, 0.0% |
| 2 | LLM-JEPA + SIGReg on Spider, two-arm comparison | delta EM >= +2pt AND cond < 100 | **FAIL both** |
| 3 | V-JEPA 2 vision bridge via Q-Former | VQA within 5pt of CLIP baseline AND cos > 0.4 | not started |
| 4 | Router + adaptive orchestration + zero-retrain demo | qualitative demo | not started |
| 5 | Ablations + writeup | NeurIPS workshop submission | not started |

Hard time budget: 14 weeks from pivot. Pivot escape hatch: if Phase 3 misses by week 9, drop V-JEPA for SigLIP-base-256 and document the swap.

---

## 4. What actually ran (Phase 0)

**Phase 0:** reproduce LatentMAS GSM8K with Qwen3-14B-AWQ on Kaggle vLLM.

- Result: 60.8% (152/250) on a 250-row GSM8K slice. Within target.
- Took 14 kernel versions to get there. Cascade of issues fixed inline:
  - private repo to public so Kaggle could clone
  - GPU not enabled in metadata
  - HF cache overflowing the 20GB /kaggle/working quota, moved to /tmp
  - fp16 OOM with bare model, switched to AWQ
  - autoawq import error against newer transformers, added an activations shim
  - realign OOM, moved that computation to CPU
  - bs=1 squeeze bug, set batch_size=2
  - vLLM prefix-cache + prompt_embeds assertion failure, disabled prefix caching, set enforce_eager=True, custom no-pad batch path
  - prompt_embeds required after prefix caching off, re-enabled enable_prompt_embeds

Files: `scripts/01_repro_latentmas_gsm8k.py`, `kaggle/phase0/`, `results/phase0_baseline.json`. Manual verification at `docs/manual_verification_phase0.md`.

Honest note: this was infrastructure work, not science. The win is "the Kaggle path is reliable enough to run heavy training". The 60.8% number itself isn't novel.

---

## 5. What actually ran (Phase 1)

**Phase 1:** sequential LoRA continual learning with N-LoRA orthogonality penalty.

Two TRACE tasks in sequence: FOMC (financial dovish/hawkish/neutral) then ScienceQA-text (multiple choice A through H). 1500 train and 300 eval per task, 2 epochs, lr 3e-4, orth_weight 0.5, collision_weight 0.01. Single run on Kaggle T4.

Final result:

```
accuracy_matrix:
  [[0.753, 0.000],          # after FOMC trained: FOMC 75.3%, ScienceQA not yet seen
   [0.753, 0.950]]           # after ScienceQA trained: FOMC STILL 75.3%, ScienceQA 95.0%
average_accuracy   = 0.852
backward_transfer  = 0.0
average_forgetting = 0.0    # gate <= 0.05, PASS
wall time          = 6.2h
```

Headline observation: the FOMC accuracy after training ScienceQA was identical to the FOMC accuracy right after FOMC training. Byte-for-byte 226/300 both times. That's what the N-LoRA orthogonal constraint is supposed to do, and it did.

**Caveats worth flagging to the auditor.**

1. The bank uses `bank.activate(task_id)` per eval, so each task is scored with its OWN adapter. This is hot-swap evaluation, not joint inference. The orthogonality of the adapters in parameter space is real, but the "zero forgetting" headline partly reflects that we never ask both adapters to fire at the same time. The harder test (task-id-unknown routing) is a Phase 4 problem.
2. Deterministic greedy decode, so the byte-exact match is reproducible but does not tell us anything about distributional shifts that don't change the argmax.
3. n=300 eval is small. A Bernoulli 95% CI on p=0.753 with n=300 is roughly +/- 5pt. Sub-5pt drift would have been invisible.
4. Two tasks is a short sequence. Real continual-learning benchmarks run 10+ tasks. The orthogonality penalty has only one prior adapter to be orthogonal to here, so the test is easy.

Files: `scripts/02_train_continual_phase1.py`, `src/u_jepa/continual/orthogonal_lora.py`, `src/u_jepa/continual/n_lora_loss.py`, `src/u_jepa/train/continual_loop.py`, `src/u_jepa/eval/continual.py`, `results/phase1_continual.json`. Manual verification at `docs/manual_verification_phase1.md`.

---

## 6. What actually ran (Phase 2)

**Phase 2:** LLM-JEPA cosine loss + SIGReg on Spider, two-arm comparison.

Design: load Qwen3-14B once, run two fresh LoRA banks back to back on the same frozen base.
- Arm A: LoRA + standard CE on Spider question -> SQL.
- Arm B: same as A, plus an LLM-JEPA cosine loss between pooled hidden states of the NL question (view A) and the gold SQL (view B), plus a SIGReg loss over per-token last-hidden vectors.

Sizing: 800 train, 200 eval per arm, 2 epochs, lr 3e-4, lambda_jepa 0.5, lambda_sigreg 0.1, grad_accum 8, max_new_tokens 96, max_len 512. Chosen to fit two arms inside Kaggle's 9h cap.

Gates: treatment EM minus baseline EM >= +0.02 absolute, AND hidden-state covariance condition number < 100 on 64 eval prompts (treatment arm).

Result, after two cancelled runs and one successful end-to-end run:

```
baseline EM (arm A)                  = 0.065   (13/200)
treatment EM (arm B)                 = 0.060   (12/200)
delta_em                             = -0.005    gate: >= +0.02  FAIL
hidden cov condition number          = 4.12e13   gate: < 100      HARD FAIL
sigma_max / sigma_min                = 41.2 / 4.7e-14
sigreg_backend                       = lejepa (N-normalized)
wall time                            = 4.4h
```

Both gates failed. The condition-number gate failed by 11 orders of magnitude. The smallest singular value of the empirical pooled-hidden covariance was 4.7e-14, which is machine epsilon. The representation is effectively rank one.

Files: `scripts/03_train_jepa_aux_phase2.py`, `src/u_jepa/losses/llm_jepa.py`, `src/u_jepa/losses/sigreg.py`, `src/u_jepa/data/spider.py`, `src/u_jepa/train/jepa_aux_loop.py`, `src/u_jepa/eval/spider_em.py`, `results/phase2_jepa_aux.json`.

---

## 7. Phase 2 runtime history

Three Kaggle attempts before the final-and-failing run:

1. **Attempt 1 (cancelled by user, t=2.7h)**: notebook ran, training was silent because train_task had no per-step logging. User cancelled thinking it was hung. Diagnosis after pulling the log: it was actually training fine, just invisible. Fix landed: heartbeat every 25 steps with ce, sps, eta. Also added a chat-template wrapper to eval since it was feeding raw prompts to a chat-tuned model.

2. **Swarm audit (38 issues filed)**: a three-agent audit-fix-test cycle. Real bugs caught:
   - View-B forward was running THROUGH the active adapter, so the JEPA target was tracking the very parameters being trained. Fixed by pre-pooling view-B once with the adapter deactivated and caching on CPU.
   - LeJEPA returns an N-scaled statistic; fallback returns a mean. Magnitudes differed ~500x. Fixed by normalizing.
   - bank_a left LoRA adapters and AdamW moments resident in VRAM during arm B. Fixed with del + gc + empty_cache.
   - Spider loader's validation -> train fallback could silently turn eval into train-on-train. Removed.
   - Notebook was pseudo-XML, not nbformat JSON. Kaggle CLI initially accepted it on push but the FIX agent later confirmed Phase 1's notebook was real JSON and rewrote Phase 2's to match.
   - SIGReg was reconstructing the lejepa module per step; cached it.
   - Several others, see `docs/phase2_swarm_audit.md` if it exists; otherwise the commit messages on 8c203c9..6248c54.

3. **Attempt 2 (errored, t=2.4h)**: crashed in arm B's first SIGReg call with `expected mat1 and mat2 to have the same dtype, but got: c10::Half != float`. The cached lejepa module's projection-direction buffers are fp32. On T4 with NF4, activations come out in fp16 (bnb downgrades from bf16 since T4 lacks bf16 tensor cores). The matmul mismatched. Arm A completed (EM 0.055), arm B died on step 1 of JEPA training. Fix: upcast embeddings to fp32 inside `sigreg_loss` before the lejepa call. Added parametrized regression tests for fp16 and bf16. Commit `6ca2a30`.

4. **Attempt 3 (gates failed cleanly, t=4.4h)**: no runtime errors. Full pipeline ran. Both gates failed. Numbers in section 6.

---

## 8. Phase 2 logic audit

This is a deliberate audit of the DESIGN, not the runtime. Runtime is now clean. The question is why Phase 2 failed as designed and what about it was wrong from the start.

### 8.1 The JEPA target was structurally collapsed

The treatment arm's JEPA loss matches the pooled last-hidden state of the NL question (view A) to the pooled last-hidden state of the gold SQL (view B). All SQL targets in Spider start with "SELECT ", proceed through similar token distributions, and end with the same syntactic shape. When you mean-pool a 14B model's last hidden over those sequences, the pooled vectors cluster tightly.

If the target distribution is approximately a single point in embedding space, the JEPA cosine loss is asking the predictor to map every view-A embedding to that one point. That is collapse by design. The loss can go to near zero by mapping all view-A embeddings to a constant, which is exactly what happened. final_jepa was 0.039, near optimal, and the cond number went to 4e13.

Verdict: this is a real conceptual error. The LLM-JEPA paper used different VIEWS of the same content (e.g. mask vs original), not different conditional samples from a joint distribution. Question/answer pairs are not views in that sense. They are causally related, but they have different content.

Fix space: either (a) use textual augmentations of the question itself as paired views (paraphrase the NL, mask spans), or (b) drop the JEPA piece for sequence-to-sequence tasks and apply it only to tasks where natural views exist (e.g. image patches, masked/unmasked sentence pairs).

### 8.2 SIGReg granularity was wrong

We applied SIGReg to per-token last-hidden vectors within a single example. The paper applies it to per-sample representations across a batch dimension. These are very different signals.

Per-token vectors within one sequence are NEVER isotropic. They have positional structure (early tokens encode prompt, later tokens encode generation context), content correlation, and rotary-positional-encoding determined geometric structure. Asking SIGReg to push them toward isotropic Gaussian is asking it to fight the model's intended representation structure.

We made this choice because batch_size=1 leaves no across-sample variation to regularize. The correct response is not "apply SIGReg to a different axis," it is "use a rolling buffer of pooled embeddings across recent micro-batches and apply SIGReg to that". With grad_accum=8 we have 8 micro-batches per optimizer step. Buffer the 8 pooled vectors, run SIGReg on the (8, D) matrix once per optimizer step, contributes one differentiable scalar per step. Cleaner and correct.

Verdict: the per-token SIGReg as configured was not a useful anti-collapse regularizer. It cost compute without buying signal.

### 8.3 SIGReg magnitude was under-calibrated

After the N-normalization fix, observed SIGReg values during training were around 0.2. With lambda_sigreg=0.1, the per-step contribution to total loss was ~0.02. CE was 0.1 to 0.6 and JEPA was ~0.05. The SIGReg gradient was the smallest of the three by an order of magnitude.

Even if the per-token granularity were correct (it isn't), the weight was too low to fight CE's pull toward collapse.

Verdict: lambda_sigreg=0.1 was a guess inherited from the plan. It should have been swept on a small subset first.

### 8.4 CE dominated and drove memorization

final_ce on arm B reached 0.004 at intermediate steps before bouncing back to 0.40. A 14B model with rank-16 LoRA on q_proj/v_proj over 800 examples has more than enough capacity to memorize those 800 prompt-target pairs. With batch_size=1 the gradient signal is high variance and the model overfits to individual examples one at a time.

In that regime, an auxiliary loss with 10% the weight of CE is noise. The model fits CE quickly and the rest is regularization that the model can mostly ignore.

Verdict: either use a bigger batch (not feasible on T4 with this model), or reduce capacity (lower LoRA rank), or train fewer steps with stronger regularization. The current setup hands the optimizer too much rope.

### 8.5 The TiedPredictor is fragile

TiedPredictor applies the SAME 5120x5120 Linear k=3 times. That is mathematically W^3 plus repeated bias additions. If W's spectral radius is greater than 1, repeated application amplifies; if less than 1, it dampens. With AdamW updates in bf16, the spectrum can drift in ways that interact poorly with the iteration.

The original LLM-JEPA paper uses a stack of DIFFERENT linears (one per [PRED] token), not a tied chain. We tied the weights to save parameters. That tradeoff is paid in stability.

Verdict: replace with three independent linears (cost: 3x the parameters, still under 1GB) or just one linear (k=1, drops the iteration entirely). The current k=3 tied design is the worst of both.

### 8.6 No baseline of the base model

We compared arm A (LoRA + CE) against arm B (LoRA + CE + JEPA + SIGReg). We never measured what the frozen base model gets on the same Spider eval. If the frozen base gets 8% and both arms get 6-6.5%, both treatments are HURTING performance. The delta gate might be measuring which adapter hurts less, not which helps more.

Verdict: a "no-adapter" arm is missing. Cheap to add: same eval script with `bank._active = None` for all hooks (so the model passes through unmodified) before generation.

### 8.7 Spider without schema is a wrong-benchmark choice

Real Spider performance for any model requires the database schema in the prompt. Our prompt is just "Translate to SQL: <question>\nSQL:". The model has no way to know the table is called "singer" vs "vocalists" vs "performers". 5-7% accuracy is roughly what you get from guessing common table names. Both arms hit a ceiling determined by the absence of schema, not by the training method.

This makes Spider EM an insensitive metric for whether the auxiliary losses are doing anything. The delta could be -2pt or +2pt purely from guessing-luck variance.

Verdict: either include the schema in the prompt (raises the ceiling, makes the metric sensitive again) or pick a different benchmark where the metric responds to representation quality without requiring external knowledge.

### 8.8 Eval n=200 is too small to detect a 2pt effect

With baseline p=0.065 and n=200, a 95% Wilson CI is roughly (0.038, 0.108). To distinguish baseline 0.065 from treatment 0.085 (a 2pt absolute lift) with 80% power, you need roughly n=2000+ per arm. Our gate requires +2pt over n=200, which is below the noise floor. Even a real +2pt effect would have been missed half the time.

Verdict: the gate is statistically underpowered. We chose n=200 to fit the time budget. Either accept that the gate is noisy and report confidence intervals alongside, or run a larger eval (Spider validation has 1034 examples; using all of them roughly doubles eval time per arm but stays within 9h).

### 8.9 We measured cond number only on the treatment arm

The collapse probe runs on the treatment arm only. We have no number for the baseline. If LoRA fine-tuning with batch=1 on a frozen 14B already drives the representation toward low rank, both arms might be collapsed and JEPA isn't to blame.

Verdict: cheap fix, run the cond probe on both arms and report both numbers. The current design cannot attribute collapse to the JEPA configuration.

### 8.10 The chat-template wrapper for view B may be wrong

For view B (the SQL string), we wrap it in the chat template as if SQL were the USER's message. The pooled last-hidden then reflects "what does Qwen think about a user who typed raw SQL?" which is off-distribution. The pooled embedding is then dominated by the assistant-turn artifact at the end of the template, not by the SQL itself.

After the cache refactor (during the swarm audit) we DROPPED the chat template for view-B and feed it as raw text. That's better, but raw SQL is still off-distribution for a chat model. The honest answer is "we don't know what the pooled embedding of view B represents semantically".

Verdict: view B's embedding has no validated semantic meaning. Matching view-A to it is matching to noise plus the SQL surface form. Combined with 8.1, this whole branch needs a redesign.

### 8.11 Predictor has no weight decay

AdamW was constructed with default args, which include `weight_decay=0.01` unless overridden. Actually I'm wrong on the default; PyTorch's AdamW default IS 0.01. So this is fine. Strike this item, but note that we never confirmed it.

Verdict: not a bug, but the optimizer config is implicit. Make it explicit so the reviewer doesn't have to dig.

### 8.12 Two-arm comparison shares the same frozen base

bank_a and bank_b are independent ModuleDicts on top of the same Qwen3-14B instance. The shared base means the only difference between arms is the LoRA training trajectory. That's actually the right experimental design (clean delta), as long as VRAM doesn't leak between arms. We confirmed the VRAM cleanup in the swarm audit. Not a bug, but worth naming the assumption.

Verdict: design is sound.

### 8.13 Single seed

Every Phase 2 number is from one run with one seed. No variance estimate. Treatment - baseline = -0.5pt could easily flip to +0.5pt on a re-run with a different shuffle. We have no way to separate signal from seed noise.

Verdict: at minimum, run two seeds and report the range. Ideal: three seeds, report mean +/- 1 SD. The 9h Kaggle budget is the constraint here.

### 8.14 Summary of logic flaws by severity

- Critical (kills the experiment): 8.1 (JEPA target collapse by design), 8.2 (SIGReg wrong axis), 8.7 (no-schema Spider is wrong benchmark), 8.8 (eval n=200 underpowered for +2pt gate).
- High (makes results uninterpretable): 8.4 (CE dominates), 8.6 (no base-model baseline), 8.9 (cond only on treatment), 8.10 (view-B has no validated semantics), 8.13 (single seed).
- Medium (stability/clarity): 8.3 (lambda not swept), 8.5 (TiedPredictor design).
- Trivial: 8.11 (implicit AdamW config).

If you only fix 8.1, 8.2, and 8.7, Phase 2 becomes a legitimately interesting experiment. As shipped, it is a thorough engineering exercise that demonstrates we can run two-arm comparisons reliably on Kaggle, and that the auxiliary losses as specified do not transfer to LoRA fine-tuning on a strongly-prompted task.

---

## 9. Decisions an external reviewer should weigh in on

1. **Should Phase 2 be redone?** A redone Phase 2 with the four critical fixes (textual-view augmentation, SIGReg over rolling buffer, schema-augmented Spider prompt, n>=1000 eval) is roughly two more Kaggle runs (~10h total). Cost is the project clock.

2. **Or should Phase 2 be reframed as an ablation?** The current numbers can be presented honestly in the paper as "naive composition of JEPA + SIGReg on top of LoRA does not improve a strongly-prompted seq2seq task and exhibits representational collapse under per-token SIGReg". That is a publishable negative result if it ships alongside positive Phase 3+4 results.

3. **Skip to Phase 3?** Phase 3 (V-JEPA vision bridge) does not depend on Phase 2's auxiliary-loss choice. It would use V-JEPA features directly. Moving on lets us still hit Phase 3 + Phase 4 (router + zero-retrain demo) inside the original 14-week budget. Phase 2 redo could happen in the ablation phase.

4. **The N-LoRA bank API.** Phase 1 worked but the bank's interface has internal access patterns (`bank._active`, `bank.adapters[task_id]`) that callers use directly. The swarm audit caught a CRIT bug (`bank._active` not being deactivated for view-B forward) precisely because the API doesn't enforce the invariant. A small interface upgrade (`bank.suspend()` context manager) would prevent that whole class of bug.

5. **TRACE vs CL-Benchmark.** Phase 1 used 2 TRACE tasks. The plan implies more tasks later (Phase 4 uses the bank with router and adaptive adaptation). Two-task continual learning is the easiest possible test. The auditor should ask whether the project plans to extend to 5-10 tasks before claiming the orthogonality story works.

---

## 10. Repo map

```
U-JEPA/
  Research.md                              one-page idea
  JEPA_Research_Context.md                 earlier context dump
  compass_artifact_*.md                    a literature scan
  pyproject.toml                           package metadata
  requirements.txt / requirements-kaggle.txt
  docs/
    decisions/2026-05-26-kaggle-pivot.md   the only architectural decision doc
    manual_verification_phase0.md
    manual_verification_phase1.md
    superpowers/plans/2026-05-26-ujepa-prototype.md   the master plan
    external_audit_context.md              this file
  scripts/
    00_smoke_env.py                        env detection sanity
    01_repro_latentmas_gsm8k.py            Phase 0 entry
    02_train_continual_phase1.py           Phase 1 entry
    03_train_jepa_aux_phase2.py            Phase 2 entry
  src/u_jepa/
    config.py                              QwenConfig, LoraConfig, HardwareConfig
    util/
      env.py                               kaggle vs local detection, results_dir
      prompting.py                         chat-template wrapper for Qwen3
    models/qwen_base.py                    NF4 loader + VRAM helpers
    data/
      trace.py                             FOMC + ScienceQA-text loaders
      spider.py                            Spider NL/SQL pairs (Phase 2)
    continual/
      orthogonal_lora.py                   the LoRA bank with forward hooks
      n_lora_loss.py                       orthogonality + non-collision penalty
    losses/
      llm_jepa.py                          TiedPredictor + cosine/MSE
      sigreg.py                            lejepa wrapper + fallback Epps-Pulley
    train/
      continual_loop.py                    Phase 1 CE loop with N-LoRA penalty
      jepa_aux_loop.py                     Phase 2 CE + JEPA + SIGReg loop
    eval/
      continual.py                         per-task eval matrix
      metrics.py                           BWT, forgetting, avg accuracy
      spider_em.py                         Spider EM proxy + cond-number probe
  tests/                                   122 tests, 1 network-gated skip
  vendored/
    LatentMAS/                             upstream multi-agent latent runtime
    lejepa/                                upstream SIGReg implementation
    llm-jepa/                              upstream paired-view loss reference
    N-LoRA/                                upstream orthogonal LoRA reference
    Online-LoRA/                           upstream online adaptation reference
  kaggle/
    phase0/                                Kaggle notebook + metadata
    phase1/
    phase2/
  results/
    phase0_baseline.json                   60.8%
    phase1_continual.json                  0.0 forgetting, 0.852 avg acc
    phase2_jepa_aux.json                   gates FAIL, see section 6
```

---

## 11. Open questions for the auditor

The questions below are the ones I genuinely don't know the answer to. An external reviewer's pushback on any of these would change the project's direction.

1. Is the JEPA-on-Q/A-pairs framing salvageable, or does it have to be dropped for Phase 2-style tasks? See 8.1, 8.10.
2. Is SIGReg even the right anti-collapse tool when most of the model is frozen? VICReg or BarlowTwins-style covariance penalties might be cheaper and more direct.
3. The N-LoRA orthogonality penalty was tested with two tasks. Does it scale to 5-10 sequential tasks without an explosion of penalty terms? The penalty is O(num_tasks) per step.
4. Is the cond-number probe a meaningful collapse metric, or is it dominated by the choice of pooled-vs-per-token and the choice of probe prompts? See 8.9.
5. The hardware budget (single T4, no bf16, 16GB) is a hard constraint. Are there steps in the plan that just won't fit and we should know now instead of in Phase 3?
6. The composition story (orth-LoRA + JEPA + SIGReg + V-JEPA) is the project's contribution. If Phase 2 is dropped or reframed, what's the strongest argument for the composition still being interesting?
7. NeurIPS workshop vs ICLR main track: the latter wants larger experiments and ablations. Given the Kaggle budget, is workshop the realistic target?

---

## 12. What "done" looks like

Original goal: a working U-JEPA prototype that demonstrates near-zero forgetting on continual learning, a measurable representation-quality lift from the auxiliary losses, working vision input, and an adaptive router. Submit to a NeurIPS workshop in late 2026.

Honest status:
- Continual-learning piece: shown on 2 tasks. Need 5+ to claim it.
- Aux-loss piece: shown not to work as configured on Spider. Needs redesign or reframing.
- Vision piece: not started, hardest, highest risk, has a documented SigLIP escape hatch.
- Router + zero-retrain: not started, depends on the above two.

The project is on track for an honest paper if Phase 3 and 4 land. If Phase 3 misses too, the paper becomes "we tried to compose four ideas and here's what didn't work and what we learned", which is still a workshop-acceptable contribution if framed correctly.

---

End of context.
