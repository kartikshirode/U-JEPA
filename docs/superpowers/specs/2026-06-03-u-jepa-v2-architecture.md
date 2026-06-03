# U-JEPA v2: central JEPA brain + domain sub-agents (design + implementation plan)

Date: 2026-06-03
Author: Kartik Shirode (solo researcher)
Status: design draft, awaiting user resolution of 3 open questions before plan finalization
Supersedes: `docs/superpowers/plans/2026-05-26-ujepa-prototype.md` for the architectural goal. Old plan's Phase 0 and Phase 1 deliverables carry over; Phase 2 onward gets rewritten under this design.

This is the consolidated record of a brainstorm session that reframed U-JEPA from "continual LoRA + auxiliary losses + vision bridge on a frozen LLM" to "central JEPA brain that dispatches to domain-specialist JEPA sub-agents, all reasoning in a shared latent space, with one frozen text decoder at the end". The new framing is a structural redesign, not a tweak. Three downstream design questions remain open and are listed in section 7 with the options that were discussed but not yet picked.

---

## 1. What we are building

### 1.1 The pitch

A new LLM-style architecture that replaces the monolithic transformer with a central JEPA brain that classifies an input, dispatches to one or more domain-specialist JEPA sub-agents, lets them reason and communicate in a shared latent space for a few rounds, synthesizes the result, and decodes once at the end through a frozen non-trainable text head.

The architecture targets three concrete pain points of stock transformer LLMs:

1. Constant retraining for new information. New domain = add a new sub-agent and a small orthogonal adapter on the central agent. Nothing else changes.
2. Token-based reasoning. JEPA agents reason in continuous latent space across multiple iterative steps. Tokens only appear at the very end when the decoder runs.
3. Context window. Long problems decompose into multiple latent reasoning steps and into multiple sub-agents working on subproblems. The effective working set at any instant is the current latent and the per-query blackboard, not the entire history of tokens.

The architecture is a structured composition of pieces that exist in the literature: JEPA-family self-supervised prediction, LatentMAS-style multi-agent latent reasoning, N-LoRA-style orthogonal continual learning, V-JEPA-style vision encoding. The novelty is in how they compose, not in any single piece.

### 1.2 The diagram

```
                       Input (text, image, ...)
                                 |
                                 v
                  [Input encoders, one per modality]    <-- open Q1
                                 |
                                 v
                +----------------------------------------+
                |  Central JEPA agent (the conductor)    |
                |  - Dispatch phase: reasons about input,|
                |    picks K sub-agents, sets up the     |
                |    blackboard, decides N rounds        |
                +----------------------------------------+
                                 |
                                 v
                       Per-query Blackboard
                  (shared public-subspace latent state)
                                 |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
       Sub-agent A         Sub-agent B         Sub-agent C
       (Finance JEPA)      (Math JEPA)         (Coding JEPA)
              |                  |                  |
              +--- round 1: parallel, read+write ---+
              +--- round 2: parallel, read+write ---+
              +---            ... N rounds       ---+
                                 |
                                 v
                       Final blackboard state
                                 |
                                 v
                +----------------------------------------+
                |  Central JEPA agent (the synthesizer)  |
                |  - Reads final blackboard              |
                |  - Runs latent steps to merge          |
                |  - Emits ONE coherent final latent     |
                +----------------------------------------+
                                 |
                                 v
                       Frozen text decoder       <-- open Q2
                                 |
                                 v
                              Output text
```

### 1.3 What the architecture explicitly is NOT

To avoid the audit confusion that happened with Phase 2, the design states its non-goals up front.

- Not a Mixture of Experts (MoE) in the standard sense. MoE has a gating network choosing among parallel experts inside one model. Here the sub-agents are FULL JEPA models with their own latent reasoning loops, not feed-forward experts inside a transformer block.
- Not a tool-use LLM. The sub-agents are not tools called by a tool-using LLM. The central agent is itself a JEPA reasoner, and the communication is in latent space, not in textual function calls.
- Not a multi-LLM ensemble. Each sub-agent is a JEPA model, structurally different from an LLM. The decoder is the only LLM-shaped piece in the whole system, and it is frozen and tiny relative to a full LLM.
- Not a replacement for the transformer EVERYWHERE. The decoder is a frozen transformer (or piece of one). What we are replacing is the reasoning core, not the language head.

---

## 2. Component specifications

### 2.1 Central JEPA agent (the conductor + synthesizer)

A full JEPA iterative reasoner that plays two roles in every query.

**Dispatch role.** Given the encoded input, the central agent runs K_central latent reasoning steps. The state at the end of those steps encodes (a) which sub-agents are relevant for this query, (b) what initial state the blackboard should hold, (c) how many sub-agent rounds N to allocate. These three outputs are read off the central agent's final-step latent via learned linear heads.

**Synthesis role.** After sub-agents finish their N rounds, the central agent re-enters. It reads the final blackboard, runs K_synth more latent reasoning steps with the blackboard as additional input context, and emits ONE coherent final latent that goes to the decoder.

**Mechanism (iterative latent updates, option A from the brainstorm).** At each step, the central agent's "predictor" network maps the current latent + context to the next latent. The predictor is a small transformer block trained via JEPA-style prediction: at training time, the target latent for step t+1 is provided by a frozen teacher LLM's hidden state trajectory on chain-of-thought data. The predictor learns to imitate that trajectory. At inference, the predictor runs autonomously for K steps.

**Training source.** Teacher distillation. Pick a strong frozen LLM (candidate: Qwen3-14B, same as Phase 0/1). Run it on a curated mix of chain-of-thought data (math, code, general reasoning). Capture its hidden-state trajectory at each layer or at chosen checkpoints. The central agent's predictor is trained to step from latent z_t to a target z_{t+1} from the teacher's trajectory, with the target detached (standard JEPA pattern).

**Continual learning.** The central agent grows via N-LoRA orthogonal adapters. Each adapter encodes "how to dispatch and synthesize for a NEW domain". When a new sub-agent is added, train one new adapter on the central agent (small data, small cost), and the orthogonality penalty keeps prior dispatch knowledge intact.

**Inputs.**
- Encoded input from input encoders (modality-agnostic latent)
- The list of currently-available sub-agents (their domain prototypes, registered at install time)

**Outputs.**
- Dispatch decision: which K sub-agents, initial blackboard state, N rounds
- Synthesis output: one final latent for the decoder

### 2.2 Sub-agent JEPA model (one per domain)

Each sub-agent is also a full JEPA iterative reasoner. Structurally similar to the central agent but trained for a specific domain (Finance, Coding, Math, Image Generation, Law, etc.).

**Internal latent structure.** Each sub-agent's latent space has two named subspaces:
- Private subspace (~75% of dimensions): the sub-agent's domain-specific working memory. Other sub-agents never read this.
- Public subspace (~25% of dimensions): aligned across all sub-agents. This is the communication channel via the blackboard.

The split is enforced architecturally by treating the latent as concatenation of two vectors, OR by learned linear projections to/from the shared public space. Implementation choice in section 3.4.

**Alignment loss.** During training, the public subspace is regularized to be CONSISTENT across sub-agents. Concretely: for any input that lands in the shared "general" pool of training data, the public-subspace projections of two different sub-agents should be similar. This is an explicit alignment loss applied during co-training (or via a shared anchor model that all sub-agents distill from).

**Per-round behavior.** In one blackboard round, a sub-agent:
1. Reads the current blackboard public-subspace state
2. Runs K_sub internal latent steps using its full latent (private + public)
3. Writes the public-subspace portion of its final-step latent back to the blackboard

The blackboard then aggregates the writes from all participating sub-agents (sum, attention-weighted mean, or learned merge; see 3.5).

**Training source.** Teacher distillation, same as central agent, but on DOMAIN-SPECIFIC data. The Finance sub-agent is trained on finance chain-of-thought trajectories from the teacher; the Coding sub-agent on code-execution traces; the Math sub-agent on math derivations; etc.

**Continual learning within a sub-agent.** This is the existing N-LoRA story from Phase 1. A given sub-agent (say Finance) can absorb new sub-tasks (FOMC, then ScienceQA-finance subset, then options pricing) via orthogonal adapters within the sub-agent. The Phase 1 mechanism is reused unchanged inside each sub-agent.

### 2.3 Blackboard (per-query shared state)

A buffer of public-subspace latent vectors scoped to ONE query.

**Shape.** A matrix of (num_slots, d_public) on the same device as the sub-agents. Initialized by the central agent at dispatch time. Updated by sub-agents at each round.

**Per-query scoping.** The blackboard is created fresh per query and discarded after. Two concurrent queries have two independent blackboards. This avoids the "central agent is the bottleneck" problem the user raised: sub-agents read/write THEIR query's blackboard without going through the central agent.

**Round-update protocol (chosen mechanism).** At round r:
1. All selected sub-agents READ the current blackboard state simultaneously
2. Each sub-agent computes its public-subspace contribution
3. The contributions are MERGED into the next blackboard state via a fixed or learned merge function (3.5)

**Termination.** Either fixed N rounds (decided by central agent at dispatch) or a learned stop predicate. v1 implementation uses fixed N from the central agent's dispatch output. Adaptive stopping is a v2 feature.

### 2.4 Frozen text decoder

A frozen, never-retrained module that maps the central agent's final synthesized latent to text tokens. Structural form is OPEN QUESTION Q2 (section 7).

**Common requirements regardless of structural choice:**
- Reads ONE latent (the central agent's synthesized output)
- Produces tokens autoregressively
- Has a fixed vocabulary tied to the teacher LLM's tokenizer
- Never trained or fine-tuned after the initial freeze

### 2.5 Input encoders

One per modality. Convert raw input (text characters, image pixels) into the shared latent space the central agent reads. Structural form is OPEN QUESTION Q1 (section 7).

**Common requirements:**
- Output dimension matches the central agent's input dimension
- Output distribution is aligned to the teacher's input-side hidden states (so the central agent reasons in a familiar latent geometry)
- Per-modality encoders can be trained independently and added later (e.g. audio after text and vision are working)

### 2.6 The teacher LLM (frozen)

Off to the side, the entire stack assumes ONE frozen teacher LLM that defines the canonical hidden-state distribution. Every JEPA-trained component is distilled against this teacher's trajectories.

**Candidate teacher: Qwen3-14B at NF4.** Already loaded successfully in Phase 0, Phase 1, and Phase 2 on Kaggle T4. Reusing it as the teacher costs zero new infrastructure. The decoder (in any of the Q2 options) is also derived from this teacher.

**Why one teacher and not a model committee.** Multiple teachers would give multiple incompatible latent distributions and would force the central agent to learn an "ensemble" alignment, which is a hard auxiliary problem. One teacher gives one canonical latent space and one decoder. The downside: the system inherits the teacher's biases and its style. That is acceptable for a research prototype.

---

## 3. Mechanisms in detail

### 3.1 Iterative latent reasoning (the JEPA predictor)

Both central agent and sub-agents share the same iterative latent reasoning mechanism. The predictor is a small transformer block. Per step:

```
z_{t+1} = predictor(z_t, context)
```

where `context` is the encoded input (for the first step), or the running residual stream (for subsequent steps). The predictor is trained against teacher trajectories: given z_t from the teacher's layer t, predict z_{t+1} from the teacher's layer t+1, with z_{t+1} detached.

K_central (number of central agent steps) and K_sub (per sub-agent per round) are HYPERPARAMETERS. Likely K_central = 3-8 and K_sub = 2-4. Set during the design exploration phase by sweep.

### 3.2 Teacher trajectory extraction (training data prep)

To train the predictors, we need (z_t, z_{t+1}) pairs. Source: run the frozen teacher LLM on chain-of-thought training data, capture hidden states at every transformer layer, treat (layer_t, layer_t+1) as a (z_t, z_{t+1}) pair.

For a teacher with L layers and a training set of N tokens, that gives N * (L-1) supervision pairs per training example. Storage: each hidden state is d-dim fp16. For Qwen3-14B (d=5120, L=40) and N=10M tokens, that's 10M * 40 * 5120 * 2 = 4 TB. Way too much.

**Mitigation: don't store, compute on the fly.** Run the teacher once per training batch, immediately use the trajectory for predictor training, discard. Costs one teacher forward per training step plus the predictor's own gradient step. Teacher forward is the expensive part.

**Alternative mitigation: subset of layers.** Use layers 0, 5, 10, 20, 30, 39 (six checkpoints) instead of all 40. Gives 6 supervision points per example. Predictor trains to step BETWEEN these checkpoints, not every layer. Reduces both compute and the "how many K steps" question to ~6 per agent.

The exact recipe is an open implementation choice but the principle holds: teacher trajectories are the training signal.

### 3.3 Shared public subspace (the alignment trick)

The shared public subspace is the communication channel. To enforce that all sub-agents emit comparable public-subspace vectors, training applies an alignment loss.

**Concrete proposal (one of several plausible).** During training, periodically sample a batch of "general purpose" data (mixed-domain, not specific to any sub-agent). Run that data through ALL sub-agents in parallel and extract their public-subspace projections. Apply an alignment loss: the public projections from sub-agent A and sub-agent B on the same input should be close.

```
L_align = mean_over_pairs (||public_A(x) - public_B(x)||^2)
```

This is added to the per-sub-agent training loss with a coefficient lambda_align. The private subspace is free; only public is regularized.

**Alternative: shared anchor projection.** Train one shared "public projection head" used by all sub-agents to project from their internal latent to the public subspace. The head is shared by hard constraint (same weights), not by soft alignment loss. Simpler but less flexible.

Both options are viable. Pick one in the design-exploration phase.

### 3.4 Private/public split implementation

Two viable patterns:

**A. Architectural split.** Sub-agent latent is `z = [z_private, z_public]` where z_private is the first d_priv dims and z_public is the last d_pub dims. Read/write to blackboard touches only the public portion.

**B. Learned projection.** Sub-agent latent is a single vector. A shared linear map P projects it to the public subspace: `public = P @ z`. Sub-agents read blackboard via the inverse-like operation: `z_updated = z + P.T @ (blackboard - P @ z)` or similar.

Option A is simpler. Option B is more flexible (the projection can learn to compress).

v1 implementation: option A.

### 3.5 Blackboard merge function

When K sub-agents write public-subspace updates in the same round, the blackboard needs to aggregate them into the next blackboard state. Options:

- **Sum**: simple, can saturate if many agents
- **Mean**: simple, dilutes strong signals
- **Attention-weighted**: learned. Central agent emits a "weight per agent" vector during dispatch, blackboard aggregates as weighted mean
- **Concatenate slots**: blackboard has a slot per agent, no merge per se; central reads ALL slots at synthesis

v1 implementation: attention-weighted mean, with weights from central agent's dispatch.

### 3.6 Continual learning protocol

Adding a new domain (e.g. "biology") to a deployed system:

1. **Train new sub-agent.** From scratch, distilled from the teacher on biology training data. Cost: comparable to training one of the existing sub-agents (substantial, but bounded). Output: a JEPA sub-agent model file plus a registered "domain prototype" (a latent vector summarizing what it specializes in).
2. **Train new N-LoRA adapter on central agent.** Small training run on a curated set of biology queries with annotated "this query needs biology sub-agent". Central agent learns to dispatch to the new sub-agent without forgetting its prior dispatch behavior (the orthogonality penalty handles this, as proven in Phase 1).
3. **Update the sub-agent registry.** A configuration file lists available sub-agents with their domain prototypes. Append the new entry.
4. **The alignment loss is RERUN periodically (or once, as a one-shot calibration) over the new sub-agent + all existing sub-agents.** The new sub-agent's public subspace must align with the existing ones. This is a small training pass; if all sub-agents were initially trained with the alignment loss against a shared anchor, the new sub-agent only needs to be aligned to that anchor, not pairwise to every existing agent.
5. Decoder unchanged. Other sub-agents unchanged.

Total cost per new domain: dominated by (1), the new sub-agent's full training. (2), (3), (4) are small.

---

## 4. Implementation phases (top-down plan)

Each phase has a goal, an acceptance gate, files to create or modify, key risks, and an estimated wall time on Kaggle T4. The plan is structured so each phase produces something runnable and gated before moving on.

### Phase A: Teacher trajectory capture + frozen decoder candidate

**Goal.** Stand up the frozen-teacher infrastructure. Pick a teacher (Qwen3-14B), capture hidden-state trajectories on a small chain-of-thought slice, build the decoder candidate (LM head + final norm, frozen), verify end to end: decoder reads teacher's final-layer hidden state, produces text identical to teacher's generation.

**Acceptance gate.** Round-trip identity: for 100 prompts, feeding the teacher's last-layer hidden state through the decoder produces the same text as the teacher's own generate(). Tolerance: 95%+ exact match.

**Files.**
- Create: `src/u_jepa/teacher/trajectory.py`. Hook the teacher's forward to capture per-layer hidden states for chosen layer checkpoints. Returns (input_ids, [hidden_layer_0, hidden_layer_K, ..., hidden_layer_L]).
- Create: `src/u_jepa/teacher/decoder.py`. Wraps the frozen teacher's final norm + LM head into a callable `decode(latent) -> text`. Provides the round-trip test.
- Create: `tests/test_teacher_roundtrip.py`. CPU stub test for the wrapper; the round-trip identity test runs on Kaggle.
- Create: `scripts/04_capture_teacher_trajectories.py`. Captures and caches trajectories for a fixed training slice.

**Risks.** Teacher's last-layer hidden state may not perfectly identity-decode if the LM head expects post-norm and the capture happens pre-norm. Mitigation: capture post-final-norm explicitly. Also: if Q2 settles on "top K transformer blocks + LM head", this phase's decoder is broader and the round-trip test needs adjustment.

**Wall time.** 4-6h Kaggle T4 (teacher load 10min, capture on 1000 examples 30min, run round-trip test 30min).

### Phase B: ONE JEPA sub-agent end-to-end (proof of mechanism)

**Goal.** Train ONE JEPA predictor on one domain (start with: general reasoning, using teacher trajectories on a math+code subset). Verify the predictor can iteratively step through the teacher's latent trajectory at inference time and produce a final latent that decodes to coherent text.

**Acceptance gate.** Two metrics.
- (i) Predictor reproduces the teacher's K-step trajectory within tolerance: cosine similarity between predictor's z_K and teacher's z_K, mean > 0.9 on a held-out 200-example set.
- (ii) Generated text quality: predictor's z_K decoded to text scores within 10pt of teacher's text on a small QA benchmark (e.g. GSM8K subset).

If (i) fails, the predictor is too small or the training data is wrong.
If (i) passes but (ii) fails, the cosine metric is misleading and a different loss is needed.

**Files.**
- Create: `src/u_jepa/agent/predictor.py`. The JEPA predictor module. Configurable depth (1-4 transformer blocks), takes (z_t, context), returns z_{t+1}.
- Create: `src/u_jepa/agent/iterative_reasoner.py`. Wraps the predictor in an iteration loop. Inputs: encoded x, K. Output: z_K.
- Create: `src/u_jepa/train/predictor_loop.py`. Training loop using teacher trajectories. Loss: cosine or MSE between predicted z_{t+1} and teacher z_{t+1}, target detached.
- Create: `tests/test_predictor.py`. CPU shape and gradient-flow tests on a tiny stub.
- Create: `scripts/05_train_general_predictor.py`. The actual Kaggle training run.

**Risks.** Predictor depth and training-set size are guessed. Likely needs a sweep. The decoder's round-trip from Phase A might not survive a predictor's z_K if z_K drifts from the teacher's z_K manifold. Mitigation: heavy weight on cosine loss early, with gradual relaxation.

**Wall time.** Per training run: 4-6h. Likely 2-3 runs for hyperparameter exploration. Total: 12-18h.

### Phase C: The central agent (a JEPA conductor + synthesizer)

**Goal.** Build the central agent: a JEPA iterative reasoner that ALSO has two learned heads. Dispatch head: from final latent, produces (sub-agent selection mask, initial blackboard state, N rounds). Synthesis head: takes the central's predictor + the final blackboard as input, runs more steps, produces one final latent.

For Phase C the central agent has only ONE sub-agent to dispatch to (the Phase B sub-agent). So dispatch is trivial. The point of Phase C is to validate the central agent's structure, not its routing intelligence.

**Acceptance gate.** End-to-end: input -> central dispatch -> single sub-agent rounds -> central synthesis -> decoder -> text. Generated text on GSM8K subset within 5pt of Phase B's stand-alone sub-agent. The central agent should not hurt performance compared to the sub-agent alone.

**Files.**
- Create: `src/u_jepa/agent/central_agent.py`. Wraps a JEPA predictor + the two heads.
- Create: `src/u_jepa/orchestration/blackboard.py`. The per-query blackboard data structure.
- Create: `src/u_jepa/orchestration/coordinator.py`. The function that runs: central.dispatch -> sub-agent rounds -> central.synthesize -> decoder.
- Modify: `src/u_jepa/agent/predictor.py` if needed to support the synthesis-mode (predictor runs with blackboard as additional context).
- Create: `tests/test_blackboard.py`, `tests/test_coordinator.py`.
- Create: `scripts/06_train_central_agent.py`. Trains the central agent's predictor and the two heads.

**Risks.** Training the central agent with only ONE sub-agent available is artificial. Mitigation: train the central's PREDICTOR on the same teacher-trajectory data as Phase B (so it learns the general reasoning skill); train the dispatch HEAD on synthetic single-sub-agent data (always dispatches to the only sub-agent, with random N rounds); train the synthesis head separately on blackboard-shaped inputs (synthetic blackboard built from teacher hidden states).

**Wall time.** 4-6h training + 2h eval. Per attempt.

### Phase D: Multiple sub-agents + actual blackboard rounds

**Goal.** Train 2-3 additional sub-agents in different domains (Math, Coding, Finance). Wire them into the coordinator. Validate that for a query needing multiple domains (e.g. "solve this math word problem about a stock portfolio"), the central agent dispatches to both Math AND Finance sub-agents, they communicate via blackboard for N rounds, and the result beats either sub-agent alone.

**Acceptance gate.** On a curated 100-example mixed-domain benchmark (math + finance, coding + math, etc.), the multi-agent system scores > max(individual sub-agents) by at least 5pt. If it doesn't beat the better single agent, the multi-agent protocol is not adding value and needs to be reconsidered.

**Files.**
- Create: `src/u_jepa/agent/sub_agent.py`. The sub-agent class wrapping a predictor + the private/public split.
- Create: `src/u_jepa/orchestration/registry.py`. A simple sub-agent registry (mapping domain -> sub-agent instance + prototype vector).
- Create: `src/u_jepa/losses/alignment.py`. The public-subspace alignment loss.
- Create: `src/u_jepa/train/coop_training.py`. Co-training loop: trains all sub-agents simultaneously with their domain-specific data PLUS the alignment loss on a general subset.
- Create: `scripts/07_train_multi_subagents.py`. The Kaggle training run for the 2-3 new sub-agents.
- Create: `scripts/08_eval_multi_domain.py`. The mixed-domain benchmark.

**Risks.** Three sub-agents trained simultaneously may not fit in 15 GB T4. Mitigation: train sub-agents SEQUENTIALLY with the alignment loss computed against a frozen "anchor" version of the previously-trained sub-agents. Trades training time for memory. Also: the mixed-domain benchmark may not exist off the shelf; we may need to construct it from existing benchmarks (e.g. mix MATH problems with a finance-context wrapper).

**Wall time.** Sub-agent training 4-6h each. Eval 2h. Total: 14-20h across multiple Kaggle sessions.

### Phase E: Continual learning (add a new sub-agent)

**Goal.** Demonstrate the "no retraining" promise. With Phase D's three sub-agents deployed, add a fourth (Law) by training only the Law sub-agent and a small N-LoRA adapter on the central agent. Verify: Law queries route correctly, prior-domain accuracy on a held-out set drops by less than 2pt (forgetting gate).

**Acceptance gate.** Law queries score reasonably on a Law benchmark slice. Math/Coding/Finance queries score within 2pt of their Phase D numbers (no forgetting).

**Files.**
- Modify: `src/u_jepa/agent/central_agent.py` to support N-LoRA adapter add on top of the dispatch head.
- Reuse: `src/u_jepa/continual/orthogonal_lora.py` from Phase 1.
- Create: `scripts/09_add_law_subagent.py`. Trains the new sub-agent + the central's new adapter.

**Risks.** The Phase 1 N-LoRA work was on transformer Linear modules. Applying it to the central agent's dispatch head (which is a few learned linears + the predictor) needs adapter placement decisions. Mitigation: target only the dispatch head's linears, leave the predictor untouched (the predictor's reasoning skill is general and should not need per-domain adaptation).

**Wall time.** 6-8h.

### Phase F: V-JEPA vision sub-agent

**Goal.** Add a vision-input pathway. Either as a new sub-agent (Image Understanding) or as a new input-encoder modality for existing sub-agents. This addresses Q3 (open).

**Acceptance gate.** TBD, depends on Q3 resolution. Likely: on a small VQA benchmark, the system correctly answers visual questions at quality within 5pt of a frozen-CLIP-projection baseline.

**Files.** Depends on Q3.

**Risks.** Vision is the highest-risk part of the original plan. The fallback to SigLIP-base-256 documented in the old plan still applies.

**Wall time.** 2-3 weeks of design + training. Hard to estimate precisely.

### Phase G: Full evaluation, ablations, writeup

**Goal.** Run the ablation matrix: with and without alignment loss, with and without multi-agent rounds (just pipeline), with and without continual learning. Quantify each component's contribution. Write the paper.

**Acceptance gate.** Submission to a NeurIPS workshop.

---

## 5. What carries over from existing Phase 0, 1, 2 work

Not everything in the repo is discarded. Mapping the existing code to v2:

| Existing code | Status in v2 |
|---|---|
| `src/u_jepa/models/qwen_base.py` (NF4 loader) | Reused. Qwen3-14B becomes the teacher and the decoder source. |
| `src/u_jepa/util/env.py`, `util/prompting.py` | Reused unchanged. |
| `src/u_jepa/continual/orthogonal_lora.py` (N-LoRA bank) | Reused for the central agent's continual-learning adapter. |
| `src/u_jepa/continual/n_lora_loss.py` | Reused. |
| `src/u_jepa/train/continual_loop.py`, `src/u_jepa/eval/continual.py` | Reused inside each sub-agent for within-sub-agent continual learning (Phase 1's exact mechanism, now nested inside a sub-agent rather than at the top level). |
| `src/u_jepa/data/trace.py`, `src/u_jepa/data/spider.py` | Reused as domain training data sources for the relevant sub-agents. |
| `src/u_jepa/losses/llm_jepa.py` (Phase 2 loss + TiedPredictor) | RECONSIDERED. The Phase 2 design used Q/A pairs as views (which we audited as collapse-prone). In v2, the JEPA target is teacher hidden-state trajectories. The TiedPredictor goes away; the JEPA predictor (section 3.1) takes its place. |
| `src/u_jepa/losses/sigreg.py` (LeJEPA SIGReg) | RECONSIDERED. SIGReg may still be useful as an anti-collapse regularizer applied to the predictor's output distribution across a batch (not per-token within one example, which was the Phase 2 mistake). Apply with care. |
| `src/u_jepa/train/jepa_aux_loop.py` (Phase 2 loop) | Mostly retired. Replaced by Phase B's `predictor_loop.py`. The `_pool_view_b` cache mechanism may inspire a teacher-trajectory cache, but the architecture is different enough that a fresh implementation is cleaner than a port. |
| `src/u_jepa/eval/spider_em.py` | Reused for the Coding/SQL sub-agent's evaluation (if SQL is one of the chosen domains). |
| Vendored `LatentMAS/` | Reused as architectural inspiration. The multi-agent latent-state passing in LatentMAS is structurally similar to our blackboard + per-round protocol. We may port specific modules (KV-cache management for sub-agent forward passes). |
| Vendored `lejepa/` | Reused if SIGReg makes a v2 comeback. |
| Vendored `llm-jepa/` | Reference only. The Q/A-pair-view framing did not transfer (see Phase 2 audit). Teacher-trajectory framing replaces it. |
| Vendored `N-LoRA/` | Reference only. Our `orthogonal_lora.py` is our implementation. |
| `kaggle/phase0/`, `kaggle/phase1/`, `kaggle/phase2/` | Phase 0 and 1 results stand. Phase 2 result is preserved as a negative-result ablation. |

What's NEW in v2 (no existing code):
- The central agent (`src/u_jepa/agent/central_agent.py`)
- The JEPA predictor as a reasoning engine (`src/u_jepa/agent/predictor.py`)
- The blackboard (`src/u_jepa/orchestration/blackboard.py`)
- The coordinator (`src/u_jepa/orchestration/coordinator.py`)
- Teacher trajectory capture (`src/u_jepa/teacher/trajectory.py`)
- The frozen decoder wrapper (`src/u_jepa/teacher/decoder.py`)
- The alignment loss (`src/u_jepa/losses/alignment.py`)
- The co-training loop (`src/u_jepa/train/coop_training.py`)

---

## 6. Hardware and budget

Project hardware available:
- Local: RTX 4060 laptop (8 GB VRAM). Used for unit tests, small-scale design exploration, paper writing.
- Cloud: Kaggle free-tier T4 (15 GB VRAM, 9h session cap, 30h/week quota). Used for all heavy training.

Per Phase 0/1/2 experience, the Kaggle pipeline is reliable and Qwen3-14B at NF4 comfortably fits with room for adapters and predictor training. v2 sticks with the same hardware envelope.

Estimated total compute for v2 phases A through G: roughly 80-150 hours of Kaggle T4 time, plus 4060 for everything else. Well within the weekly quota over a 14-week timeline.

The earlier framing that suggested "RTX 4060 only" or "laptop+free-tier" was misleading. Heavy training has always been planned for Kaggle. The laptop is the development machine, not the training machine. The audit context doc will be updated to reflect this.

---

## 7. Open questions (to resolve before final implementation plan)

These three questions were explicitly deferred during the brainstorm. Each one shapes a real component. The implementation plan above proceeds in a way that's compatible with most reasonable answers, but resolving these locks in the details.

### Q1. Input encoder pathway

How does a raw input (text, image, eventually audio) become a latent that the central agent reads?

Options that were on the table:
- **Q1.a Reuse teacher LLM for text, V-JEPA for vision, project each to common space.** Text input goes through Qwen3's first few transformer blocks (the teacher's "input side", frozen). Vision goes through V-JEPA. Each then passes through a small per-modality projection. Minimum new components, maximum compatibility with the frozen decoder. Locks us to the teacher's tokenizer for text.
- **Q1.b Train a unified multi-modal encoder from scratch.** Cleaner architecturally, much harder training. Probably out of scope.
- **Q1.c Per-modality JEPA encoders, no shared backbone.** Text JEPA, V-JEPA for vision, alignment heads to common space. More modular.

Recommendation when resuming: pick Q1.a for v1. It minimizes new training and reuses the teacher we already trust. Revisit if the constraint becomes painful.

### Q2. Decoder structure

What exactly is the frozen text decoder?

Options:
- **Q2.a Just the LM head + final norm of the teacher.** Cheapest. Sub-agents must produce latents in the teacher's penultimate hidden-state distribution. Identity round-trip is a hard but achievable training target.
- **Q2.b Top K transformer blocks + LM head of the teacher.** More forgiving. The transformer blocks can "clean up" slightly off-distribution latents. Costs more per inference. Recommended if Q2.a's round-trip is fragile.
- **Q2.c Small purpose-built decoder trained once.** Most flexible latent space. Needs a bootstrap (we have no JEPA latents until at least one sub-agent works). Probably wrong for v1.

Recommendation when resuming: start with Q2.a in Phase A. If the round-trip identity test fails at 95%+ , fall back to Q2.b with K=2-4.

### Q3. Vision and image generation

The user named "Image Gen" as one of the sub-agent examples. The current architecture has ONE frozen text decoder. Image generation needs a different output modality.

Options:
- **Q3.a Multi-decoder architecture.** Add a frozen image decoder (e.g. a vendored diffusion model) alongside the text decoder. Central agent decides which decoder to invoke at synthesis time. Major architectural addition. More complex training.
- **Q3.b Text-described image generation.** Image-Gen sub-agent outputs a text description that a separate (downstream, not part of the U-JEPA system) image renderer turns into pixels. The U-JEPA architecture stays text-output-only. Simpler, but "Image Gen sub-agent" becomes "Image Prompt sub-agent" which is a weaker claim.
- **Q3.c Vision input only, no image generation in v1.** A Vision sub-agent for understanding images is included, but image generation is out of scope until v2 of the project. Simplest. Most defensible scope decision for a NeurIPS workshop paper.

Recommendation when resuming: pick Q3.c for v1 (vision-input via a Vision sub-agent, no image generation in the architecture). Q3.a is interesting research but doubles the architectural surface area. Q3.b is fine but weak.

---

## 8. Risks and mitigations

### 8.1 The single biggest risk: teacher distillation may not transfer to multi-step iteration

The JEPA predictor is trained to step from teacher's layer t to layer t+1. At inference, the predictor runs autonomously for K steps. If the predictor's errors compound, by step K the latent has drifted off the teacher manifold and the decoder produces garbage.

This is the "compounding error" problem that plagues all autoregressive latent predictors. Mitigations:
- Use a smaller K (3-5) to limit compounding
- Train the predictor on TRUE multi-step trajectories (chain the teacher across multiple forward passes on multi-turn or chain-of-thought data, not just single-forward layer-to-layer)
- Add an EMA target encoder (I-JEPA-style) so the target latent is a smoothed version of the teacher's, which tolerates small predictor drift

If after Phase B these mitigations don't get cosine > 0.9 at step K, the architecture has a fundamental problem and we should pivot to Option B from the original brainstorm (deep JEPA stack, single forward pass).

### 8.2 The public-subspace alignment loss may not actually align

Pushing sub-agents to produce similar public-subspace projections on general data is a soft constraint. If domain-specific training dominates, alignment can degrade as sub-agents converge to specialized representations.

Mitigations:
- Periodic alignment re-training as a regular maintenance step
- Co-train all sub-agents from the start with the alignment loss always on
- Larger alignment loss weight

If alignment fails, the "sub-agents talk to each other" claim is meaningless and the system degrades to a router + independent specialists (still useful but less novel).

### 8.3 The blackboard protocol may not converge in finite rounds

If sub-agents' contributions oscillate or fail to integrate, the blackboard never stabilizes and the synthesis produces noise.

Mitigations:
- Fixed N rounds (no adaptive stopping in v1)
- Decay the contribution weights over rounds so later rounds are smaller updates
- Inspect blackboard state across rounds during Phase D eval to verify convergence

### 8.4 Continual learning may forget despite N-LoRA

Phase 1 showed zero forgetting on TWO tasks. v2 will eventually push to 5-10 sub-agents and routing decisions. The orthogonality constraint scales as O(num_prior_tasks); at 10 prior tasks, the per-step penalty has 10 terms.

Mitigations:
- Monitor forgetting as sub-agent count grows; report it in the ablation
- If forgetting > 5% at any point, switch to gated experts (router prevents activation of old adapters at inference) instead of soft orthogonality

### 8.5 Kaggle 9h cap and 30h/week quota

Several phases need more than 9h end to end (Phase D's three-sub-agent training, in particular). Mitigations:
- Sequential sub-agent training across sessions, checkpointing between
- Reduce sub-agent training data per session
- Use weekend bursts to maximize the 30h/week budget

### 8.6 Single seed across all phases

Phase 1 and Phase 2 used single seeds. Variance estimates are missing from all reported numbers. v2 should report mean +/- 1 SD across at least 2 seeds for the final ablation matrix. Budget constraint: this doubles eval time.

---

## 9. What success looks like

End of Phase G, the deliverable is:

- A working prototype of the central JEPA brain + 4-5 sub-agents + frozen decoder
- A continual-learning demonstration: add a new sub-agent without retraining the rest
- A multi-modal demonstration: vision input through a Vision sub-agent
- An ablation matrix quantifying each component's contribution
- A workshop paper (NeurIPS or similar) describing the architecture and results

If any phase B-D fails its gate, the project pivots:
- Phase B fail: drop iterative latent reasoning, fall back to deep JEPA stack (single forward)
- Phase D fail: drop multi-agent rounds, fall back to sequential pipeline of sub-agents
- Phase E fail: drop continual-learning claim, retrain central agent when new sub-agent added (weaker claim, still publishable)

If Phase A fails, the project pivots all the way back: drop the JEPA-based reasoning and revert to the original Phase 4 plan (LoRA + router on a frozen LLM). Phase A failure would mean the decoder round-trip is fundamentally not viable with the teacher we picked, which would be a strong signal to rethink.

---

## 10. Decision points for the user before implementation starts

Before kicking off Phase A, the user should resolve:

1. **Q1, Q2, Q3 in section 7.** Pick one option per question OR confirm the recommended defaults (Q1.a, Q2.a with Q2.b fallback, Q3.c).
2. **Sub-agent domain list for Phase D.** Need 3 domains beyond the Phase B "general" sub-agent. Suggested: Math, Coding, Finance. Confirm or adjust.
3. **Teacher choice.** Default is Qwen3-14B (reuses Phase 0/1/2 infrastructure). Confirm or pick a different teacher.
4. **Whether to keep Phase 2's negative result in the writeup or scrub it.** Recommendation: keep it as an honest ablation showing what the wrong framing of paired-views looks like.

Once these are decided, this design doc becomes the implementation plan and Phase A can start.

---

End of design + implementation plan. Get back to this when you're ready and we'll resolve the open questions, then start Phase A.
