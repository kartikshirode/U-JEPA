# U-JEPA v3 design: admission control for knowledge editing

Date: 2026-08-11
Status: draft for external review
Author: Kartik Shirode
Repo: github.com/kartikshirode/U-JEPA
Supersedes: `v2/docs/pipeline-1-build-plan.md`
Target: ICML 2027 main track, with a NeurIPS 2026 workshop checkpoint in September

This document is written to be read cold. A reviewer with no history on the project should be able to judge it without opening anything else. Section 12 lists what I most want pushed back on.

---

## 1. Why v3 exists

v1 and v2 were both shaped by not having compute. The project opened in May 2026 with a plan to publish an architecture paper containing no experiments, because the only hardware was an 8 GB laptop. v1 then forked LatentMAS onto a Kaggle T4 and failed its Phase 2 gates. v2 was designed around the same T4: 9 hour sessions, 15 GB, a 1.5B model ceiling, single seeds and 200-row evaluations.

Two things changed.

Hardware. 4 NVIDIA H200 GPUs, 141 GB each. Topology is not yet confirmed, so this design assumes **no single job may exceed 141 GB**, and flags separately what opens up if the cards turn out to share NVLink.

Literature. v2's headline claim died in May. More on that below.

## 2. What the literature did to the old plan

v2's paper was going to argue that a probe-gated intake loop survives long edit sequences where ungated editing collapses. Sequential-edit collapse was a documented open problem and the gate was the fix.

UltraEdit (TMLR 2026, arXiv 2505.14679) scales to 1 million sequential edits with model stability preserved, and at 20K edits shows almost no deviation from the unedited baseline on SST, MMLU, MRPC and NLI. It is training-free and memory-free, edits a 7B model on a 24 GB consumer card, and was evaluated up to Gemma-3-27B. "More Edits, More Stable" (arXiv 2605.11836) then shows UltraEdit, RLEdit and StableEdit all independently converged on the same normalization trick.

So ungated editing already survives. The claim is false as stated and building it would have wasted the quarter.

Read that result as a threat model instead and it inverts. UltraEdit performs **no verification or gating of any kind**. Stability was the last accidental barrier between a malicious edit and permanence, and it is gone.

Three papers say the resulting hole is real and open:

- **EditRisk-Bench** (arXiv 2605.10146) finds in-context editing achieves near-perfect attack success while general benchmarks stay clean. Attacks are largely stealthy. Detection does not generalize across editing strategies, parameter edits are hard to revert, and the authors conclude existing mitigation remains limited.
- **Position: Editing LLMs Poses Serious Safety Risks** (arXiv 2502.02958) calls for safety protocols inside editing workflows. A call, not a solution.
- **SSGM** (arXiv 2603.11768) proposes a governed-memory architecture with a write gate and rollback, and runs zero experiments. It is explicitly a set of design principles with three testable hypotheses left as future work.

SSGM matters most. Somebody has planted a flag on the concept. They have a theorem and no numbers. The empirical instantiation is unclaimed, and in security work the evidence is worth more than the diagram. This design cites them as the architecture we are testing, not as prior art we are dodging.

One more result shapes the mechanism. **Do LLMs Really Know What They Don't Know?** (ACL Findings 2026) shows hidden states mainly encode knowledge *recall* rather than truthfulness, and that hallucinated and factual outputs have overlapping hidden-state geometry. Unknown unknowns, by contrast, cluster distinctively and detect reliably.

That vindicates the one design instinct v2 got right for weak reasons: the latent model judges coherence and novelty, never truth. Truth belongs to provenance and verification. v3 keeps that separation and now has a citation for it.

## 3. What Q1 established

The architecture originally assumed knowledge sorts into two layers, one that never changes and one that does. That assumption was worth checking before it became a system, so I checked it first.

Method and full results are in `v3/spikes/q1_volatility/FINDINGS.md`. Summary:

Data was WikiBigEdit, 8 Wikidata snapshot diffs covering 2024-02-01 to 2024-07-01. After dropping 11,038 rows with a null subject or relation, 491,344 rows over 941 relations.

**Volatility is real and large.** CNC film rating sits at 0.000 churn over 2,288 rows. Site of astronomical discovery sits at 0.922 over 17,363.

**It is predictable.** Churn measured on timesteps 0-3 correlates with churn on timesteps 4-7 at Spearman 0.695, p = 2.2e-41. Churn is a trait of the relation, so layers can be assigned without a human ruling case by case. This was the result that could have killed the design and it came back positive.

**It is not curation noise.** Median concentration of a relation's updates in any single timestep is 0.286. Only 5% of relations are lumpy at 0.8 or above, accounting for 1,407 updates against 78,129 in the evenly spread ones.

**But it is a continuum, not a binary.** Across the 206 relations with at least 200 rows, the churn histogram by decile runs 139, 39, 10, 8, 3, 0, 4, 0, 2, 1. 67% below 0.1, 0.5% above 0.9. One hump with a long tail. There is no valley to cut at.

**And the biggest finding was not the one I asked for.** 78% of rows are tagged `new`, only 20% `update`. Four fifths of knowledge change is adding facts that were never held. Adding cannot contradict anything, so it needs provenance and nothing else. Every overwrite of an existing belief lives in the remaining 20%.

Three consequences, all of which are folded into the design below:

1. Layer assignment becomes a threshold on a continuum with a reported error rate, not an assumed dichotomy. That converts an assumption a reviewer would argue with into a measurable result.
2. The primary routing decision is accretion versus revision, not invariant versus volatile. The expensive path runs on a fifth of the traffic.
3. Wikidata revisions conflate the world changing with the database being tidied. The design has to separate them and the evaluation has to report them apart.

## 4. Thesis

> Lifelong editors now absorb a million edits without degrading. That removes the only accidental defense LLM knowledge had. We build and evaluate the missing admission layer: a gate deciding what enters, routed by whether the claim adds or overwrites, thresholded by how volatile the target knowledge is, scored by provenance, verification and a belief-state energy over the model's own latents, and measured under adaptive attack.

## 5. Architecture

```
candidate (claim, source, evidence)
        |
        v
  [ ROUTE ]  does this add, or overwrite something held?
        |
        +--- accretion (~78%) ---> provenance only ---> admit ---> editor
        |
        +--- revision (~20%)
                 |
                 v
           [ VOLATILITY SCORE ]  how much does this relation churn?
                 |                (sets the bar, does not pick a store)
                 v
           [ SIGNALS ]  provenance | retrieve + NLI | belief-state energy
                 |
                 v
           [ COMBINER ] ---> admit | refuse | quarantine
                 |
              (admit)
                 v
           [ EDITOR ]  UltraEdit | AlphaEdit | RLEdit
                 |
     ............ v ............................
     : LAYER 2, sampled and async               :
     : shadow copy -> probe -> promote/rollback :
     ...........................................
```

Five decisions worth defending.

**The layers survive as a policy gradient, not as separate stores.** The original vision had two physical knowledge layers. Q1 says volatility is continuous, so instead every fact carries a volatility score and that score sets the admission threshold and the retraction policy. Low volatility means a high bar, and a contradiction is read as the incoming claim being wrong. High volatility means a low bar, and a contradiction is read as the world having moved. This is the faithful translation of the two-layer intuition into what the data supports.

**The gate is pre-commit and runs on the unedited model.** It never asks whether an edit hurt; it asks whether a claim conflicts with what is already held. That keeps it O(1) per candidate and batchable, which is the only way a gate survives the million-edit regime that makes the threat urgent. v2's design applied the edit and then measured, costing a full edit per candidate, which does not scale to the regime that motivates it.

**The gate is editor-agnostic and never touches editor internals.** Editor state of the art moved twice in six months. An editor becomes a row in a table rather than a rewrite.

**The decision is three-way.** Quarantine exists because the honest answer for a genuinely novel true fact is to hold it pending evidence. Without that class the false-refusal number is a lie.

**Layer 2 is sampled, not per-edit.** A shadow copy of the core, applied speculatively, probed, then promoted or discarded. That was unaffordable on a T4 and is cheap on an H200. It catches what the pre-commit gate misses and it directly tests SSGM's rollback hypothesis, which they state and never run.

### 5.1 The belief-state energy

This is the JEPA component and it is deliberately staged last, because RQ4 is the only question whose failure does not sink the paper.

Train a predictor over the **frozen** core's latent trajectory: given hidden states up to step t, predict the next latent. The objective follows NextLat (arXiv 2511.05963), which shows such latents provably converge toward belief states, meaning compressed history sufficient to predict the future. The core is never touched, so the frozen-core premise holds and the cost is one small predictor rather than a pretraining run.

For a candidate fact f: build a probe context from f stated declaratively plus its retrieved knowledge neighborhood, run the frozen core, and take the predictor's error normalized against a matched control context. High energy means the claim does not fit the trajectory the core expects.

The claim about this signal is narrow on purpose. Per the ACL Findings result above, we never assert that energy measures truth. We assert it measures conflict with held knowledge, which is the recall axis, which is the axis that separates empirically. Truth stays with provenance and the verifier.

### 5.2 The combiner

Logistic regression over the signal vector, with volatility score as a feature and as a threshold modifier. Escalate to gradient boosting only on demonstrated underfit. Six interpretable parameters beating a hand-tuned threshold is a stronger result than a black box doing the same, and it is far easier to defend.

The obvious trap: train the combiner on our own injected attacks and it learns our attack distribution rather than harm. The mitigation is structural. Train on one attack family, test on held-out families, and report cross-family generalization as a headline number rather than an appendix.

## 6. Research questions

**RQ1, the threat.** Do modern stable editors admit adversarial knowledge as readily as benign knowledge, and does their stability make the resulting corruption more persistent than it was under ROME or MEMIT?

**RQ2, the gate.** Can pre-commit signals separate harmful from benign edits, at what operating point, at what latency, and at what cost in legitimate knowledge refused?

**RQ3, the stratification.** Does conditioning the admission threshold on a relation's volatility beat a single global threshold? What is the error rate of volatility-based layer assignment itself?

**RQ4, the JEPA question.** Does belief-state energy carry information that surface fact-verification does not, specifically on stealth edits that pass individual verification while conflicting with held knowledge?

**RQ5, the adversary.** Does the gate survive an attacker with white-box knowledge of it, optimizing edits to minimize belief-state energy while preserving the payload?

RQ3 is the contribution that traces back to the project's founding idea. RQ4 is the one that decides whether the name still fits.

## 7. Expected results and failure branches

| RQ | Expected | If it fails |
|---|---|---|
| 1 | Attack success comparable to benign success, above 90%. Corruption persisting longer under stable editors. General benchmarks flat, confirming stealth | If stable editors resist attacks the threat model weakens; pivot to the coherence question alone |
| 2 | Combined gate AUROC 0.80 to 0.90. Provenance strong on sourced attacks, weak on plausible-but-false. Sub-second per edit. False refusal 5 to 15% | Below 0.7 the gate is the bottleneck; report the negative and characterize what is undetectable |
| 3 | Volatility conditioning adds 3 to 8 AUROC points over a global threshold, concentrated on low-volatility relations where the bar should be high | No lift means volatility is not decision-relevant even though it is predictable, which is itself worth reporting |
| 4 | Energy adds 5 to 15 AUROC points over verification alone on the stealth subset, and near zero on the crude subset | Zero lift cuts the JEPA half honestly as an ablation. RQ1, 2, 3 and 5 still make a complete paper |
| 5 | Meaningful but not catastrophic degradation, AUROC down 0.10 to 0.20 white-box. Provenance holds where energy bends | Full collapse is a strong publishable negative |

Headline figure: detection rate against false-refusal rate, curves for provenance only, plus verification, plus volatility conditioning, plus belief-state energy, on the stealth subset, with the adaptive-attack curve dashed underneath.

## 8. Experimental design

| | |
|---|---|
| Scales | 8B primary (Llama-3-8B or Qwen2.5-7B). 27B second scale (Gemma-3-27B). Both appear in UltraEdit's own table so numbers sit beside theirs without translation. 70B needs sharding and is NVLink-dependent, so it is flagged as a stretch and not planned |
| Editors | UltraEdit, AlphaEdit, RLEdit, gate held fixed across all three |
| Arms | untouched base / editor ungated / provenance only / plus verifier / plus volatility conditioning / plus belief-state energy |
| Seeds | 5, mean and standard deviation reported, no exceptions |
| Eval n | Power-calculated per gate before the run |
| Edit counts | 1K, 10K, 100K, so the result is a curve and not a point |
| Parallelism | 4 independent GPUs over seeds by editors by attack families. No collectives, so it survives whatever the topology turns out to be |

Benign edits come from WikiBigEdit for real-world scale, with zsRE and CounterFact for comparability against the editing literature. Adversarial edits come from EditRisk-Bench (7,893 instances across misinformation, bias and safety violations, 1-hop to 3-hop) plus generated families held out for the cross-family test. General-ability probes match UltraEdit's set exactly: SST, MMLU, MRPC, NLI.

Volatility labels come from the Q1 pipeline, extended to multi-year Wikidata snapshots to fix the recurrence power problem, and with curation-driven relations separated from world-change relations before they enter any threshold.

## 9. Staging, with kill switches

| Stage | Delivers | Kill switch |
|---|---|---|
| 0 | Harness: editors behind one interface, datasets loaded, shadow-copy plumbing, run sharder | Not a gate. Infrastructure is not a phase and claims no result |
| 1 | RQ1, 3 editors, 2 scales, 5 seeds | Editors resisting attacks weakens the threat model; pivot |
| 2 | RQ2 and RQ3, gate with provenance, verifier and volatility conditioning, full ROC | AUROC below 0.7 means report the negative |
| 3 | **Workshop submission, late September** | Hard date. Whatever is finished, ships |
| 4 | RQ4, belief-state predictor trained, energy added | No lift on the stealth subset cuts the JEPA half |
| 5 | RQ5, adaptive white-box attacker | Full collapse is a publishable negative |
| 6 | Ablations, seeds, writeup, ICML 2027 | |

Stage 3 is a deliverable inside the plan rather than a hope. Stages 1 and 2 constitute a complete paper on their own, which is the entire reason the gate ships before the JEPA half.

**Scope of the first implementation plan: stages 0 through 2 only**, ending at the workshop submission. Stages 4 and 5 each get their own spec and plan once the stage 2 results are in, because both are contingent on what stage 2 finds. Planning all 7 stages now would be planning against numbers that do not exist yet.

## 10. Non-goals

Named explicitly because scope creep is this project's documented failure mode.

- No vision, no V-JEPA. It was v2's highest-risk unstarted phase and was never load-bearing.
- No multi-agent, no LatentMAS, no latent chain-of-thought. That is v1 and it stays frozen.
- No domain sub-agent routing. Mixture-of-experts already routes to specialists. It becomes interesting once a substrate exists to route between, and not before.
- No new editor. We consume the editor family; competing with a field shipping a new state of the art every quarter is a losing position.
- No core retraining. Only the predictor and the combiner ever see a gradient.
- No orthogonal-LoRA consolidation. Interesting, and it goes in future work.

## 11. Rules carried from the v1 audit

Every one of these traces to a specific documented failure in `docs/external_audit_context.md`.

1. Latent objectives only where a real view relation exists. Question-to-answer pairs are not views; that error collapsed v1's Phase 2 target by construction.
2. Statistical regularizers act across samples. If the batch is too small for that, the batch is the bug.
3. Every metric gets a ceiling check and a sensitivity check before it is allowed to be a gate.
4. Power calculation before the run. v1 gated a 2 point effect on n=200, roughly 10 times underpowered.
5. Five seeds minimum. The compute excuse is gone.
6. Every comparison carries an untouched-model arm.
7. Infrastructure is not a phase and gates only on scientific results.
8. One mechanism at a time, each with its own kill switch. v1 composed four separately-validated ideas at once and they did not compose.
9. The latent predictor is load-bearing or the name changes.

## 12. What I want the reviewer to attack

Ordered by how much a wrong answer would cost.

1. **Is the threat model real, or is it a paper threat?** The argument is that stable editors plus zero verification plus stealthy attacks equals an open hole. The counter is that nobody deploys open knowledge editing on untrusted input, so the attack has no realistic delivery path. If that counter holds, RQ1 is uninteresting and the project should be reframed around benign knowledge maintenance instead of adversarial admission.
2. **Is the SSGM position defensible?** We arrive second on the concept and first with evidence. Is that a contribution reviewers reward, or does it read as implementing somebody else's design?
3. **Does the belief-state energy actually have a mechanism, or is it hope?** The ACL Findings result says hidden states track recall rather than truth. We are betting the recall axis is exactly what is needed. If that reasoning is wrong, RQ4 should be cut now rather than at stage 4.
4. **Is volatility conditioning worth a research question, or is it a feature?** Q1 shows volatility is predictable. It does not show it is decision-relevant. RQ3 might be a paragraph rather than a contribution.
5. **Is 5 stages before the September workshop realistic?** Stages 0 through 2 in roughly 6 weeks, on hardware not yet configured, with datasets not yet ingested.
6. **Is the accretion/revision split real, or a WikiBigEdit artifact?** 78/20 came from one dataset built for a different purpose. If accretion is over-represented by construction, the cheap-path argument weakens considerably.
7. **Should this still be called U-JEPA?** If RQ4 comes back flat, the honest paper is a systems and security contribution with a negative JEPA ablation. Renaming then is cheaper than defending the name.

## 13. Open items

- Hardware topology is unconfirmed. This design assumes no single job exceeds 141 GB. NVLink would unlock 70B editing experiments and a larger belief-state predictor.
- Multi-year Wikidata snapshots need sourcing to fix Q1's recurrence power problem.
- Wikidata property statement counts need a query-service pass to give true churn rates rather than raw edit shares.
- No decision yet on whether curation-driven relations are excluded from the volatility threshold or modelled as a separate class.
