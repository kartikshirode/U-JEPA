# U-JEPA v3 design: admission control for automated knowledge maintenance

Date: 2026-09-05
Status: draft, second revision, for external review
Supersedes: `docs/superpowers/specs/2026-08-11-u-jepa-v3-design.md`
Target: ICML 2027 main track, with an arXiv preprint once stage 2 lands
Author: Kartik Shirode

Written to be read cold. Section 13 lists what I want attacked.

---

## 1. What changed since the last draft, and why

The 2026-08-11 draft went out for external review. The review was correct on every finding and two of them were fatal to the science rather than the code.

**The old RQ1 was a tautology.** It asked whether stable editors admit adversarial knowledge as readily as benign knowledge. An editor is built to fit whatever target you hand it without judging its truth, so a near-zero gap between benign and adversarial efficacy is guaranteed by construction. An editor that refused the adversarial target would be a broken editor. I had written an expected-results table predicting "attack success above 90%" as though that were a finding.

**The threat model had no delivery path.** I cited EditRisk-Bench, BadEdit, a safety position paper and SSGM as though they established one attack surface. They don't. EditRisk-Bench assumes malicious edits have already been supplied. BadEdit assumes the attacker holds editing access. The position paper is about malicious model distribution, where an attacker would simply switch the gate off. SSGM governs an agent's external memory. None of them shows an attacker-controlled input reaching a trusted parametric editing service.

Three more corrections carried into this draft:

**Q1's central metric measured the wrong thing.** `n_updates / n_rows` is the composition of observed change, not the probability a fact changes, because the denominator holds only rows that already changed. It's renamed `update_share` and the verdict is downgraded. Details in `v3/spikes/q1_volatility/FINDINGS.md`.

**The 78/20 accretion split was load-bearing and shouldn't have been.** It's a WikiBigEdit statistic, not a property of knowledge, and the reasoning built on it was worse than the statistic. "Adding cannot contradict" is false: a newly added criminal conviction or medical condition is harmful without colliding with any existing slot. The provenance-only bypass is gone.

**The NextLat theorem doesn't transfer.** NextLat gets its belief-state convergence result under joint next-token and transition-consistency training of the representation. Training a small predictor over an otherwise frozen model doesn't make the frozen hidden states satisfy that theorem. The belief-state energy stays as an empirical hypothesis with no theorem behind it.

Full disposition table in `docs/reviews/2026-09-05-codex-review-v3.md`.

## 2. The deployment

Everything below hangs on one concrete system, named up front, because deriving a threat model from papers instead of from a deployment is how the last two drafts went wrong.

**Automated knowledge maintenance.** An operator keeps a deployed model factually current by ingesting a public knowledge feed and applying the resulting facts as parametric edits. Volume is high enough that per-edit human review is impractical, which is the whole reason lifelong editing exists as a field.

The trust boundary:

| | |
|---|---|
| Attacker can | alter a small fraction of entries in the upstream feed |
| Attacker cannot | touch the gate, the editor, the model weights, or the evaluation |
| Operator controls | the editor, the gate, the probes, the rollback path |

This is not hypothetical. Wikidata is world-editable by design and vandalism there is a studied phenomenon with its own detection literature and labelled corpora, going back to the Wikidata Vandalism Corpus and the WSDM Cup 2017 detection task. That gives a real attacker, a real delivery path, a measurable base rate, and human-labelled ground truth for what a malicious entry looks like.

It also unifies the failure causes, which matters for scope. Vandalism, a compromised upstream source, a stale fact and an honest data error are different origins of the same event: an unsafe write. The gate doesn't need to tell them apart.

## 3. What makes the threat non-obvious

A reasonable objection to the whole project: public feeds already self-correct. Wikidata vandalism gets reverted, usually fast. The operator's next maintenance batch carries the revert. So the model heals itself and there is nothing to defend.

That objection is what makes RQ1 worth running, because the editing literature says it probably fails.

"Exposing the Illusion of Erasure in Knowledge Editing" (arXiv 2606.23276) finds that edits suppress rather than remove, and that low-rank updates redistribute existing knowledge instead of overwriting it. Context-guided elicitation recovers supposedly-erased facts at over 85% in white-box settings, blind reconstruction at 15 to 48.5%, and models fall back to pre-trained knowledge in roughly 47% of implicit reasoning tasks.

If that holds, an upstream revert does not clear the poison. The source is corrected, the operator applies the correction in good faith, and the model still holds a recoverable false fact. The pipeline looks healthy and isn't.

Two things make this a genuine open question rather than a restatement. That paper tested ROME, MEMIT, MEND and FT-L, none of the modern stable editors this design targets. And the interaction with stability runs the wrong way from intuition: an editor that degrades under accumulation destroys its own poison over time, while an editor that preserves every edit perfectly preserves the poison too. UltraEdit's headline result becomes a liability under this reading, which is the kind of claim worth measuring.

## 4. What the literature establishes

**Stability is solved and unguarded.** UltraEdit (TMLR 2026, arXiv 2505.14679) scales to 1 million sequential edits with stability preserved, and at 20K edits shows almost no deviation from the unedited baseline on SST, MMLU, MRPC and NLI. It was evaluated on GPT-J 6B, Mistral-7B-v0.3, Llama-3-8B-Instruct, Qwen2.5-7B-Instruct, Phi-4-14B and Gemma-3-27B. It performs no verification of any kind. "More Edits, More Stable" (arXiv 2605.11836) shows UltraEdit, RLEdit and StableEdit independently converged on the same normalization mechanism.

**Harm from injected knowledge is established.** EditRisk-Bench (arXiv 2605.10146) covers 7,893 instances across misinformation, bias and safety violations at 1 to 3 hops. Attacks stay stealthy against general benchmarks, detection doesn't generalize across editing strategies, and the authors conclude mitigation remains limited. This is where the downstream-harm measurement comes from; it is not the delivery path.

**Retraction is unreliable.** Section 3 above.

**Governance has been proposed and never tested.** SSGM (arXiv 2603.11768) sets out a write gate, a read gate and reconciliation for agent memory, with a drift bound and three testable hypotheses, and runs zero experiments. It's convergent architectural motivation for this work, not the architecture being implemented, and its hypotheses concern external memory rather than parametric editing. The novelty claim here is narrower than "first with evidence", which would be false given EditRisk-Bench and BadEdit already show malicious editing works. See section 6.

**Latents track recall, not truth.** "Do LLMs Really Know What They Don't Know?" (ACL Findings 2026) shows hidden states mainly encode whether the model is recalling parametric knowledge, with hallucinated and factual outputs sharing overlapping geometry, while unknown unknowns cluster distinctively and detect reliably. That's why the gate's latent signal is scoped to conflict-with-held-knowledge and never to truth.

## 5. What Q1 contributes, after correction

The spike ran 8 Wikidata snapshot diffs from WikiBigEdit, 491,344 rows over 941 relations covering 2024-02-01 to 2024-07-01.

What survives: relations differ enormously in the composition of their change, and that composition is stable enough to predict from history, split-half Spearman 0.695 at p = 2.2e-41 across 278 relations. The distribution is one hump with a long tail, two thirds below 0.1 and half a percent above 0.9, so any layer split is a threshold rather than a natural boundary.

What doesn't: this is not volatility, and nothing here licenses a threshold on how carefully to check an incoming fact. Getting the real number needs revisions divided by statements at risk, which needs Wikidata property statement counts from the query service. That's a prerequisite task, not a footnote.

So relation-level features enter the design as candidate gate features whose value is an empirical question, not as an architectural layer. That is a smaller claim than the last draft made and it's the one the data supports.

## 6. Thesis

> An operator who keeps a model current from a public feed inherits that feed's attackers. Modern editors will absorb whatever the feed carries, a million times over, without verification and without degrading. And because editing suppresses rather than erases, the feed's own correction does not reliably undo a poisoned write. We measure that failure in a realistic pipeline, then build and evaluate the admission gate it implies, at the base rate real vandalism actually occurs.

The novelty claim, stated narrowly enough to defend: **pre-commit admission control for long-horizon parametric editing, evaluated under matched benign and malicious writes at a realistic base rate, against multiple stable editors, with an adaptive attacker.** Whether that survives a proper novelty search is an open item, listed in section 14.

## 7. Architecture

```
upstream feed entry (claim, source, evidence, provenance metadata)
        |
        v
  [ ROUTE ]  does this add, or overwrite something held?
        |            both paths continue; neither bypasses verification
        v
  [ SIGNALS ]
        provenance ...... editor history, account age, source reputation
        verification .... retrieve evidence, NLI entailment
        relation prior .. update_share and friends, if they earn their place
        belief energy ... conflict with held knowledge (stage 4)
        |
        v
  [ COMBINER ] ---> admit | refuse | quarantine
        |
     (admit)
        v
  [ EDITOR ]  UltraEdit | AlphaEdit | ROME | MEMIT
        |
  .......v.......................................
  : sampled shadow copy -> probe -> rollback     :
  ...............................................
```

Five decisions worth defending.

**No path bypasses verification.** The previous draft sent accretion straight to a provenance check on the grounds that adding cannot contradict. That was wrong, so both kinds get scored. The accretion/revision distinction stays as a feature and as a cost dimension, and system cost is reported across revision shares of 20%, 50% and 100% rather than assuming WikiBigEdit's mix.

**The gate is pre-commit and runs on the unedited model.** It asks whether a claim conflicts with what's held, not whether an edit hurt. That keeps it O(1) per candidate and batchable, which is what a gate needs to survive the volume that motivates it. Measuring after the edit would cost a full edit per candidate, and section 3 says you cannot rely on undoing one.

**The gate never touches editor internals.** Editor state of the art moved twice in six months. A new method should be a row in a table.

**Three-way output.** Quarantine exists because the right answer for a genuinely novel true fact is to hold it pending evidence. Without that class the false-refusal number is a fiction.

**Rollback is measured, not assumed.** The shadow-copy audit is how we test whether rollback works at all, given section 3. If it doesn't, that strengthens the case for pre-commit gating and it belongs in the paper either way.

### 7.1 The belief-state energy, stage 4

Train a predictor over the frozen core's latent trajectory, then score a candidate by the predictor's error on a probe context built from the claim and its retrieved neighbourhood, normalized against a matched control.

No theorem supports this. NextLat's belief-state result requires joint training of the representation and does not transfer to a frozen core. The hypothesis is empirical and rests on the ACL Findings result that the recall axis separates even where truth does not. If it adds nothing, it gets cut and reported as a negative.

### 7.2 The combiner

Logistic regression over the signal vector first, escalating only on demonstrated underfit. Trained on one attack family and tested on held-out families, with cross-family generalization reported as a headline number rather than an appendix, because a gate trained on its own synthetic attacks otherwise learns the generator.

## 8. Research questions

**RQ1, does poison survive the pipeline?** For a poisoned entry that enters an automated maintenance pipeline: does it survive N subsequent benign edits, does it survive the upstream revert that corrects the source, and does it corrupt multi-hop reasoning that depends on it while general-ability probes stay flat? How does survival differ between stable editors and collapse-prone ones?

**RQ2, can the gate separate poison from legitimate change?** At the base rate real vandalism occurs, what is the operating curve, the precision, the latency per candidate and the share of legitimate knowledge refused?

**RQ3, do relation-level priors help?** Does conditioning on relation features beat a single global threshold, and what is the error rate of the assignment itself?

**RQ4, does belief-state energy add anything?** Specifically on stealth cases that pass individual verification while conflicting with held knowledge.

**RQ5, does it survive an adaptive attacker?** White-box knowledge of the gate, optimizing entries to minimize the gate's score while preserving the payload.

RQ1 is the finding. RQ2 is the system. RQ4 decides whether the project keeps its name.

## 9. Expected results

| RQ | Expected | If it fails |
|---|---|---|
| 1 | Poison survives benign accumulation under stable editors and degrades under collapse-prone ones, so stability trades against recoverability. Reverts recover the surface answer while leaving the fact elicitable, following the erasure result. Multi-hop corruption persists with general ability flat | If reverts cleanly remove poison, the pipeline self-heals, the gate is much less necessary, and the honest paper is that negative result plus the conditions under which it holds |
| 2 | AUROC 0.80 to 0.90. Precision poor at the true base rate, because a rare positive class punishes any imperfect specificity, which is the practical finding operators need | Below 0.7 AUROC the gate is the bottleneck; characterize what is undetectable and report it |
| 3 | Modest lift, a few points, concentrated where the prior is strongest | No lift means relation priors are predictable but not decision-relevant, which is worth one paragraph and no more |
| 4 | Lift on the stealth subset, near zero on crude attacks | Zero lift cuts the JEPA half as an honest ablation. RQ1, 2, 3 and 5 remain a complete paper, and the project gets renamed |
| 5 | Degradation under white-box adaptation, with provenance holding where the learned signal bends | Full collapse is a strong publishable negative |

The headline figure is precision against recall at the measured base rate, not ROC on balanced data. Balanced-data AUROC is the number everyone reports and it flatters a detector that would be unusable in the deployment it claims to serve.

## 10. Experimental design

| | |
|---|---|
| Scales | 8B primary. 27B second scale, Gemma-3-27B, which is in UltraEdit's own table |
| Editors | UltraEdit and AlphaEdit as the stable arm, ROME and MEMIT as the collapse-prone contrast. RLEdit if EasyEdit support is confirmed |
| Arms | untouched base / ungated pipeline / provenance only / plus verification / plus relation prior / plus belief energy |
| Seeds | 5, actually used for candidate ordering and sampling, not merely recorded |
| Matching | benign and poisoned entries drawn from the same relations, the same edit kinds and the same subject distribution, so the comparison is not measuring dataset difficulty |
| Edit counts | 1K, 10K, 100K, retained as an analysis dimension so the result is a curve |
| Base rate | poison injected at the rate observed in vandalism data, with a sensitivity sweep either side of it |
| Parallelism | 4 independent GPUs over seeds by editors by attack families. No collectives, so it survives whatever the topology turns out to be. No single job above 141 GB |

Benign stream from WikiBigEdit. Poisoned entries from real labelled Wikidata vandalism where the corpus supports it, which is the primary source because it needs no synthetic generator to be believable, plus EditRisk-Bench for downstream harm at multiple hops, plus generated families held out for the cross-family test. Generated families must differ mechanically, not just by label: the previous plan gave three families the same uniform random object substitution and called them different attacks.

General-ability probes match UltraEdit's set exactly so numbers sit beside theirs without translation.

## 11. Staging

| Stage | Delivers | Kill switch |
|---|---|---|
| 0 | Harness: editors behind one interface, a pipeline simulator carrying the feed and its reverts, corpora, shadow-copy plumbing, sharder. Claims no result | Not a gate |
| 1 | RQ1: survival, revert resistance, downstream harm, stable against collapse-prone | Clean revert recovery reframes the paper around that negative |
| 2 | RQ2 and RQ3: the gate, precision at base rate, latency, false refusal | AUROC below 0.7 means report the negative and characterize it |
| 3 | arXiv preprint on stages 1 and 2 | Fixed point, whatever is done ships |
| 4 | RQ4: belief-state predictor and energy | No lift cuts it and renames the project |
| 5 | RQ5: adaptive white-box attacker | Collapse is a publishable negative |
| 6 | Ablations, seeds, writeup, ICML 2027 | |

The first implementation plan covers stages 0 and 1. Stage 2 gets its own plan once RQ1 numbers exist, because the gate's signal design depends on which attacks actually survive and what stealth looks like in practice.

**On the schedule.** The last draft targeted a September workshop. Today is 2026-09-05, ICLR 2027 closes on 2026-09-25 and the NeurIPS workshops around 2026-09-30. Neither is reachable with a rebuilt threat model, a rebuilt plan and no results. Both are dropped rather than pretended at. ICML 2027 in late January is the target, roughly 4.5 months, with a preprint at stage 3.

## 12. Non-goals

- No vision, no V-JEPA.
- No multi-agent, no latent chain-of-thought. That is v1 and it stays frozen.
- No domain sub-agent routing. It becomes interesting once there is a substrate to route between.
- No new editor. We consume the editor family.
- No core retraining. Only the combiner and, at stage 4, the predictor take gradients.
- No orthogonal-LoRA consolidation.

## 13. What I want attacked

1. **Is the pipeline real?** The whole design assumes operators run automated feed-driven parametric editing at volumes that preclude review. If nobody does this, and everyone uses retrieval instead, the deployment is invented and the paper should be about retrieval poisoning.
2. **Does WikiBigEdit's revert data support RQ1?** RQ1 needs poison and its subsequent correction as an ordered pair in the feed. If the vandalism corpora and WikiBigEdit cannot be joined into that, RQ1 needs simulated reverts and loses realism.
3. **Is the stability-versus-recoverability trade the finding I think it is?** An editor preserving poison because it preserves everything may read as obvious to a reviewer even though the erasure interaction is not.
4. **Is precision-at-base-rate a contribution or a footnote?** I think reporting it is the useful part and that balanced AUROC is misleading here. That could be one paragraph rather than a headline.
5. **Does the novelty claim in section 6 survive a proper search?** It has not had one yet.
6. **Is RQ3 worth a research question?** Q1 shows relation priors are predictable and says nothing about whether they help a decision.
7. **Should this still be called U-JEPA?** If RQ4 comes back flat, the honest paper is systems and security with a negative JEPA ablation, and renaming then costs more than renaming now.

## 14. Open items

- Verify the vandalism corpus statistics against the primary sources before any number reaches the paper. The counts I have came from secondary summaries.
- Confirm whether vandalism annotations can be joined to WikiBigEdit entities, which decides item 2 above.
- Wikidata property statement counts from the query service, to compute a real churn rate.
- Multi-year snapshots, so slow-turnover relations stop reading as invariant.
- GPU topology is still unconfirmed. Nothing may assume NVLink.
- Novelty search on the section 6 claim.
