# External review of the v3 spec and first implementation plan

Date: 2026-09-05
Reviewed: `docs/superpowers/specs/2026-08-11-u-jepa-v3-design.md`, `docs/superpowers/plans/2026-08-11-u-jepa-v3-harness-and-rq1.md`
Verdict returned: do not implement unchanged
Outcome: spec rewritten as `docs/superpowers/specs/2026-09-05-u-jepa-v3-design.md`, plan rewritten, Q1 metric corrected

Kept because the review is the reason v3 changed shape, and a reader who only sees the new spec would not know what it is reacting to.

## What I checked myself

Every finding was verified against the files rather than accepted. All of them held.

| Finding | Verification |
|---|---|
| update share is not churn | `stats["churn"] = stats["n_updates"] / stats["n_rows"]` where `n_rows` counts new plus update |
| edited model discarded | `editor.edit(**payload)` at plan line 1333, return value dropped, responder never connected |
| seeds unused | `config.seed` appears only inside checkpoint dicts, never in ordering or sampling |
| attack families identical | all 3 branches call the same `rng.choice(alternatives)`; family is a label passed to `_rewrite` |
| stage count inconsistent | section 12 asks about 5 stages before September while the table lists 3 |

## Framing challenges

**Is the threat model real?** Partly. The integrity risk is real, the delivery path was not specified. The killer observation: editors fit whatever target they are given without judging its truth, so a near-zero benign-versus-adversarial efficacy gap is expected by construction. RQ1 measured editor obedience.

Accepted. The new spec names one deployment, automated knowledge maintenance from a public feed, with an explicit attacker capability. RQ1 now asks whether poison survives the pipeline and its own correction, which the erasure literature says is open.

**Is "second on the concept, first with evidence" a contribution?** Not as phrased. EditRisk-Bench and BadEdit already provide empirical evidence that malicious editing works. SSGM governs external agent memory, and its stated hypotheses are not the ones v3 tests, so describing it as the architecture being implemented overstated the relationship.

Accepted. SSGM is now convergent motivation. The novelty claim is narrowed to pre-commit admission control for long-horizon parametric editing under matched writes, a realistic base rate, multiple stable editors and an adaptive attacker, and a novelty search on that claim is an open item rather than an assumption.

**Is 78/20 a property of knowledge?** No. It is a WikiBigEdit statistic: the corpus is a filtered QA benchmark, the denominator holds only surviving changed triples, `new` means absent from the earlier snapshot rather than absent from the model, and knowledge graphs are open world. Worse, "adding cannot contradict anything" is false, so the provenance-only bypass was unsafe regardless.

Accepted in full. The bypass is deleted, no path skips verification, accretion survives as a feature and a cost dimension, and cost is reported across revision shares of 20%, 50% and 100%.

## Findings and dispositions

| # | Finding | Disposition |
|---|---|---|
| 1 | The churn statistic is not churn; split-half shows composition is stable, not that facts are volatile | Renamed `update_share`, docstrings rewritten, FINDINGS verdict downgraded, real rate listed as a prerequisite needing query service data |
| 2 | Harness cannot observe an edited model; tests miss it because the fake responder is never updated either | Adapter rebuilt to capture the returned model and expose a responder bound to it. Test doubles rebuilt so a responder that never updates fails a test |
| 3 | Resume is scientifically invalid: counters saved, weights and editor state not | Cells become atomic. Only finished cells are skipped; an interrupted cell reruns from zero |
| 4 | Worker never runs an experiment, only prints "would run" | Real run path: cell to model, editor and corpus, driver invocation, device assignment |
| 5 | Attack families are labels, not families; benign and adversarial arms unmatched | Families differ mechanically. Real labelled vandalism becomes the primary adversarial source. Benign and poisoned entries matched on relation, edit kind and subject distribution |
| 6 | RQ1 does not measure its persistence claim | RQ1 rewritten around survival through benign accumulation, survival through the upstream revert, and downstream multi-hop harm |
| 7 | Seeds recorded but never used; first-N loader takes a sorted prefix | Seed drives ordering and sampling. Sampling replaces the lexicographic prefix |
| 8 | Analysis groups only by editor and corpus, losing model, edit count, family, edit kind and checkpoint | All dimensions retained as grouping keys, which is what the 1K/10K/100K curves need |
| 9 | The NextLat theorem does not transfer to a frozen core | Dropped as mechanism justification. RQ4 is an empirical hypothesis resting on the ACL Findings recall result |
| 10 | Schedule does not close: workshop needs stage 2, stage 2 has no plan, prerequisites deferred | September targets dropped as unreachable. ICML 2027 with a preprint at stage 3 |

## Where I scoped differently

The review recommended replacing tasks 7 through 12. Tasks 7, 11 and 12 are replaced. Tasks 8, 9 and 10 are modified instead:

- Task 8, the probes, is sound. The probes do not know where answers come from. What was broken is that nothing bound a responder to the edited model.
- Task 9 is a correct atomic write. What was invalid is the resume policy, not the state file, so the fix is making cells atomic rather than rewriting the mechanism. Checkpointing 8B to 27B weights plus editor normalization state per interval is affordable on H200s and buys resume at a price not worth paying, and a half-resumed editor is exactly the silent corruption that ruins a long run.
- Task 10's grid and sharding are fine. Only the worker's run path was a stub.

## The pattern worth recording

This is the second time the framing collapsed. UltraEdit killed the v2 claim in May, and the missing delivery path killed the first v3 RQ1 in September. Both times the framing was derived from literature rather than from a deployment, so it inherited whatever the cited papers assumed and never had to survive contact with a system anyone would run.

The rewrite starts from a named pipeline with a stated trust boundary. That is the structural fix, not a better choice of citations.
