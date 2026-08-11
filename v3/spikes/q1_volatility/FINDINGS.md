# Q1 spike: does knowledge split into invariant and volatile layers?

Date: 2026-08-11
Status: answered, with a correction to the architecture

## The question

The v3 architecture assumed knowledge sorts into two layers. Layer 1 holds facts that never change (the sun rises in the east). Layer 2 holds facts that do (the current best agentic model). Before building anything on that, I wanted to know whether the split is real and whether you can tell which layer a fact belongs to without a human deciding case by case.

The trick that makes it cheap: layer 1 doesn't have to mean "certainly true", only "certainly not going to change". Volatility is a much weaker property than truth and you can read it straight off time-stamped data.

## What I ran

WikiBigEdit ships 8 Wikidata snapshot diffs covering 2024-02-01 to 2024-07-01. Every row is a (subject, relation, object) triple that changed in that window, tagged `new` when an entity gained a property and `update` when an existing object got replaced.

After dropping 11,038 rows with a null subject or relation, that's 491,344 rows over 941 relations and 489,878 distinct (subject, relation) pairs.

Two scripts. `analyze_volatility.py` measures how often each relation gets revised and what shape the distribution takes. `analyze_stability.py` asks whether that rate is a stable trait you could predict, or just one Wikidata bot pass landing in one snapshot.

## What came back

**Most knowledge change is growth, not revision.** 78% of rows are `new`, 20% are `update` and the remaining 1.4% carry an empty tag. So four fifths of what happens to a knowledge base is adding facts that weren't there before.

**Almost nothing changes twice.** Of 99,404 updated pairs, 98,491 changed once. 901 changed twice, 11 changed three times, 1 changed four times. That's 0.92% recurring.

**Relations differ enormously and the gap is large.** CNC film rating sits at 0.000 churn across 2,288 rows. Site of astronomical discovery sits at 0.922 across 17,363. Instance of runs 0.656 over 57,150 rows.

**But the distribution is one hump with a tail, not two humps.** Across the 206 relations with at least 200 rows, the churn histogram is 139, 39, 10, 8, 3, 0, 4, 0, 2, 1 across deciles. 67% fall below 0.1 churn and 0.5% sit above 0.9. The bimodality coefficient reads 0.755 against a 0.556 reference, but that's picking up the right skew rather than two modes. The histogram is the honest picture.

**The rate is predictable, which is the result that matters.** Churn measured on timesteps 0-3 correlates with churn measured on timesteps 4-7 at Spearman 0.695, p = 2.2e-41. So how much a relation churns is a property of the relation, not noise.

**And it isn't a curation artifact.** Median concentration (the largest share of a relation's updates falling in any single timestep) is 0.286. Only 5% of relations are lumpy at 0.8 or above, and those account for 1,407 updates against 78,129 in the evenly spread ones. Updates trickle across all 8 timesteps rather than arriving in sweeps.

## Verdict

Pass, but the two-layer story needs fixing.

What holds up: volatility is real, it's large, it's stable over time and you can predict it from the relation. Layer assignment doesn't need a human in the loop.

What doesn't: it isn't binary. It's a continuum with most relations clustered near zero and a long tail running up to near-total churn. A hard two-layer split has to become a threshold on that continuum, with a measured error rate attached.

I'd argue the fix is an improvement. An assumed binary is a design decision reviewers can argue with. A threshold with a reported ROC is a result.

## The thing I didn't ask for, which matters more

78% of change is accretion. Adding a fact that didn't exist can't contradict anything already held, so it needs a provenance check and nothing else. Every dangerous operation, every overwrite of something already believed, lives in the 20% tagged `update`.

That reframes the gate. The expensive coherence machinery only has to run on a fifth of the traffic. The rest takes a cheap path.

Worth flagging that the 78/20 split could partly reflect how WikiBigEdit was built rather than how Wikidata actually moves, so it needs checking against raw dumps before the paper leans on it.

## Limits on what this shows

Five months is a short window. A relation that genuinely turns over every four years (head of state, say) shows zero recurrence here, so read low recurrence as absence of evidence and not as proof of invariance. Confirming true invariance needs multi-year snapshots.

Wikidata also mixes two things the design cares about separately. When a boxer remarries, the world changed. When a bot reclassifies 37,477 instance-of statements, the database got tidied. Both look the same in a diff. Position held, member of political party and member of sports team read like real world churn; instance of and site of astronomical discovery read like ontology work.

Base rates are missing too. A relation can look busy just by being common. Normalising by how many Wikidata statements use each property would give a true churn rate, and that needs a query service pass I haven't run.

## Files

- `load_wikibigedit.py` downloads the 8 timesteps and flattens them
- `analyze_volatility.py` writes `results.json` and `per_relation.csv`
- `analyze_stability.py` writes `stability.json`
