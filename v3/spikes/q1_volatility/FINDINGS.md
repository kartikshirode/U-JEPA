# Q1 spike: does knowledge split into invariant and volatile layers?

Date: 2026-08-11
Revised: 2026-09-05, after external review found the central metric measures something other than what I named it
Status: partially answered, verdict downgraded

## Read this first

The original version of this document claimed volatility is predictable, on the strength of a split-half Spearman of 0.695. That claim was too strong and I've withdrawn it.

The metric I called `churn` is `n_updates / n_rows`, where `n_rows` counts both revisions and additions. So it's the composition of observed change, not the probability that a fact changes. It's now named `update_share` in the code and throughout this document.

A real volatility estimate needs statements at risk in the denominator: of every Wikidata statement using property P, what share got revised in the window. This corpus can't give you that, because it only contains rows that already changed. A relation posts a high update share simply by rarely gaining new subjects, and a genuinely churning relation that's also growing fast posts a low one.

What survives is narrower but still useful. A relation's revision-to-addition mix is a stable trait you can predict from its own past. Whether the relation is volatile is a different question and it's still open.

## The question

The v3 architecture assumed knowledge sorts into two layers. Layer 1 holds facts that never change, like the sun rising in the east. Layer 2 holds facts that do, like which model is currently best at agentic work. Before building on that I wanted to know whether the split is real, and whether you can tell which layer a fact belongs to without a human deciding case by case.

The trick that makes it cheap: layer 1 doesn't have to mean "certainly true", only "certainly not going to change". Volatility is a much weaker property than truth and you can read it off time-stamped data.

## What I ran

WikiBigEdit ships 8 Wikidata snapshot diffs covering 2024-02-01 to 2024-07-01. Every row is a (subject, relation, object) triple that changed in that window, tagged `new` when an entity gained a property and `update` when an existing object got replaced.

After dropping 11,038 rows with a null subject or relation, that's 491,344 rows over 941 relations and 489,878 distinct (subject, relation) pairs.

Two scripts. `analyze_volatility.py` measures the update share per relation and the shape of its distribution. `analyze_stability.py` asks whether that share is a stable trait or one Wikidata bot pass landing in one snapshot.

## What came back

**Relations differ enormously in their update share.** CNC film rating sits at 0.000 across 2,288 rows. Site of astronomical discovery sits at 0.922 across 17,363. Instance of runs 0.656 over 57,150 rows.

**The distribution is one hump with a tail, not two humps.** Across the 206 relations with at least 200 rows, the histogram by decile reads 139, 39, 10, 8, 3, 0, 4, 0, 2, 1. Two thirds sit below 0.1, and 0.5% sit above 0.9. The bimodality coefficient reads 0.755 against a 0.556 reference, but it's picking up right skew rather than two modes. Trust the histogram.

**The share is predictable.** Update share on timesteps 0 to 3 correlates with update share on timesteps 4 to 7 at Spearman 0.695, p = 2.2e-41, across 278 relations. So the mix is a property of the relation. Note what this does and doesn't say: a relation's balance between revision and growth is stable, which is not the same as its facts being volatile at a predictable rate.

**Almost nothing changes twice.** Of 99,404 updated pairs, 98,491 changed once. 901 changed twice, 11 changed three times, 1 changed four times. That's 0.92% recurring. Recurrence is a genuine rate, but only over pairs already known to have changed at least once, and 5 months is short enough that it has almost no power.

**Update timing is mostly spread, not lumpy.** Median concentration, meaning the largest share of a relation's updates falling in one timestep, is 0.286. Only 5% of relations are lumpy at 0.8 or above, and those account for 1,407 updates against 78,129 in the evenly spread ones.

I originally read that as separating real-world change from curation, on the theory that the world trickles and bots arrive in a lump. That rule doesn't hold. Elections, transfer windows and sports seasons are genuinely lumpy real-world change, and scheduled bot maintenance can spread evenly across every snapshot. Concentration flags a relation worth looking at by hand. It doesn't classify one.

## Verdict

Partial pass, and weaker than I first wrote.

Holds up: relations differ hugely in the composition of their change, the difference is large, and it's stable enough over time to predict from history. Assignment doesn't need a human in the loop.

Doesn't hold up: this isn't volatility. Calling it that conflated a composition ratio with a rate. Nothing here licenses a threshold that decides how carefully to check an incoming fact, because the quantity a threshold would need is the one the corpus can't produce.

Also doesn't hold up: the binary. It's a continuum with most relations clustered near zero and a long tail. A hard two-layer split has to become a threshold with a measured error rate, whatever quantity it ends up thresholding.

## Before any of this sets a threshold

1. Get Wikidata property statement counts from the query service, and recompute as revisions divided by statements at risk. That's the number the architecture actually wants.
2. Extend past 5 months. Multi-year snapshots, so relations that turn over every 4 years stop reading as invariant.
3. Validate against raw unfiltered dumps rather than a QA benchmark built by filtering them.

## The 78/20 split, demoted

78% of rows are tagged `new`, 20% `update`, and 1.4% carry an empty tag.

I originally built an architectural decision on that, routing accretion down a cheap provenance-only path on the grounds that adding a fact can't contradict anything already held. Review killed both halves of that and it was right to.

The statistic itself is a WikiBigEdit property, not a property of knowledge. Four reasons:

- WikiBigEdit is a filtered QA benchmark. Its construction drops ambiguous subject-relation pairs holding multiple objects, and applies language and entity filters on top.
- The denominator holds only surviving changed triples. Not all knowledge, not all Wikidata statements, and not the candidate stream a real maintainer would see.
- `new` means absent from the earlier Wikidata snapshot. It does not mean absent from the model's parameters, and those are different things.
- Knowledge graphs are open world. A missing relation doesn't establish that the model holds no conflicting fact.

The reasoning was worse than the statistic. "Adding can't contradict anything" is false. A newly added criminal conviction, medical condition or affiliation is harmful without overwriting the same Wikidata slot, and it can contradict a different relation or the model's semantic neighbourhood even when no slot collides.

So accretion keeps a verification floor. The split stays as a feature the gate can use, and system cost gets reported across revision shares of 20%, 50% and 100% rather than assuming one number.

## Files

- `load_wikibigedit.py` downloads the 8 timesteps and flattens them
- `analyze_volatility.py` writes `results.json` and `per_relation.csv`
- `analyze_stability.py` writes `stability.json`
