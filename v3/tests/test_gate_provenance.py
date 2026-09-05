"""Account trust, the simulation behind it, and when outcomes are credited."""
from __future__ import annotations

import pytest

from u_jepa_v3.gate.provenance import SourceTrust, TrustTracker, simulate_sources
from u_jepa_v3.schema import EditCandidate, EditKind, FeedEntry


def cand(subj, obj, adversarial=False):
    return EditCandidate(
        subject_id=subj, subject=f"S{subj}", relation_id="P26", relation="spouse",
        object_id=None, object=obj, prompt=f"Who is the spouse of S{subj}?",
        kind=EditKind.REVISION, source="wikibigedit", timestep=0,
        is_adversarial=adversarial,
        risk_category="misinformation" if adversarial else None, n_hops=1,
    )


def feed(n_benign=40, n_poison=6):
    entries = []
    position = 0
    for i in range(n_benign):
        entries.append(FeedEntry(candidate=cand(f"Q{i}", f"o{i}"), position=position,
                                 entry_id=f"benign-{i}", is_poison=False,
                                 reverts=None, attack_family=None))
        position += 1
    for i in range(n_poison):
        entries.append(FeedEntry(candidate=cand(f"P{i}", f"bad{i}", True),
                                 position=position, entry_id=f"poison-{i}",
                                 is_poison=True, reverts=None,
                                 attack_family="type_consistent"))
        position += 1
    return entries


def test_an_unknown_account_gets_exactly_the_prior():
    trust = SourceTrust(prior_trust=0.5, prior_weight=10)
    assert trust.trust("never-seen") == 0.5


def test_reverts_pull_trust_down_and_clean_entries_pull_it_up():
    trust = SourceTrust()
    for _ in range(20):
        trust.observe("bad", reverted=True)
        trust.observe("good", reverted=False)
    assert trust.trust("bad") < 0.2
    assert trust.trust("good") > 0.8


def test_a_heavier_prior_absorbs_more_before_moving():
    light, heavy = SourceTrust(prior_weight=2), SourceTrust(prior_weight=50)
    for t in (light, heavy):
        for _ in range(3):
            t.observe("s", reverted=True)
    assert heavy.trust("s") > light.trust("s")


def test_the_record_reports_the_raw_counts():
    trust = SourceTrust()
    trust.observe("s", True)
    trust.observe("s", False)
    record = trust.record("s")
    assert (record.n_seen, record.n_reverted) == (2, 1)
    assert record.revert_rate == 0.5


def test_an_invalid_prior_is_refused():
    with pytest.raises(ValueError):
        SourceTrust(prior_trust=1.5)
    with pytest.raises(ValueError):
        SourceTrust(prior_weight=0)


def test_every_entry_gets_an_account():
    entries = feed()
    sources = simulate_sources(entries, seed=0)
    assert set(sources) == {e.entry_id for e in entries}


def test_poison_always_comes_from_an_attacker_account():
    entries = feed()
    sources = simulate_sources(entries, seed=1, n_sources=8, n_attacker_sources=2)
    attacker = {"src-00", "src-01"}
    assert all(sources[e.entry_id] in attacker for e in entries if e.is_poison)


def test_attacker_accounts_also_carry_ordinary_traffic():
    """Otherwise the account is the label and any gate scores perfectly."""
    entries = feed(n_benign=200, n_poison=10)
    sources = simulate_sources(entries, seed=2, cover_rate=0.35)
    attacker = {"src-00", "src-01"}
    cover = [e for e in entries if not e.is_poison and sources[e.entry_id] in attacker]
    assert len(cover) > 20


def test_no_cover_traffic_makes_the_account_a_perfect_label():
    entries = feed(n_benign=100, n_poison=10)
    sources = simulate_sources(entries, seed=3, cover_rate=0.0)
    attacker = {"src-00", "src-01"}
    flagged = {e.entry_id for e in entries if sources[e.entry_id] in attacker}
    poison = {e.entry_id for e in entries if e.is_poison}
    assert flagged == poison


def test_the_same_seed_gives_the_same_attribution():
    entries = feed()
    assert simulate_sources(entries, seed=7) == simulate_sources(entries, seed=7)


def test_a_bad_attacker_count_is_refused():
    with pytest.raises(ValueError):
        simulate_sources(feed(), seed=0, n_sources=4, n_attacker_sources=9)


def test_an_entry_is_credited_only_after_it_survives_the_lag():
    trust = SourceTrust()
    tracker = TrustTracker(trust, lag=10)
    tracker.submitted("e1", "src-00", position=0)
    tracker.advance(position=5)
    assert tracker.pending() == 1
    assert trust.record("src-00").n_seen == 0
    tracker.advance(position=10)
    assert tracker.pending() == 0
    assert trust.record("src-00").n_seen == 1


def test_a_correction_charges_the_account_that_submitted_the_original():
    trust = SourceTrust()
    tracker = TrustTracker(trust, lag=100)
    tracker.submitted("poison-0", "src-01", position=0)
    assert tracker.reverted("poison-0") is True
    assert trust.record("src-01").n_reverted == 1
    assert tracker.pending() == 0


def test_correcting_something_that_was_never_admitted_is_a_no_op():
    tracker = TrustTracker(SourceTrust(), lag=10)
    assert tracker.reverted("never-applied") is False


def test_an_entry_cannot_be_credited_twice():
    trust = SourceTrust()
    tracker = TrustTracker(trust, lag=5)
    tracker.submitted("e1", "src-00", position=0)
    tracker.advance(position=20)
    tracker.advance(position=40)
    assert trust.record("src-00").n_seen == 1
