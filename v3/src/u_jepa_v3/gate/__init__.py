"""Stage 2: admission control. Decides before the edit is applied.

Import order matters here only in that base imports provenance, so provenance
comes first.
"""
from .base import GateContext, GateDecision, GateInput, GateScore, Signal, is_revision
from .combiner import Calibration, LinearCombiner, OperatingPoint, Thresholds, calibrate, sweep
from .provenance import SourceTrust, TrustTracker, attacker_sources, simulate_sources
from .rollback import RollbackAudit, ShadowLedger, audit_rollback
from .signals import (
    BeliefContradictionSignal,
    BurstSignal,
    PriorMismatchSignal,
    SlotChurnSignal,
    SourceTrustSignal,
    TypeViolationSignal,
    default_signals,
)

__all__ = [
    "GateContext", "GateDecision", "GateInput", "GateScore", "Signal", "is_revision",
    "Calibration", "LinearCombiner", "OperatingPoint", "Thresholds", "calibrate", "sweep",
    "SourceTrust", "TrustTracker", "attacker_sources", "simulate_sources",
    "RollbackAudit", "ShadowLedger", "audit_rollback",
    "BeliefContradictionSignal", "BurstSignal", "PriorMismatchSignal",
    "SlotChurnSignal", "SourceTrustSignal", "TypeViolationSignal", "default_signals",
]
