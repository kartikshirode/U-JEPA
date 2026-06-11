"""Evaluation primitives for U-JEPA v2.

`edit_metrics` holds the measurement functions the F1 spike uses to judge an
editing substrate (edit-success, locality, perplexity). Two of those signals
(locality and perplexity) carry forward into the Phase 2 probe, so they live
here rather than inside the spike script.
"""

from .edit_metrics import (
    Edit,
    match_prefix,
    mean_heldout_perplexity,
    normalize,
    perplexity_from_logits,
    score_edits,
    sequence_perplexity,
)

__all__ = [
    "Edit",
    "match_prefix",
    "mean_heldout_perplexity",
    "normalize",
    "perplexity_from_logits",
    "score_edits",
    "sequence_perplexity",
]
