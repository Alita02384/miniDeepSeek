"""Consistency gates: log-probs, router replay, checkpoint round-trips."""

from .checks import (
    CheckpointReport,
    LogprobReport,
    RoutingReport,
    check_checkpoint_consistency,
    check_logprob_consistency,
    check_routing_consistency,
    run_all_checks,
)

__all__ = [
    "CheckpointReport",
    "LogprobReport",
    "RoutingReport",
    "check_checkpoint_consistency",
    "check_logprob_consistency",
    "check_routing_consistency",
    "run_all_checks",
]
