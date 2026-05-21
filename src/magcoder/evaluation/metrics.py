"""Metric helpers for scenario evaluation."""

from __future__ import annotations

from magcoder.evaluation.checkpoints import CheckpointResult
from magcoder.refinement import ConsistencyReport


def checkpoint_accuracy(checkpoints: list[CheckpointResult]) -> float:
    """Compute the fraction of passed checkpoints."""

    if not checkpoints:
        return 0.0
    passed = sum(1 for checkpoint in checkpoints if checkpoint.passed)
    return passed / len(checkpoints)


def mandatory_success(
    checkpoints: list[CheckpointResult],
    mandatory_checkpoint_ids: list[str],
) -> bool:
    """Return whether all mandatory checkpoints passed."""

    checkpoints_by_id = {
        checkpoint.checkpoint_id: checkpoint for checkpoint in checkpoints
    }
    return all(
        checkpoints_by_id.get(checkpoint_id) is not None
        and checkpoints_by_id[checkpoint_id].passed
        for checkpoint_id in mandatory_checkpoint_ids
    )


def framework_passed(report: ConsistencyReport) -> bool:
    """Return whether the framework-level report passed."""

    return report.passed and not report.framework_violations


def framework_violation_count(report: ConsistencyReport) -> int:
    """Return the number of framework violations."""

    return len(report.framework_violations)
