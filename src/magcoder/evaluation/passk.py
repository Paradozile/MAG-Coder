"""pass@k metrics and success criteria."""

from __future__ import annotations

from math import comb
from typing import Literal

SuccessCriterion = Literal[
    "mandatory_success",
    "framework_passed",
    "checkpoint_accuracy_1.0",
    "parse_and_framework_passed",
]


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Estimate pass@k with the standard unbiased estimator."""

    if num_samples < 0 or num_correct < 0 or k <= 0:
        raise ValueError("num_samples, num_correct, and k must be positive counts.")
    if num_correct > num_samples:
        raise ValueError("num_correct cannot exceed num_samples.")
    if num_correct == 0 or num_samples == 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - comb(num_samples - num_correct, k) / comb(num_samples, k)


def observed_pass_at_k(successes: list[bool], k: int) -> bool:
    """Return whether any success appears in the first k observed samples."""

    if k <= 0:
        raise ValueError("k must be positive.")
    return any(successes[:k])


def evaluate_success_criterion(
    *,
    criterion: str,
    parse_success: bool,
    framework_passed: bool,
    mandatory_success: bool,
    checkpoint_accuracy: float,
) -> bool:
    """Evaluate a supported per-sample success criterion."""

    if criterion == "mandatory_success":
        return mandatory_success
    if criterion == "framework_passed":
        return framework_passed
    if criterion == "checkpoint_accuracy_1.0":
        return checkpoint_accuracy >= 1.0
    if criterion == "parse_and_framework_passed":
        return parse_success and framework_passed
    raise ValueError(f"Unsupported success criterion: {criterion}")
