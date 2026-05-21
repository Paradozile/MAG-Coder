"""Evaluation and ablation utilities."""

from magcoder.evaluation.checkpoints import CheckpointResult, evaluate_checkpoints
from magcoder.evaluation.metrics import (
    checkpoint_accuracy,
    framework_passed,
    framework_violation_count,
    mandatory_success,
)
from magcoder.evaluation.pilot import (
    BackendCheckConfig,
    BackendCheckResult,
    PilotAuditResult,
    PilotComparisonResult,
    PilotRunBundle,
    PilotRunConfig,
    audit_real_pilot,
    check_generator_backend,
    compare_pilot_runs,
    load_pilot_run_config,
    run_pilot_from_config,
    save_backend_check_result,
)
from magcoder.evaluation.report import (
    save_scenario_run_result,
    write_scenario_summary_reports,
)
from magcoder.evaluation.runner import (
    ScenarioEvaluationResult,
    ScenarioEvaluationRunner,
    ScenarioRunResult,
)
from magcoder.evaluation.scenario import ScenarioSpec, load_scenario_spec

__all__ = [
    "BackendCheckConfig",
    "BackendCheckResult",
    "CheckpointResult",
    "PilotComparisonResult",
    "PilotAuditResult",
    "PilotRunBundle",
    "PilotRunConfig",
    "ScenarioEvaluationResult",
    "ScenarioEvaluationRunner",
    "ScenarioRunResult",
    "ScenarioSpec",
    "audit_real_pilot",
    "checkpoint_accuracy",
    "check_generator_backend",
    "compare_pilot_runs",
    "evaluate_checkpoints",
    "framework_passed",
    "framework_violation_count",
    "load_pilot_run_config",
    "load_scenario_spec",
    "mandatory_success",
    "run_pilot_from_config",
    "save_backend_check_result",
    "save_scenario_run_result",
    "write_scenario_summary_reports",
]
