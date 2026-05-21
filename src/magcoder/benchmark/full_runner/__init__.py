"""Full MAG-Bench experiment runner integration."""

from magcoder.benchmark.full_runner.abms_functional_evaluator import (
    ABMSFunctionalResult,
    run_abms_case_tests,
    run_abms_case_tests_from_module,
)
from magcoder.benchmark.full_runner.candidate_adapter import (
    AdaptedCandidate,
    build_candidate_with_entrypoint_adapter,
)
from magcoder.benchmark.full_runner.case_loader import (
    FullBenchRunCase,
    load_full_bench_run_cases,
)
from magcoder.benchmark.full_runner.evaluation_adapter import (
    FullBenchEvaluationResult,
    evaluate_full_bench_code,
)
from magcoder.benchmark.full_runner.functional_evaluator import (
    FunctionalEvaluationReport,
    FunctionalTestResult,
    load_python_module_from_path,
    load_python_module_from_source,
    run_case_tests,
)
from magcoder.benchmark.full_runner.input_modes import (
    SUPPORTED_FULL_INPUT_MODES,
    FullBenchPromptInput,
    build_full_bench_prompt_input,
)
from magcoder.benchmark.full_runner.passk_runner import (
    FullMagBenchPassKRunner,
    build_full_bench_summary_rows,
)
from magcoder.benchmark.full_runner.result_schema import (
    FullBenchPassKConfig,
    FullBenchPassKResult,
    FullBenchSampleResult,
)

__all__ = [
    "ABMSFunctionalResult",
    "SUPPORTED_FULL_INPUT_MODES",
    "AdaptedCandidate",
    "FullBenchEvaluationResult",
    "FullBenchPassKConfig",
    "FullBenchPassKResult",
    "FullBenchPromptInput",
    "FullBenchRunCase",
    "FullBenchSampleResult",
    "FunctionalEvaluationReport",
    "FunctionalTestResult",
    "FullMagBenchPassKRunner",
    "build_full_bench_prompt_input",
    "build_candidate_with_entrypoint_adapter",
    "build_full_bench_summary_rows",
    "evaluate_full_bench_code",
    "load_python_module_from_path",
    "load_python_module_from_source",
    "load_full_bench_run_cases",
    "run_abms_case_tests",
    "run_abms_case_tests_from_module",
    "run_case_tests",
]
