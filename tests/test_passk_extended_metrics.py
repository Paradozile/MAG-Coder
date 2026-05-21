"""Tests for extended Full MAG-Bench pass@k and latency reporting."""

from __future__ import annotations

from magcoder.benchmark.full_runner import (
    FullBenchPassKConfig,
    FullBenchSampleResult,
    build_full_bench_summary_rows,
)


def test_extended_passk_outputs_na_for_k_greater_than_n() -> None:
    """pass@10/50/100 are supported and reported as N/A when n is too small."""

    samples = [
        _sample(index, success=index < 3, latency_seconds=None)
        for index in range(5)
    ]
    rows = build_full_bench_summary_rows(
        samples,
        FullBenchPassKConfig(
            bench_path="bench",
            output_dir="out",
            samples_per_case=5,
            k_values=[1, 3, 5, 10, 50, 100],
        ),
    )
    by_k = {row["k"]: row for row in rows}
    assert by_k[1]["contract_estimated_pass_at_k"] != "N/A"
    assert by_k[5]["abms_estimated_pass_at_k"] != "N/A"
    assert by_k[10]["contract_estimated_pass_at_k"] == "N/A"
    assert by_k[50]["functional_estimated_pass_at_k"] == "N/A"
    assert by_k[100]["abms_observed_pass_at_k"] == "N/A"
    assert by_k[1]["generation_time_seconds"] == "N/A"


def test_generation_time_averages_known_latencies() -> None:
    """Generation-time aggregation averages available sample latency fields."""

    rows = build_full_bench_summary_rows(
        [
            _sample(0, success=True, latency_seconds=1.0),
            _sample(1, success=True, latency_seconds=3.0),
        ],
        FullBenchPassKConfig(
            bench_path="bench",
            output_dir="out",
            samples_per_case=2,
            k_values=[1],
        ),
    )
    assert rows[0]["generation_time_seconds"] == "2.000"


def _sample(
    index: int,
    *,
    success: bool,
    latency_seconds: float | None,
) -> FullBenchSampleResult:
    return FullBenchSampleResult(
        case_id="case_a",
        input_mode="structured_graph_nl",
        method="full_with_refinement",
        sample_index=index,
        seed=100 + index,
        generation_result_path="generation.json",
        code_initial_path="code_initial.py",
        initial_report_path="initial.json",
        evaluation_path="evaluation.json",
        parse_success=True,
        framework_passed=success,
        mandatory_success=success,
        checkpoint_accuracy=1.0 if success else 0.0,
        success=success,
        cache_hit=False,
        contract_pass=success,
        functional_pass=success,
        abms_pass=success,
        latency_seconds=latency_seconds,
        metadata={
            "split": "dev",
            "source": "humaneval",
            "task_type": "strings",
            "data_type": "sequence",
            "template_family": "delegated",
            "mode_family": "structured",
            "nl_condition": "graph_nl",
            "baseline_family": "mag_coder_full",
            "refinement_strategy": "balanced",
        },
    )
