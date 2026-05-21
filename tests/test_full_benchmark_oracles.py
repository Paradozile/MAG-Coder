"""Tests for Full MAG-Bench oracle validation and functional runner mode."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.validate_full_mag_bench_oracles import validate_oracles

from magcoder.benchmark.full_runner import (
    FullBenchPassKConfig,
    FullMagBenchPassKRunner,
)

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks/full_mag_bench_v1"


def test_oracle_validation_report_passes_for_seed_benchmark() -> None:
    """All seed gold oracles pass and all broken oracles fail."""

    report = validate_oracles(BENCH)
    assert report["passed"]
    assert report["gold_passed_count"] == report["case_count"]
    assert report["broken_failed_count"] == report["case_count"]


def test_full_runner_evaluation_both_writes_functional_columns(
    tmp_path: Path,
) -> None:
    """The Full runner can emit both contract and functional evaluation output."""

    output = tmp_path / "fake_functional_smoke"
    config = FullBenchPassKConfig(
        bench_path=str(BENCH),
        output_dir=str(output),
        provider="fake",
        model_name="fake-generator",
        samples_per_case=1,
        k_values=[1],
        modes=["structured_graph_nl"],
        methods=["full_with_refinement", "no_refinement"],
        max_cases=1,
        overwrite=True,
        evaluation="both",
    )
    result = FullMagBenchPassKRunner().run(config)
    assert len(result.sample_results) == 2
    assert all(sample.evaluation_mode == "both" for sample in result.sample_results)
    assert all(sample.functional_pass is not None for sample in result.sample_results)

    with (output / "summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert "contract_pass" in rows[0]
    assert "functional_pass" in rows[0]
    assert {row["evaluation_mode"] for row in rows} == {"both"}
