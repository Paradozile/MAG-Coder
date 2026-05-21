"""Tests for ABMS oracle validation and runner modes."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.validate_full_mag_bench_abms_oracles import validate_abms_oracles

from magcoder.benchmark.full_builder import build_full_mag_bench
from magcoder.benchmark.full_runner import FullBenchPassKConfig, FullMagBenchPassKRunner

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / "examples/source_tasks/medium_humaneval_style_seed.jsonl",
    ROOT / "examples/source_tasks/medium_mbpp_style_seed.jsonl",
]


def test_abms_oracle_validation_passes_for_built_subset(tmp_path: Path) -> None:
    """ABMS gold passes and broken fails for a built benchmark subset."""

    bench = _build_subset(tmp_path)
    report = validate_abms_oracles(bench)
    assert report["passed"]
    assert report["supported_cases"] == 2
    assert report["gold_abms_pass_count"] == 2
    assert report["broken_abms_fail_count"] == 2


def test_full_runner_supports_evaluation_all(tmp_path: Path) -> None:
    """The Full runner emits ABMS fields when evaluation=all."""

    bench = _build_subset(tmp_path)
    output = tmp_path / "fake_all"
    result = FullMagBenchPassKRunner().run(
        FullBenchPassKConfig(
            bench_path=str(bench),
            output_dir=str(output),
            provider="fake",
            model_name="fake-generator",
            samples_per_case=1,
            k_values=[1],
            modes=["structured_graph_nl"],
            methods=["full_with_refinement"],
            max_cases=1,
            overwrite=True,
            evaluation="all",
        )
    )
    assert len(result.sample_results) == 1
    assert result.sample_results[0].abms_supported is not None
    with (output / "summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert "abms_pass" in rows[0]
    assert rows[0]["evaluation_mode"] == "all"


def _build_subset(tmp_path: Path) -> Path:
    bench = tmp_path / "bench"
    build_full_mag_bench(
        source_paths=SOURCES,
        output_dir=bench,
        max_cases=2,
        formats=["structured", "graphical"],
    )
    return bench
