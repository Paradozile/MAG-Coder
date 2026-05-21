"""Tests for Full MAG-Bench adapter validation."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.validate_full_mag_bench_adapters import validate_adapters

from magcoder.benchmark.full_builder import build_full_mag_bench
from magcoder.benchmark.full_runner import FullBenchPassKConfig, FullMagBenchPassKRunner

ROOT = Path(__file__).resolve().parents[1]
MEDIUM_SOURCES = [
    ROOT / "examples/source_tasks/medium_humaneval_style_seed.jsonl",
    ROOT / "examples/source_tasks/medium_mbpp_style_seed.jsonl",
]


def test_adapter_validation_passes_on_built_subset(tmp_path: Path) -> None:
    """Built Full MAG-Bench cases carry supported adapter specs."""

    bench = tmp_path / "bench"
    build_full_mag_bench(
        source_paths=MEDIUM_SOURCES,
        output_dir=bench,
        max_cases=4,
        formats=["structured", "graphical"],
    )
    report = validate_adapters(bench)
    assert report["passed"]
    assert report["supported_count"] == 4


def test_fake_runner_writes_adapter_fields_in_summary(tmp_path: Path) -> None:
    """Functional summary CSV exposes direct/adapter alignment diagnostics."""

    bench = tmp_path / "bench"
    build_full_mag_bench(
        source_paths=MEDIUM_SOURCES,
        output_dir=bench,
        max_cases=2,
        formats=["structured", "graphical"],
    )
    output = tmp_path / "fake_adapter"
    config = FullBenchPassKConfig(
        bench_path=str(bench),
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
    FullMagBenchPassKRunner().run(config)
    with (output / "sample_summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert "entrypoint_mode" in rows[0]
    assert "adapter_used" in rows[0]
    assert "required_candidate_symbols_present" in rows[0]
