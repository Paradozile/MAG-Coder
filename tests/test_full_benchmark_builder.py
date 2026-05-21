"""Tests for Full MAG-Bench v1 package construction."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from magcoder.benchmark.full_builder import (
    build_full_mag_bench,
    load_full_benchmark_package,
    validate_full_mag_bench,
)

ROOT = Path(__file__).resolve().parents[1]


def test_full_benchmark_package_can_build_and_validate(tmp_path: Path) -> None:
    """The seed pipeline builds a valid package without real LLM calls."""

    output = tmp_path / "full_mag_bench_v1"
    package, report = build_full_mag_bench(
        source_paths=[
            ROOT / "examples/source_tasks/humaneval_style_seed.jsonl",
            ROOT / "examples/source_tasks/mbpp_style_seed.jsonl",
        ],
        output_dir=output,
        max_cases=4,
        formats=["structured", "graphical"],
    )
    assert package.benchmark_id == "full_mag_bench_v1"
    assert report.case_count == 4
    validation = validate_full_mag_bench(output)
    assert validation.passed
    first_case = output / package.cases[0].case_dir
    for filename in [
        "case.yaml",
        "source_task.json",
        "objectified_task.json",
        "mag.yaml",
        "mag.dot",
        "mag_manifest.json",
        "tests.json",
        "tests.py",
        "gold_solution.py",
        "broken_solution.py",
        "framework.yaml",
        "metadata.json",
    ]:
        assert (first_case / filename).exists()


def test_full_benchmark_package_can_be_loaded(tmp_path: Path) -> None:
    """Generated benchmark metadata round-trips through benchmark.yaml."""

    output = tmp_path / "bench"
    build_full_mag_bench(
        source_paths=[ROOT / "examples/source_tasks/humaneval_style_seed.jsonl"],
        output_dir=output,
        max_cases=2,
    )
    package = load_full_benchmark_package(output)
    assert len(package.cases) == 2
    assert package.cases[0].supported_input_modes == ["structured", "graphical"]


def test_build_and_validate_clis_work_on_seed_benchmark(tmp_path: Path) -> None:
    """CLI entry points build and validate the seed benchmark."""

    output = tmp_path / "cli_bench"
    build_cmd = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/build_full_mag_bench.py"),
        "--source",
        str(ROOT / "examples/source_tasks/humaneval_style_seed.jsonl"),
        "--source",
        str(ROOT / "examples/source_tasks/mbpp_style_seed.jsonl"),
        "--output",
        str(output),
        "--max-cases",
        "3",
        "--formats",
        "structured,graphical",
        "--overwrite",
    ]
    subprocess.run(build_cmd, check=True, cwd=ROOT, capture_output=True, text=True)
    validate_cmd = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/validate_full_mag_bench.py"),
        "--bench",
        str(output),
    ]
    subprocess.run(validate_cmd, check=True, cwd=ROOT, capture_output=True, text=True)
    report = json.loads((output / "validation_report.json").read_text())
    assert report["passed"] is True
