"""Check local readiness for a controlled Full MAG-Bench real-model run."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_frozen_paper2500_benchmark import (
    validate_frozen_paper2500_benchmark,
)


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Check readiness for a Full MAG-Bench real provider run."
    )
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--frozen-benchmark", type=Path, default=None)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", "--model-name", dest="model_name", required=True)
    parser.add_argument("--samples-per-case", type=int, default=2)
    parser.add_argument("--expected-cases", type=int, default=None)
    parser.add_argument("--planned-max-cases", type=int, default=None)
    parser.add_argument("--planned-samples-per-case", type=int, default=None)
    parser.add_argument("--modes", default="structured_graph_nl,graphical_graph_nl")
    parser.add_argument("--methods", default="full_with_refinement,no_refinement")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check_readiness(
        bench_path=args.bench,
        frozen_benchmark_path=args.frozen_benchmark,
        provider=args.provider,
        model_name=args.model_name,
        samples_per_case=args.samples_per_case,
        expected_cases=args.expected_cases,
        planned_max_cases=args.planned_max_cases,
        planned_samples_per_case=args.planned_samples_per_case,
        modes=_split(args.modes),
        methods=_split(args.methods),
        output_path=args.output,
    )
    _save_json(report, args.output)
    print(f"Readiness status: {report['readiness_status']}")
    print(f"Readiness report: {args.output}")


def check_readiness(
    *,
    bench_path: str | Path,
    frozen_benchmark_path: str | Path | None = None,
    provider: str,
    model_name: str,
    samples_per_case: int,
    modes: list[str],
    methods: list[str],
    output_path: str | Path,
    expected_cases: int | None = None,
    planned_max_cases: int | None = None,
    planned_samples_per_case: int | None = None,
) -> dict[str, Any]:
    """Return a structured readiness report without downloading any model."""

    bench = Path(bench_path)
    frozen_benchmark = (
        None if frozen_benchmark_path is None else Path(frozen_benchmark_path)
    )
    output = Path(output_path)
    checks: dict[str, Any] = {}
    warnings: list[str] = []
    checks["benchmark_root_exists"] = bench.exists() and bench.is_dir()
    checks["frozen_benchmark_path"] = (
        None if frozen_benchmark is None else str(frozen_benchmark)
    )
    checks["frozen_benchmark_exists"] = (
        None if frozen_benchmark is None else frozen_benchmark.exists()
    )
    frozen_validation: dict[str, Any] | None = None
    if frozen_benchmark is not None and frozen_benchmark.exists():
        try:
            frozen_validation = validate_frozen_paper2500_benchmark(frozen_benchmark)
        except Exception as error:  # noqa: BLE001 - readiness should stay structured.
            frozen_validation = {
                "passed": False,
                "errors": [f"{type(error).__name__}: {error}"],
            }
    checks["frozen_benchmark_validation"] = frozen_validation
    checks["frozen_benchmark_valid"] = (
        None
        if frozen_benchmark is None
        else bool(frozen_validation and frozen_validation.get("passed"))
    )
    case_count = _case_count(bench)
    checks["case_count"] = case_count
    checks["expected_50_cases"] = case_count >= 50
    checks["expected_cases"] = expected_cases
    checks["expected_cases_match"] = (
        True if expected_cases is None else case_count == expected_cases
    )
    reports = {
        "validation_report": bench / "validation_report.json",
        "oracle_validation_report": bench / "oracle_validation_report.json",
        "adapter_validation_report": bench / "adapter_validation_report.json",
        "abms_oracle_validation_report": bench / "abms_oracle_validation_report.json",
    }
    checks["quality_reports"] = {
        name: path.exists() for name, path in reports.items()
    }
    checks["quality_reports_exist"] = all(checks["quality_reports"].values())
    checks["output_parent_writable"] = _parent_writable(output)
    checks["python_version"] = sys.version.split()[0]
    checks["platform"] = platform.platform()
    checks["git_commit"] = _git_commit()
    checks["estimated_sample_count"] = (
        case_count * samples_per_case * len(modes) * len(methods)
    )
    planned_cases = planned_max_cases if planned_max_cases is not None else case_count
    planned_samples = (
        planned_samples_per_case
        if planned_samples_per_case is not None
        else samples_per_case
    )
    checks["planned_max_cases"] = planned_max_cases
    checks["planned_samples_per_case"] = planned_samples_per_case
    checks["planned_estimated_sample_count"] = (
        min(case_count, planned_cases) * planned_samples * len(modes) * len(methods)
    )
    fake_smoke_candidates = [
        bench / "passk_runs" / "fake_paper_smoke",
        bench / "passk_runs" / "fake_2500_smoke",
        bench / "passk_runs" / "fake_medium_smoke",
    ]
    checks["fake_smoke_exists"] = any(path.exists() for path in fake_smoke_candidates)
    checks["fake_smoke_candidates"] = [
        str(path) for path in fake_smoke_candidates if path.exists()
    ]
    checks["provider"] = provider
    checks["model_name"] = model_name
    checks["model_download_attempted"] = False
    if provider == "ollama_native":
        ollama_path = shutil.which("ollama")
        checks["ollama_command_available"] = ollama_path is not None
        checks["ollama_command_path"] = ollama_path
        model_check = _check_ollama_model(model_name) if ollama_path else {
            "available": False,
            "models": [],
            "error": "ollama command not found",
        }
        checks["model_available"] = bool(model_check["available"])
        checks["available_models"] = model_check["models"]
        if model_check.get("error"):
            warnings.append(str(model_check["error"]))
    else:
        checks["ollama_command_available"] = None
        checks["model_available"] = provider in {"fake", "dry_run"}
        checks["available_models"] = []
        if provider not in {"fake", "dry_run"}:
            warnings.append(f"No local readiness probe implemented for {provider}.")
    required = [
        checks["benchmark_root_exists"],
        checks["expected_50_cases"],
        checks["quality_reports_exist"],
        checks["expected_cases_match"],
        checks["output_parent_writable"],
        bool(checks["model_available"]),
    ]
    if frozen_benchmark is not None:
        required.append(bool(checks["frozen_benchmark_valid"]))
    readiness_status = "ready" if all(required) else "unavailable"
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bench_path": str(bench),
        "output_path": str(output),
        "readiness_status": readiness_status,
        "checks": checks,
        "warnings": warnings,
        "notes": [
            "This script never pulls or downloads models.",
            "If model_available is false, start Ollama or choose an installed model.",
        ],
    }


def _case_count(bench: Path) -> int:
    cases_dir = bench / "cases"
    if not cases_dir.exists():
        return 0
    return sum(1 for item in cases_dir.iterdir() if item.is_dir())


def _parent_writable(path: Path) -> bool:
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        probe = parent / ".readiness_write_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True


def _check_ollama_model(model_name: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["ollama", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "models": [], "error": str(error)}
    if completed.returncode != 0:
        return {
            "available": False,
            "models": [],
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    models = _parse_ollama_list(completed.stdout)
    return {
        "available": model_name in models,
        "models": models,
        "error": "" if model_name in models else f"model {model_name!r} not found",
    }


def _parse_ollama_list(output: str) -> list[str]:
    models: list[str] = []
    for line in output.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    main()
