"""Load generated Full MAG-Bench cases for experiment runners."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from magcoder.benchmark.full_builder import (
    ABMSOracleSpec,
    FullBenchmarkCase,
    load_full_benchmark_package,
)
from magcoder.benchmark.full_builder.adapter_spec import EntryPointAdapterSpec


@dataclass(slots=True)
class FullBenchRunCase:
    """Loaded case artifact paths and metadata for a Full MAG-Bench run."""

    case_id: str
    case_dir: Path
    source_task: dict[str, Any]
    objectified_task: dict[str, Any]
    mag_yaml_path: Path
    mag_json_path: Path
    dot_path: Path
    manifest_path: Path
    tests_json_path: Path
    tests_py_path: Path
    gold_solution_path: Path
    broken_solution_path: Path
    abms_oracle_path: Path
    abms_tests_py_path: Path
    gold_abms_solution_path: Path
    broken_abms_solution_path: Path
    framework_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    package_case: FullBenchmarkCase

    @property
    def expected_contract(self) -> dict[str, Any]:
        """Return expected contract metadata from tests.json."""

        tests = _load_json(self.tests_json_path)
        contract = tests.get("expected_contract", {})
        if not isinstance(contract, dict):
            raise TypeError(f"tests.json expected_contract must be an object: {self}")
        return dict(contract)

    @property
    def entry_point_adapter_spec(self) -> EntryPointAdapterSpec | None:
        """Return the source-to-MAG functional adapter spec when available."""

        tests = _load_json(self.tests_json_path)
        raw_spec = tests.get("entry_point_adapter_spec")
        if raw_spec is None:
            raw_spec = self.metadata.get("entry_point_adapter_spec")
        if raw_spec is None:
            raw_spec = self.objectified_task.get("metadata", {}).get(
                "entry_point_adapter_spec",
            )
        if raw_spec is None:
            return None
        if not isinstance(raw_spec, Mapping):
            raise TypeError(
                f"entry_point_adapter_spec must be an object for {self.case_id}."
            )
        return EntryPointAdapterSpec.from_dict(raw_spec)

    @property
    def abms_oracle_spec(self) -> ABMSOracleSpec | None:
        """Return the ABMS-level oracle spec when available."""

        raw_spec: Any | None = None
        if self.abms_oracle_path.exists():
            raw_spec = _load_json(self.abms_oracle_path)
        if raw_spec is None:
            raw_spec = self.metadata.get("abms_oracle_spec")
        if raw_spec is None:
            raw_spec = self.objectified_task.get("metadata", {}).get(
                "abms_oracle_spec",
            )
        if raw_spec is None:
            return None
        if not isinstance(raw_spec, Mapping):
            raise TypeError(f"abms_oracle_spec must be an object for {self.case_id}.")
        return ABMSOracleSpec.from_dict(raw_spec)


def load_full_bench_run_cases(
    bench_path: str | Path,
    *,
    max_cases: int | None = None,
) -> list[FullBenchRunCase]:
    """Load runnable case metadata from a Full MAG-Bench package."""

    package = load_full_benchmark_package(bench_path)
    root = Path(package.root_dir)
    selected = package.cases[:max_cases] if max_cases is not None else package.cases
    return [_load_run_case(root, case) for case in selected]


def _load_run_case(root: Path, case: FullBenchmarkCase) -> FullBenchRunCase:
    case_dir = root / case.case_dir
    case_yaml = _load_yaml(case_dir / "case.yaml")
    if isinstance(case_yaml, Mapping) and str(case_yaml["case_id"]) != case.case_id:
        raise ValueError(f"case.yaml id mismatch for {case.case_id}.")
    metadata_path = case_dir / case.metadata_path
    return FullBenchRunCase(
        case_id=case.case_id,
        case_dir=case_dir,
        source_task=_load_json(case_dir / "source_task.json"),
        objectified_task=_load_json(case_dir / "objectified_task.json"),
        mag_yaml_path=case_dir / case.mag_path,
        mag_json_path=case_dir / case.mag_json_path,
        dot_path=case_dir / case.dot_path,
        manifest_path=case_dir / case.manifest_path,
        tests_json_path=case_dir / case.tests_json_path,
        tests_py_path=case_dir / case.tests_py_path,
        gold_solution_path=case_dir / case.gold_solution_path,
        broken_solution_path=case_dir / case.broken_solution_path,
        abms_oracle_path=case_dir / case.abms_oracle_path,
        abms_tests_py_path=case_dir / case.abms_tests_py_path,
        gold_abms_solution_path=case_dir / case.gold_abms_solution_path,
        broken_abms_solution_path=case_dir / case.broken_abms_solution_path,
        framework_path=case_dir / case.framework_path,
        metadata_path=metadata_path,
        metadata=_load_json(metadata_path),
        package_case=case,
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return data


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)
