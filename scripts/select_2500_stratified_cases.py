"""Select a deterministic stratified subset from the paper2500 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_CASE_FILES = [
    "source_task.json",
    "objectified_task.json",
    "variant_spec.json",
    "mag.yaml",
    "mag.json",
    "mag.dot",
    "mag_manifest.json",
    "tests.py",
    "tests.json",
    "gold_solution.py",
    "broken_solution.py",
    "abms_oracle.json",
    "abms_tests.py",
    "gold_abms_solution.py",
    "broken_abms_solution.py",
    "framework.yaml",
    "metadata.json",
]
STRATA_KEYS = ["source", "task_type", "data_type", "template_family"]


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Select deterministic stratified paper2500 benchmark cases."
    )
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = select_stratified_cases(
        bench_path=args.bench,
        count=args.count,
        seed=args.seed,
        output_path=args.output,
    )
    print(f"Selected cases: {report['selected_count']}")
    print(f"Selection JSON: {args.output}")
    print(f"Selection report: {report['selection_report_path']}")


def select_stratified_cases(
    *,
    bench_path: str | Path,
    count: int,
    seed: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Select and save a deterministic stratified subset."""

    if count <= 0:
        raise ValueError("count must be positive.")
    bench = Path(bench_path)
    output = Path(output_path)
    cases = _load_case_index(bench)
    available, skipped = _filter_available_cases(bench, cases)
    selected = _greedy_stratified_select(available, count=count, seed=seed)
    distributions = {
        key: dict(Counter(str(case.get(key, "unknown")) for case in selected))
        for key in STRATA_KEYS
    }
    warnings = _coverage_warnings(selected, count, available)
    payload = {
        "selection_id": output.stem,
        "selection_strategy": "deterministic_greedy_stratified",
        "benchmark_root": str(bench),
        "count_requested": count,
        "seed": seed,
        "available_case_count": len(available),
        "selected_count": len(selected),
        "case_ids": [str(case["case_id"]) for case in selected],
        "selected_cases": selected,
        "cases": selected,
        "distributions": distributions,
        "skipped_unavailable_cases": skipped,
        "warnings": warnings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(payload, output)
    report_path = output.with_suffix(".md")
    _write_selection_report(payload, report_path)
    payload["selection_report_path"] = str(report_path)
    _write_json(payload, output)
    return payload


def _load_case_index(bench: Path) -> list[dict[str, Any]]:
    path = bench / "case_index.json"
    if not path.exists():
        raise FileNotFoundError(f"case_index.json not found: {path}")
    data = _read_json(path)
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise TypeError("case_index.json field 'cases' must be a list.")
    normalized: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        normalized.append(
            {
                "case_id": str(case.get("case_id", "")),
                "source": str(case.get("source", "unknown")),
                "split": str(case.get("split", "none")),
                "task_type": str(case.get("task_type", "unknown")),
                "data_type": str(case.get("data_type", "unknown")),
                "template_family": str(case.get("template_family", "single")),
                "source_task_id": str(case.get("source_task_id", "")),
                "variant_id": str(case.get("variant_id", "")),
                "case_dir": str(case.get("case_dir", "")),
            }
        )
    return [case for case in normalized if case["case_id"] and case["case_dir"]]


def _filter_available_cases(
    bench: Path,
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    available: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for case in cases:
        case_dir = bench / str(case["case_dir"])
        missing = [
            name for name in REQUIRED_CASE_FILES if not (case_dir / name).exists()
        ]
        if missing:
            skipped.append(
                {
                    "case_id": str(case["case_id"]),
                    "reason": "missing_artifacts",
                    "missing": ",".join(missing),
                }
            )
            continue
        available.append(case)
    return available, skipped


def _greedy_stratified_select(
    cases: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count > len(cases):
        raise ValueError(f"requested {count} cases but only {len(cases)} available")
    targets = {
        key: _quota(Counter(str(case[key]) for case in cases), count)
        for key in STRATA_KEYS
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    current = {key: Counter() for key in STRATA_KEYS}

    def add_case(case: dict[str, Any]) -> None:
        selected.append(case)
        selected_ids.add(str(case["case_id"]))
        for key in STRATA_KEYS:
            current[key][str(case[key])] += 1

    # Seed the set with one representative for every stratum value when possible.
    for key in STRATA_KEYS:
        for value in sorted(targets[key]):
            if len(selected) >= count:
                break
            candidates = [
                case
                for case in cases
                if str(case["case_id"]) not in selected_ids
                and str(case[key]) == value
            ]
            if candidates:
                add_case(_best_candidate(candidates, current, targets, seed))

    while len(selected) < count:
        candidates = [
            case for case in cases if str(case["case_id"]) not in selected_ids
        ]
        add_case(_best_candidate(candidates, current, targets, seed))
    return selected


def _best_candidate(
    candidates: list[dict[str, Any]],
    current: dict[str, Counter[str]],
    targets: dict[str, dict[str, int]],
    seed: int,
) -> dict[str, Any]:
    def sort_key(case: dict[str, Any]) -> tuple[float, str]:
        score = 0.0
        for key in STRATA_KEYS:
            value = str(case[key])
            target = max(1, targets[key].get(value, 1))
            deficit = max(0, target - current[key][value])
            score += deficit / target
        return (-score, _stable_hash(seed, str(case["case_id"])))

    return sorted(candidates, key=sort_key)[0]


def _quota(counts: Counter[str], total: int) -> dict[str, int]:
    if not counts:
        return {}
    denominator = sum(counts.values())
    raw = {
        key: (value * total / denominator)
        for key, value in sorted(counts.items())
    }
    quotas = {key: int(value) for key, value in raw.items()}
    for key in quotas:
        quotas[key] = max(1, quotas[key])
    while sum(quotas.values()) > total:
        reducible = [
            key for key, value in quotas.items() if value > 1
        ]
        if not reducible:
            break
        key = min(reducible, key=lambda item: (raw[item] - quotas[item], item))
        quotas[key] -= 1
    while sum(quotas.values()) < total:
        key = max(
            quotas,
            key=lambda item: (raw[item] - quotas[item], item),
        )
        quotas[key] += 1
    return quotas


def _coverage_warnings(
    selected: list[dict[str, Any]],
    count: int,
    available: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    if len(selected) != count:
        warnings.append(f"selected {len(selected)} cases, expected {count}")
    for key in STRATA_KEYS:
        selected_values = {str(case[key]) for case in selected}
        available_values = {str(case[key]) for case in available}
        if len(selected_values) < min(len(available_values), 2):
            warnings.append(f"low {key} coverage: {sorted(selected_values)}")
    return warnings


def _write_selection_report(payload: dict[str, Any], path: Path) -> None:
    distributions = payload["distributions"]
    lines = [
        "# Paper2500 Stratified Case Selection",
        "",
        f"- Strategy: `{payload['selection_strategy']}`",
        f"- Seed: {payload['seed']}",
        f"- Requested count: {payload['count_requested']}",
        f"- Selected count: {payload['selected_count']}",
        f"- Available cases: {payload['available_case_count']}",
        f"- Skipped unavailable cases: {len(payload['skipped_unavailable_cases'])}",
        "",
        "## Distributions",
        "",
        f"- Source: `{distributions['source']}`",
        f"- Task type: `{distributions['task_type']}`",
        f"- Data type: `{distributions['data_type']}`",
        f"- Template family: `{distributions['template_family']}`",
        "",
        "## Warnings",
        "",
        *[f"- {warning}" for warning in payload["warnings"]],
        "",
        "## Selected Cases",
        "",
        *[
            (
                f"- `{case['case_id']}` source={case['source']} "
                f"task_type={case['task_type']} data_type={case['data_type']} "
                f"template_family={case['template_family']}"
            )
            for case in payload["selected_cases"]
        ],
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _stable_hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return data


def _write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    main()
