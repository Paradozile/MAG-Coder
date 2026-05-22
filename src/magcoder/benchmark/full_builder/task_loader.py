"""Load HumanEval/MBPP-style source tasks from local JSONL files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from magcoder.benchmark.full_builder.source_task import (
    SourceTask,
    source_task_from_humaneval,
    source_task_from_mbpp,
)


@dataclass(slots=True)
class SourceTaskLoadReport:
    """Machine-readable report for source-task loading."""

    source_files: list[str]
    loaded_tasks: int = 0
    skipped_rows: int = 0
    malformed_rows: int = 0
    duplicate_task_ids: int = 0
    duplicate_policy: str = "raise"
    skip_reasons: dict[str, int] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render this report as JSON-compatible containers."""

        return asdict(self)


def load_source_tasks_from_jsonl(path: str | Path) -> list[SourceTask]:
    """Load source tasks from one local JSONL file."""

    input_path = Path(path)
    tasks: list[SourceTask] = []
    with input_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            if not isinstance(data, Mapping):
                raise TypeError(
                    f"JSONL row {line_number} in {input_path} must be an object.",
                )
            tasks.append(_source_task_from_mapping(data))
    return tasks


def load_source_tasks_from_jsonl_with_report(
    path: str | Path,
) -> tuple[list[SourceTask], SourceTaskLoadReport]:
    """Load one JSONL source file while skipping malformed rows with a report."""

    input_path = Path(path)
    tasks: list[SourceTask] = []
    report = SourceTaskLoadReport(source_files=[str(input_path)])
    with input_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                if not isinstance(data, Mapping):
                    raise TypeError("row must be a JSON object")
                task = _source_task_from_mapping(data)
            except Exception as error:  # noqa: BLE001 - report and skip row.
                reason = f"{type(error).__name__}: {error}"
                _record_skip(report, input_path, line_number, reason)
                report.malformed_rows += 1
                continue
            tasks.append(task)
            report.rows.append(
                {
                    "path": str(input_path),
                    "line_number": line_number,
                    "task_id": task.task_id,
                    "status": "loaded",
                    "source": task.source,
                }
            )
    report.loaded_tasks = len(tasks)
    return tasks, report


def load_source_tasks(paths: Iterable[str | Path]) -> list[SourceTask]:
    """Load source tasks from multiple local JSONL files."""

    tasks: list[SourceTask] = []
    seen: set[str] = set()
    for path in paths:
        for task in load_source_tasks_from_jsonl(path):
            if task.task_id in seen:
                raise ValueError(f"Duplicate source task id: {task.task_id!r}.")
            seen.add(task.task_id)
            tasks.append(task)
    return tasks


def load_source_tasks_with_report(
    paths: Iterable[str | Path],
    *,
    duplicate_policy: str = "rename",
) -> tuple[list[SourceTask], SourceTaskLoadReport]:
    """Load multiple JSONL files and report malformed/duplicate rows.

    ``duplicate_policy`` may be ``rename``, ``skip``, or ``raise``.
    """

    if duplicate_policy not in {"rename", "skip", "raise"}:
        raise ValueError(f"Unsupported duplicate policy: {duplicate_policy!r}")
    path_list = [Path(path) for path in paths]
    merged = SourceTaskLoadReport(
        source_files=[str(path) for path in path_list],
        duplicate_policy=duplicate_policy,
    )
    tasks: list[SourceTask] = []
    seen: dict[str, int] = {}
    for path in path_list:
        loaded, report = load_source_tasks_from_jsonl_with_report(path)
        _merge_report(merged, report)
        for task in loaded:
            original_id = task.task_id
            if original_id in seen:
                merged.duplicate_task_ids += 1
                if duplicate_policy == "raise":
                    raise ValueError(f"Duplicate source task id: {original_id!r}.")
                if duplicate_policy == "skip":
                    reason = f"duplicate_task_id:{original_id}"
                    merged.skipped_rows += 1
                    merged.skip_reasons[reason] = (
                        merged.skip_reasons.get(reason, 0) + 1
                    )
                    merged.rows.append(
                        {
                            "path": str(path),
                            "task_id": original_id,
                            "status": "skipped",
                            "reason": reason,
                        }
                    )
                    continue
                seen[original_id] += 1
                task.task_id = f"{original_id}_{seen[original_id]}"
                task.metadata["original_task_id"] = original_id
                task.metadata["duplicate_rename_index"] = seen[original_id]
            else:
                seen[original_id] = 1
            tasks.append(task)
    merged.loaded_tasks = len(tasks)
    return tasks, merged


def _source_task_from_mapping(data: Mapping[str, Any]) -> SourceTask:
    source = str(data.get("source", "")).lower()
    if source == "humaneval" or "HumanEval" in str(data.get("task_id", "")):
        return source_task_from_humaneval(data)
    if source == "mbpp" or "test_list" in data:
        return source_task_from_mbpp(data)
    return SourceTask.from_dict(data)


def _record_skip(
    report: SourceTaskLoadReport,
    path: Path,
    line_number: int,
    reason: str,
) -> None:
    report.skipped_rows += 1
    report.skip_reasons[reason] = report.skip_reasons.get(reason, 0) + 1
    report.rows.append(
        {
            "path": str(path),
            "line_number": line_number,
            "status": "skipped",
            "reason": reason,
        }
    )


def _merge_report(target: SourceTaskLoadReport, source: SourceTaskLoadReport) -> None:
    target.skipped_rows += source.skipped_rows
    target.malformed_rows += source.malformed_rows
    target.duplicate_task_ids += source.duplicate_task_ids
    target.rows.extend(source.rows)
    for reason, count in source.skip_reasons.items():
        target.skip_reasons[reason] = target.skip_reasons.get(reason, 0) + count
