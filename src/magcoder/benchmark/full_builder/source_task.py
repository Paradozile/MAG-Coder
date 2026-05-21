"""Source task schema for Full MAG-Bench construction."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SourceTask:
    """A normalized HumanEval/MBPP-style source task."""

    task_id: str
    source: str
    prompt: str
    tests: list[str]
    canonical_solution: str | None = None
    entry_point: str | None = None
    input_signature: str | None = None
    output_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render the task as JSON-serializable containers."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SourceTask:
        """Build a normalized source task from a parsed mapping."""

        task_id = data.get("task_id") or data.get("id")
        if task_id is None:
            raise ValueError("Source task requires 'task_id' or 'id'.")

        prompt = data.get("prompt") or data.get("text")
        if prompt is None:
            raise ValueError(f"Source task {task_id!r} requires prompt/text.")

        tests = _tests_from_value(
            data.get("tests", data.get("test", data.get("test_list", []))),
        )
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Source task metadata must be an object.")

        return cls(
            task_id=str(task_id),
            source=str(data.get("source", "unknown")),
            prompt=str(prompt),
            canonical_solution=_optional_str(
                data.get("canonical_solution", data.get("code")),
            ),
            tests=tests,
            entry_point=_optional_str(data.get("entry_point"))
            or _infer_entry_point(
                tests,
                _optional_str(data.get("canonical_solution", data.get("code"))),
            ),
            input_signature=_optional_str(data.get("input_signature")),
            output_type=_optional_str(data.get("output_type")),
            metadata=dict(metadata),
        )


def source_task_from_humaneval(data: Mapping[str, Any]) -> SourceTask:
    """Normalize a HumanEval-style task mapping."""

    enriched = {
        **data,
        "source": data.get("source", "humaneval"),
        "tests": data.get("tests", data.get("test", [])),
    }
    return SourceTask.from_dict(enriched)


def source_task_from_mbpp(data: Mapping[str, Any]) -> SourceTask:
    """Normalize an MBPP-style task mapping."""

    task_id = data.get("task_id", data.get("id"))
    enriched = {
        **data,
        "task_id": f"mbpp_{task_id}" if isinstance(task_id, int) else task_id,
        "source": data.get("source", "mbpp"),
        "prompt": data.get("prompt", data.get("text")),
        "canonical_solution": data.get("canonical_solution", data.get("code")),
        "tests": data.get("tests", data.get("test_list", [])),
    }
    return SourceTask.from_dict(enriched)


def _tests_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line for line in value.splitlines() if line.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError("Source task tests must be a string or list.")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _infer_entry_point(tests: list[str], code: str | None) -> str | None:
    """Infer a source entry point from assertion tests or function code."""

    if code:
        match = re.search(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code, re.M)
        if match:
            return match.group(1)
    for test in tests:
        match = re.search(r"\bassert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", test)
        if match:
            return match.group(1)
    return None
