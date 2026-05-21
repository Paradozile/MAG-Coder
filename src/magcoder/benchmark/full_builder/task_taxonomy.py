"""Deterministic source-task taxonomy for MAG-Bench construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from magcoder.benchmark.full_builder.source_task import SourceTask

DataType = Literal["basic", "sequence", "mapping", "nested", "tree", "matrix"]
TaskType = Literal[
    "string_processing",
    "sorting_searching",
    "data_structure_operation",
    "numeric_computation",
    "transformation",
]


@dataclass(slots=True)
class TaskClassification:
    """A deterministic classification for one source task."""

    task_id: str
    data_type: DataType
    task_type: TaskType
    confidence: float
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this classification as JSON-compatible containers."""

        return asdict(self)


def classify_source_task(task: SourceTask) -> TaskClassification:
    """Classify a source task using deterministic keyword heuristics."""

    text = " ".join(
        [
            task.prompt,
            task.entry_point or "",
            task.input_signature or "",
            task.output_type or "",
            " ".join(task.tests),
        ]
    ).lower()
    data_type, data_reasons = _classify_data_type(text)
    task_type, task_reasons = _classify_task_type(text)
    confidence = 0.55 + 0.1 * min(len(data_reasons) + len(task_reasons), 4)
    return TaskClassification(
        task_id=task.task_id,
        data_type=data_type,
        task_type=task_type,
        confidence=min(confidence, 0.95),
        reasons=[*data_reasons, *task_reasons],
        metadata={"heuristic_version": 1},
    )


def _classify_data_type(text: str) -> tuple[DataType, list[str]]:
    if any(keyword in text for keyword in ["matrix", "grid", "2d", "rows"]):
        return "matrix", ["mentions matrix/grid/rows"]
    if any(keyword in text for keyword in ["tree", "node", "binary"]):
        return "tree", ["mentions tree/node"]
    if any(keyword in text for keyword in ["nested", "list of lists"]):
        return "nested", ["mentions nested data"]
    if any(keyword in text for keyword in ["dict", "map", "key"]):
        return "mapping", ["mentions mapping/key"]
    if any(keyword in text for keyword in ["list", "array", "sequence", "tuple"]):
        return "sequence", ["mentions list/array/sequence"]
    return "basic", ["default basic data"]


def _classify_task_type(text: str) -> tuple[TaskType, list[str]]:
    if any(keyword in text for keyword in ["string", "substring", "vowel", "word"]):
        return "string_processing", ["mentions string/text operation"]
    if any(keyword in text for keyword in ["sort", "search", "find", "index"]):
        return "sorting_searching", ["mentions sorting/searching"]
    if any(keyword in text for keyword in ["insert", "delete", "merge", "count"]):
        return "data_structure_operation", ["mentions data-structure operation"]
    if any(keyword in text for keyword in ["sum", "product", "prime", "number"]):
        return "numeric_computation", ["mentions numeric computation"]
    if any(keyword in text for keyword in ["reverse", "convert", "transform"]):
        return "transformation", ["mentions transformation"]
    return "transformation", ["default transformation task"]
