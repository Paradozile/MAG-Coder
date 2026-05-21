"""Mini MAG-Bench runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from magcoder.benchmark.package import BenchmarkPackage
from magcoder.benchmark.report import (
    benchmark_summary_row,
    write_benchmark_summary_reports,
)
from magcoder.benchmark.task import BenchmarkTask
from magcoder.evaluation import (
    ScenarioEvaluationRunner,
    ScenarioRunResult,
    save_scenario_run_result,
)

BenchmarkInputMode = Literal["structured", "graphical", "both"]


@dataclass(slots=True)
class BenchmarkRunResult:
    """Result bundle for a benchmark run."""

    benchmark_id: str
    results: list[ScenarioRunResult]
    summary_rows: list[dict[str, Any]]
    output_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Run all tasks in a benchmark package."""

    def run(
        self,
        package: BenchmarkPackage,
        *,
        input_mode: BenchmarkInputMode,
        output_dir: str | Path,
    ) -> BenchmarkRunResult:
        """Run benchmark tasks and write per-task plus summary outputs."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        runner = ScenarioEvaluationRunner()
        results: list[ScenarioRunResult] = []
        rows: list[dict[str, Any]] = []

        for task in package.tasks:
            for mode in _modes_for_task(task, input_mode):
                scenario = task.to_scenario_spec(
                    base_dir=Path(package.root_dir) / "tasks" / task.task_id,
                )
                result = runner.run(scenario, input_mode=mode)
                results.append(result)
                result_path = output_path / f"{task.task_id}_{mode}_result.json"
                save_scenario_run_result(result, result_path)
                rows.append(
                    benchmark_summary_row(
                        benchmark_id=package.benchmark_id,
                        task=task,
                        result=result,
                    )
                )

        write_benchmark_summary_reports(
            rows,
            csv_path=output_path / "summary.csv",
            markdown_path=output_path / "summary.md",
        )
        return BenchmarkRunResult(
            benchmark_id=package.benchmark_id,
            results=results,
            summary_rows=rows,
            output_dir=str(output_path),
            metadata={"input_mode": input_mode},
        )


def _modes_for_task(
    task: BenchmarkTask,
    requested_mode: BenchmarkInputMode,
) -> list[Literal["structured", "graphical"]]:
    if requested_mode == "both":
        modes: list[Literal["structured", "graphical"]] = []
        for mode in ("structured", "graphical"):
            if mode in task.supported_input_modes:
                modes.append(mode)
        return modes
    if requested_mode not in task.supported_input_modes:
        return []
    return [requested_mode]
