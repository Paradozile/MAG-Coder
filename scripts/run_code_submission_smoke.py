"""Run a lightweight code-submission smoke without real providers or downloads."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import magcoder
from magcoder.benchmark.full_builder import (
    build_full_mag_bench,
    validate_full_mag_bench,
)
from magcoder.benchmark.full_runner import (
    FullBenchPassKConfig,
    FullBenchSampleResult,
    FullMagBenchPassKRunner,
    build_full_bench_summary_rows,
)
from magcoder.providers.provider_registry import registry_report
from scripts.extract_mag_from_rendered_graph import extract_mag_from_rendered_graph
from scripts.render_mag_graphics import render_case_graphics


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_code_submission_smoke(args.output)
    print(f"Code submission smoke output: {args.output}")
    print(f"Passed: {report['passed']}")


def run_code_submission_smoke(output_dir: str | Path) -> dict[str, Any]:
    """Run tiny build/validation/fake-provider/provider/VLM smoke."""

    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    tiny_bench = output / "tiny_bench"
    package, build_report = build_full_mag_bench(
        source_paths=[Path("examples/source_tasks/humaneval_style_seed.jsonl")],
        output_dir=tiny_bench,
        max_cases=2,
        formats=("structured", "graphical"),
        overwrite=True,
        split="auto",
        seed=2026,
        skip_unsupported=True,
        write_source_loader_report=True,
    )
    validation = validate_full_mag_bench(tiny_bench)
    fake_output = output / "fake_passk"
    fake_result = FullMagBenchPassKRunner().run(
        FullBenchPassKConfig(
            bench_path=str(tiny_bench),
            output_dir=str(fake_output),
            provider="fake",
            model_name="fake-generator",
            samples_per_case=1,
            k_values=[1, 3, 10],
            modes=["structured_graph_nl"],
            methods=["full_with_refinement", "without_ast"],
            max_cases=1,
            overwrite=True,
            evaluation="all",
            refinement_strategy="balanced",
        )
    )
    passk_rows = build_full_bench_summary_rows(
        [_sample(0, True), _sample(1, False)],
        FullBenchPassKConfig(
            bench_path=str(tiny_bench),
            output_dir=str(output / "passk_fixture"),
            samples_per_case=2,
            k_values=[1, 3, 10],
        ),
    )
    first_case = package.cases[0].case_id
    graphics_dir = output / "graphics"
    manifest = render_case_graphics(
        bench=tiny_bench,
        case_id=first_case,
        output_dir=graphics_dir,
    )
    wrapper = json.loads(manifest.read_text(encoding="utf-8"))
    recovered = extract_mag_from_rendered_graph(
        image_path=graphics_dir / wrapper["image_path"],
        manifest_path=manifest,
        extractor_name="mock",
        output_path=graphics_dir / "recovered_mag.json",
    )
    providers = registry_report()
    report = {
        "schema_version": 1,
        "passed": bool(
            validation.passed
            and fake_result.sample_results
            and providers["external_calls_made"] is False
            and Path(str(recovered["output"])).exists()
        ),
        "magcoder_import": getattr(magcoder, "__name__", "magcoder"),
        "tiny_benchmark": {
            "path": str(tiny_bench),
            "case_count": build_report.case_count,
            "validation_passed": validation.passed,
        },
        "fake_provider_smoke": {
            "path": str(fake_output),
            "sample_count": len(fake_result.sample_results),
            "real_provider_invoked": False,
        },
        "passk_fixture": {
            "rows": passk_rows,
            "k_greater_than_n": [
                row for row in passk_rows if row["k"] in {3, 10}
            ],
        },
        "vlm_mock": {
            "manifest": str(manifest),
            "recovered_mag": recovered["output"],
            "real_vlm_invoked": False,
        },
        "provider_registry": {
            "families": providers["provider_families"],
            "external_calls_made": providers["external_calls_made"],
        },
    }
    _write_json(report, output / "smoke_report.json")
    (output / "smoke_report.md").write_text(_render_report(report), encoding="utf-8")
    return report


def _sample(index: int, success: bool) -> FullBenchSampleResult:
    return FullBenchSampleResult(
        case_id="fixture_case",
        input_mode="structured_graph_nl",
        method="full_with_refinement",
        sample_index=index,
        seed=1000 + index,
        generation_result_path="generation.json",
        code_initial_path="code_initial.py",
        initial_report_path="initial.json",
        evaluation_path="evaluation.json",
        parse_success=True,
        framework_passed=success,
        mandatory_success=success,
        checkpoint_accuracy=1.0 if success else 0.0,
        success=success,
        cache_hit=False,
        contract_pass=success,
        functional_pass=success,
        abms_pass=success,
        metadata={
            "split": "dev",
            "source": "fixture",
            "task_type": "smoke",
            "data_type": "basic",
            "template_family": "smoke",
            "mode_family": "structured",
            "nl_condition": "graph_nl",
            "baseline_family": "mag_coder_full",
            "refinement_strategy": "balanced",
        },
    )


def _render_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Code Submission Smoke",
            "",
            f"- Passed: {report['passed']}",
            f"- Tiny benchmark cases: {report['tiny_benchmark']['case_count']}",
            f"- Fake samples: {report['fake_provider_smoke']['sample_count']}",
            "- Real provider invoked: False",
            "- Real VLM invoked: False",
        ]
    ) + "\n"


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
