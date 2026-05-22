"""Build Full MAG-Bench v1 packages from local source-task JSONL files."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from magcoder.benchmark.full_builder.mag_synthesizer import (
    synthesize_mag_from_objectified_task,
)
from magcoder.benchmark.full_builder.manifest import (
    FullBenchmarkCase,
    FullBenchmarkPackage,
)
from magcoder.benchmark.full_builder.objectifier import objectify_source_task
from magcoder.benchmark.full_builder.source_task import SourceTask
from magcoder.benchmark.full_builder.task_loader import load_source_tasks_with_report
from magcoder.benchmark.full_builder.task_taxonomy import classify_source_task
from magcoder.benchmark.full_builder.template_variants import (
    TemplateVariantSpec,
    VariantGenerationConfig,
    generate_template_variant_specs,
    summarize_variant_specs,
)
from magcoder.benchmark.full_builder.test_synthesizer import (
    save_abms_oracle_json,
    save_abms_tests_py,
    save_broken_abms_solution_py,
    save_broken_solution_py,
    save_gold_abms_solution_py,
    save_gold_solution_py,
    save_tests_json,
    save_tests_py,
    synthesize_tests,
)
from magcoder.benchmark.full_builder.variant_filters import (
    VariantFilterState,
    build_variant_filter_report,
    evaluate_variant,
)
from magcoder.graphics.artifact import render_mag_artifact
from magcoder.models import save_mag_to_json, save_mag_to_yaml

DEFAULT_FRAMEWORK_PATH = Path("examples/simple_abms_framework.yaml")


@dataclass(slots=True)
class FullBenchmarkBuildReport:
    """Build output summary for Full MAG-Bench v1."""

    benchmark_id: str
    output_dir: str
    source_files: list[str]
    case_count: int
    case_ids: list[str]
    total_source_tasks: int = 0
    accepted_cases: int = 0
    skipped_cases: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    task_type_distribution: dict[str, int] = field(default_factory=dict)
    data_type_distribution: dict[str, int] = field(default_factory=dict)
    artifact_counts: dict[str, int] = field(default_factory=dict)
    graphviz_available: bool | None = None
    official_datasets_downloaded: list[str] = field(default_factory=list)
    source_urls: dict[str, str] = field(default_factory=dict)
    raw_file_sha256: dict[str, str] = field(default_factory=dict)
    normalized_file_sha256: dict[str, str] = field(default_factory=dict)
    source_task_distribution: dict[str, int] = field(default_factory=dict)
    raw_humaneval_count: int = 0
    raw_mbpp_count: int = 0
    duplicate_task_ids: int = 0
    parse_failures: int = 0
    unsupported_taxonomy_cases: int = 0
    unsupported_objectification_cases: int = 0
    unsupported_oracle_cases: int = 0
    unsupported_abms_oracle_cases: int = 0
    target_cases: int | None = None
    target_reached: bool = False
    candidate_variants_generated: int = 0
    variants_generated: int = 0
    variants_accepted: int = 0
    skipped_variants: int = 0
    duplicate_variants_removed: int = 0
    illogical_variants_removed: int = 0
    template_family_distribution: dict[str, int] = field(default_factory=dict)
    source_distribution: dict[str, int] = field(default_factory=dict)
    split_distribution: dict[str, int] = field(default_factory=dict)
    graphical_artifact_count: int = 0
    abms_oracle_supported_count: int = 0
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this report as JSON-compatible containers."""

        return asdict(self)


def build_full_mag_bench(
    *,
    source_paths: Sequence[str | Path],
    output_dir: str | Path,
    max_cases: int | None = None,
    formats: Sequence[str] = ("structured", "graphical"),
    overwrite: bool = False,
    framework_path: str | Path = DEFAULT_FRAMEWORK_PATH,
    split: str = "none",
    case_id_prefix: str = "",
    seed: int | None = None,
    skip_unsupported: bool = True,
    write_source_loader_report: bool = False,
    variants_per_task: int = 1,
    max_variants: int | None = None,
    target_cases: int | None = None,
    variant_policy: str = "single",
    deduplicate_variants: bool = False,
    drop_illogical: bool = False,
) -> tuple[FullBenchmarkPackage, FullBenchmarkBuildReport]:
    """Build a reproducible Full MAG-Bench v1 package from seed source tasks."""

    root = Path(output_dir)
    if root.exists() and overwrite:
        preserved_artifacts = _preserve_existing_artifact_dirs(
            root,
            ("passk_runs", "frozen_results"),
        )
        shutil.rmtree(root)
        if preserved_artifacts:
            root.mkdir(parents=True, exist_ok=True)
            for name, preserved_path in preserved_artifacts.items():
                shutil.move(str(preserved_path), root / name)
    root.mkdir(parents=True, exist_ok=True)
    normalized_formats = _normalize_formats(formats)
    source_tasks, loader_report = load_source_tasks_with_report(
        source_paths,
        duplicate_policy="rename",
    )
    if loader_report.skipped_rows and not skip_unsupported:
        raise ValueError(
            f"Source loader skipped {loader_report.skipped_rows} rows; "
            "rerun with skip_unsupported=True to build accepted cases only."
        )
    selected_tasks = (
        source_tasks
        if target_cases is not None
        else source_tasks[:max_cases]
        if max_cases is not None
        else source_tasks
    )
    cases: list[FullBenchmarkCase] = []
    warnings: list[str] = []
    skipped: list[dict[str, str]] = []
    used_case_ids: set[str] = set()
    filter_rows: list[dict[str, Any]] = []
    filter_state = VariantFilterState()
    effective_deduplicate = deduplicate_variants or variant_policy == "paper2500"
    effective_drop_illogical = drop_illogical or variant_policy == "paper2500"
    variant_config = VariantGenerationConfig(
        variants_per_task=variants_per_task,
        variant_policy=variant_policy,
        target_cases=target_cases,
        seed=seed,
        deduplicate_variants=effective_deduplicate,
        drop_illogical=effective_drop_illogical,
    )
    variant_inputs: list[tuple[SourceTask, Any, TemplateVariantSpec, int]] = []
    for source_task in selected_tasks:
        try:
            classification = classify_source_task(source_task)
        except Exception as error:  # noqa: BLE001 - report and skip bad tasks.
            skipped.append(
                {
                    "task_id": source_task.task_id,
                    "variant_index": "",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            if not skip_unsupported:
                raise
            continue
        specs = generate_template_variant_specs(
            source_task,
            classification,
            variant_config,
        )
        for variant_index, variant_spec in enumerate(specs):
            variant_inputs.append(
                (source_task, classification, variant_spec, variant_index)
            )
    if variant_policy == "paper2500":
        variant_inputs.sort(key=lambda item: item[3])
    if max_variants is not None:
        variant_inputs = variant_inputs[:max_variants]
    variant_generation_report = summarize_variant_specs(
        [item[2] for item in variant_inputs],
        source_task_count=len(selected_tasks),
        config=variant_config,
    )
    variants_generated = len(variant_inputs)
    for source_task, classification, variant_spec, variant_index in variant_inputs:
        try:
            if target_cases is not None and len(cases) >= target_cases:
                _record_variant_skip(
                    skipped,
                    filter_rows,
                    source_task=source_task,
                    variant_spec=variant_spec,
                    variant_index=variant_index,
                    reason="target_reached",
                )
                continue
            try:
                objectified = objectify_source_task(
                    source_task,
                    classification,
                    variant_index=variant_index,
                    variant_spec=variant_spec,
                )
                _validate_source_task_for_oracle(source_task)
                decision = evaluate_variant(
                    source_task=source_task,
                    classification=classification,
                    objectified=objectified,
                    variant_spec=variant_spec,
                    state=filter_state,
                    deduplicate=effective_deduplicate,
                    drop_illogical=effective_drop_illogical,
                )
                filter_rows.append(
                    {
                        "accepted": decision.accepted,
                        "source_task_id": source_task.task_id,
                        "variant_id": variant_spec.variant_id,
                        "variant_index": variant_index,
                        "reason": decision.reason,
                        "template_family": variant_spec.template_family,
                        "task_type": classification.task_type,
                        "data_type": classification.data_type,
                        "metadata": decision.metadata,
                    }
                )
                if not decision.accepted:
                    skipped.append(
                        {
                            "task_id": source_task.task_id,
                            "variant_index": str(variant_index),
                            "reason": decision.reason,
                        }
                    )
                    continue
                if case_id_prefix:
                    objectified.task_id = (
                        f"{case_id_prefix}_{objectified.task_id}"
                    )
                case_id = _unique_case_id(objectified.task_id, used_case_ids)
                used_case_ids.add(case_id)
                if case_id != objectified.task_id:
                    objectified.task_id = case_id
                split_name = _assign_split(len(cases), split)
                case = _write_case(
                    root=root,
                    source_task=source_task,
                    classification=classification,
                    objectified=objectified,
                    framework_path=Path(framework_path),
                    formats=normalized_formats,
                    warnings=warnings,
                    split=split_name,
                    variant_spec=variant_spec,
                )
                cases.append(case)
            except Exception as error:  # noqa: BLE001 - report and skip bad tasks.
                skipped.append(
                    {
                        "task_id": source_task.task_id,
                        "variant_index": str(variant_index),
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                filter_rows.append(
                    {
                        "accepted": False,
                        "source_task_id": source_task.task_id,
                        "variant_id": variant_spec.variant_id,
                        "variant_index": variant_index,
                        "reason": f"{type(error).__name__}: {error}",
                        "template_family": variant_spec.template_family,
                        "task_type": classification.task_type,
                        "data_type": classification.data_type,
                    }
                )
                if not skip_unsupported:
                    raise
        except Exception:
            if not skip_unsupported:
                raise

    variant_filter_report = build_variant_filter_report(filter_rows)
    package = FullBenchmarkPackage(
        benchmark_id=(
            "full_mag_bench_2500_v1"
            if variant_policy == "paper2500" or target_cases is not None
            else "full_mag_bench_v1"
        ),
        name=(
            "Full MAG-Bench 2500 Template Reproduction"
            if variant_policy == "paper2500" or target_cases is not None
            else "Full MAG-Bench v1 Seed Package"
        ),
        version="0.1.0",
        root_dir=".",
        cases=cases,
        source_files=[str(path) for path in source_paths],
        metadata={
            "builder": "full_mag_bench_v1",
            "source_task_count": len(source_tasks),
            "selected_task_count": len(selected_tasks),
            "skipped_cases": skipped,
            "formats": normalized_formats,
            "split": split,
            "seed": seed,
            "case_id_prefix": case_id_prefix,
            "variants_per_task": variants_per_task,
            "max_variants": max_variants,
            "target_cases": target_cases,
            "variant_policy": variant_policy,
            "deduplicate_variants": effective_deduplicate,
            "drop_illogical": effective_drop_illogical,
            "notes": (
                "Full MAG-Bench package generated from local normalized "
                "HumanEval/MBPP-style SourceTask JSONL files."
            ),
        },
    )
    _save_package(package, root / "benchmark.yaml")
    _write_readme(root, package)
    _copy_official_reports(root, source_paths)
    report = FullBenchmarkBuildReport(
        benchmark_id=package.benchmark_id,
        output_dir=str(root),
        source_files=[str(path) for path in source_paths],
        case_count=len(cases),
        case_ids=[case.case_id for case in cases],
        total_source_tasks=len(source_tasks),
        accepted_cases=len(cases),
        skipped_cases=len(skipped),
        skip_reasons=_skip_reason_counts(skipped),
        task_type_distribution=dict(Counter(case.task_type for case in cases)),
        data_type_distribution=dict(Counter(case.data_type for case in cases)),
        source_distribution=dict(Counter(case.source for case in cases)),
        split_distribution=dict(
            Counter(str(case.metadata.get("split", "none")) for case in cases)
        ),
        artifact_counts={
            "structured_mag_files": len(cases) * 2,
            "graphical_dot_files": len(cases),
            "graphical_manifest_files": len(cases),
            "oracle_files": len(cases) * 4,
            "abms_oracle_files": len(cases) * 4,
        },
        graphviz_available=_graphviz_available(warnings),
        official_datasets_downloaded=_official_datasets(root),
        source_urls=_official_source_urls(root),
        raw_file_sha256=_official_sha256(root, "raw_file_sha256"),
        normalized_file_sha256=_normalized_sha256(source_paths),
        source_task_distribution=dict(Counter(task.source for task in source_tasks)),
        raw_humaneval_count=sum(
            1 for task in source_tasks if task.source == "humaneval"
        ),
        raw_mbpp_count=sum(1 for task in source_tasks if task.source == "mbpp"),
        duplicate_task_ids=loader_report.duplicate_task_ids,
        parse_failures=loader_report.malformed_rows,
        unsupported_oracle_cases=_reason_prefix_count(skipped, "ValueError"),
        unsupported_abms_oracle_cases=(
            variant_filter_report.unsupported_abms_oracle_variants
        ),
        target_cases=target_cases,
        target_reached=target_cases is not None and len(cases) >= target_cases,
        candidate_variants_generated=variants_generated,
        variants_generated=variants_generated,
        variants_accepted=len(cases),
        skipped_variants=variant_filter_report.skipped_variants,
        duplicate_variants_removed=variant_filter_report.duplicate_variants_removed,
        illogical_variants_removed=variant_filter_report.illogical_variants_removed,
        template_family_distribution=dict(
            Counter(
                str(case.metadata.get("template_family", "single"))
                for case in cases
            )
        ),
        graphical_artifact_count=len(cases) * 2,
        abms_oracle_supported_count=len(cases),
        warnings=warnings,
        metadata={
            "formats": normalized_formats,
            "skipped_cases": skipped,
            "source_loader_report": loader_report.to_dict(),
            "split": split,
            "seed": seed,
            "skip_unsupported": skip_unsupported,
            "deduplicate_variants": effective_deduplicate,
            "drop_illogical": effective_drop_illogical,
            "variant_generation_report": variant_generation_report.to_dict(),
            "variant_filter_report": variant_filter_report.to_dict(),
        },
    )
    _write_json(loader_report.to_dict(), root / "source_loader_report.json")
    _write_json(
        variant_generation_report.to_dict(),
        root / "variant_generation_report.json",
    )
    _write_json(variant_filter_report.to_dict(), root / "variant_filter_report.json")
    _write_distribution_files(root, cases, report)
    _write_paper2500_notes(root, report, variant_policy)
    save_build_report(report, root)
    return package, report


def load_full_benchmark_package(path: str | Path) -> FullBenchmarkPackage:
    """Load a Full MAG-Bench package from a directory or benchmark YAML."""

    input_path = Path(path)
    package_path = input_path / "benchmark.yaml" if input_path.is_dir() else input_path
    with package_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise TypeError("Full benchmark YAML root must be an object.")
    package = FullBenchmarkPackage.from_dict(data)
    package.root_dir = str(package_path.parent)
    return package


def save_build_report(report: FullBenchmarkBuildReport, root: Path) -> None:
    """Save build reports in JSON and Markdown."""

    json_path = root / "build_report.json"
    md_path = root / "build_report.md"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")
    lines = [
        "# Full MAG-Bench v1 Build Report",
        "",
        f"- Benchmark: `{report.benchmark_id}`",
        f"- Output: `{report.output_dir}`",
        f"- Cases: {report.case_count}",
        f"- Total source tasks: {report.total_source_tasks}",
        f"- Accepted cases: {report.accepted_cases}",
        f"- Skipped cases: {report.skipped_cases}",
        f"- Target cases: {report.target_cases if report.target_cases else 'n/a'}",
        f"- Target reached: {report.target_reached}",
        f"- Candidate variants generated: {report.candidate_variants_generated}",
        f"- Variants generated: {report.variants_generated}",
        f"- Variants accepted: {report.variants_accepted}",
        f"- Skipped variants: {report.skipped_variants}",
        f"- Duplicate variants removed: {report.duplicate_variants_removed}",
        f"- Illogical variants removed: {report.illogical_variants_removed}",
        f"- Source files: {', '.join(f'`{path}`' for path in report.source_files)}",
        "- Official datasets: "
        f"{', '.join(report.official_datasets_downloaded) or 'n/a'}",
        f"- Raw HumanEval source tasks: {report.raw_humaneval_count}",
        f"- Raw MBPP source tasks: {report.raw_mbpp_count}",
        f"- Graphviz available: {report.graphviz_available}",
        "",
        "## Cases",
        "",
        *[f"- `{case_id}`" for case_id in report.case_ids],
        "",
        "## Distributions",
        "",
        f"- Task types: `{report.task_type_distribution}`",
        f"- Data types: `{report.data_type_distribution}`",
        f"- Source tasks: `{report.source_task_distribution}`",
        f"- Sources: `{report.source_distribution}`",
        f"- Splits: `{report.split_distribution}`",
        f"- Template families: `{report.template_family_distribution}`",
        f"- Artifacts: `{report.artifact_counts}`",
        f"- ABMS oracle supported: {report.abms_oracle_supported_count}",
        "",
    ]
    if report.skip_reasons:
        lines.extend(
            [
                "## Skips",
                "",
                *[
                    f"- {reason}: {count}"
                    for reason, count in sorted(report.skip_reasons.items())
                ],
                "",
            ]
        )
    if report.warnings:
        lines.extend(
            ["## Warnings", "", *[f"- {warning}" for warning in report.warnings]]
        )
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_case(
    *,
    root: Path,
    source_task: SourceTask,
    classification: Any,
    objectified: Any,
    framework_path: Path,
    formats: list[str],
    warnings: list[str],
    split: str,
    variant_spec: TemplateVariantSpec | None = None,
) -> FullBenchmarkCase:
    case_dir = root / "cases" / objectified.task_id
    case_dir.mkdir(parents=True, exist_ok=True)
    mag = synthesize_mag_from_objectified_task(objectified)

    source_task_path = case_dir / "source_task.json"
    classification_path = case_dir / "classification.json"
    objectified_path = case_dir / "objectified_task.json"
    variant_spec_path = case_dir / "variant_spec.json"
    mag_yaml_path = case_dir / "mag.yaml"
    mag_json_path = case_dir / "mag.json"
    framework_target = case_dir / "framework.yaml"
    tests_json_path = case_dir / "tests.json"
    tests_py_path = case_dir / "tests.py"
    gold_solution_path = case_dir / "gold_solution.py"
    broken_solution_path = case_dir / "broken_solution.py"
    abms_oracle_path = case_dir / "abms_oracle.json"
    abms_tests_py_path = case_dir / "abms_tests.py"
    gold_abms_solution_path = case_dir / "gold_abms_solution.py"
    broken_abms_solution_path = case_dir / "broken_abms_solution.py"
    metadata_path = case_dir / "metadata.json"
    case_yaml_path = case_dir / "case.yaml"

    _write_json(source_task.to_dict(), source_task_path)
    _write_json(classification.to_dict(), classification_path)
    if variant_spec is not None:
        _write_json(variant_spec.to_dict(), variant_spec_path)
    else:
        _write_json({}, variant_spec_path)
    save_mag_to_yaml(mag, mag_yaml_path)
    save_mag_to_json(mag, mag_json_path)
    shutil.copy2(framework_path, framework_target)

    graphical_artifact = render_mag_artifact(
        mag_yaml_path,
        case_dir,
        formats=_graphical_formats(formats),
    )
    warnings.extend(
        str(warning)
        for warning in graphical_artifact.metadata.get("warnings", [])
    )
    tests = synthesize_tests(
        source_task=source_task,
        objectified_task=objectified,
        mag=mag,
        framework_ref="framework.yaml",
    )
    objectified.metadata["entry_point_adapter_spec"] = (
        tests.entry_point_adapter_spec.to_dict()
    )
    objectified.metadata["abms_oracle_spec"] = tests.abms_oracle_spec.to_dict()
    _write_json(objectified.to_dict(), objectified_path)
    save_tests_json(tests, tests_json_path)
    save_tests_py(tests, tests_py_path)
    save_gold_solution_py(tests, gold_solution_path)
    save_broken_solution_py(tests, broken_solution_path)
    save_abms_oracle_json(tests, abms_oracle_path)
    save_abms_tests_py(tests, abms_tests_py_path)
    save_gold_abms_solution_py(tests, gold_abms_solution_path)
    save_broken_abms_solution_py(tests, broken_abms_solution_path)
    metadata = {
        "case_id": objectified.task_id,
        "source_task_id": source_task.task_id,
        "variant_spec_path": variant_spec_path.name,
        "variant_spec": variant_spec.to_dict() if variant_spec is not None else {},
        "variant_id": variant_spec.variant_id if variant_spec is not None else "",
        "template_family": (
            variant_spec.template_family if variant_spec is not None else "single"
        ),
        "classification": classification.to_dict(),
        "objectified_task_path": objectified_path.name,
        "gold_solution_path": gold_solution_path.name,
        "broken_solution_path": broken_solution_path.name,
        "entry_point_adapter_spec": tests.entry_point_adapter_spec.to_dict(),
        "abms_oracle_spec": tests.abms_oracle_spec.to_dict(),
        "abms_oracle_path": abms_oracle_path.name,
        "abms_tests_py_path": abms_tests_py_path.name,
        "gold_abms_solution_path": gold_abms_solution_path.name,
        "broken_abms_solution_path": broken_abms_solution_path.name,
        "graphical_artifact": graphical_artifact.to_dict(),
        "supports_graph_only": True,
        "supports_graph_nl": True,
        "builder": "full_mag_bench_v1",
        "split": split,
    }
    _write_json(metadata, metadata_path)

    case = FullBenchmarkCase(
        case_id=objectified.task_id,
        source_task_id=source_task.task_id,
        source=source_task.source,
        data_type=classification.data_type,
        task_type=classification.task_type,
        case_dir=f"cases/{objectified.task_id}",
        mag_path="mag.yaml",
        mag_json_path="mag.json",
        dot_path=Path(graphical_artifact.dot_path).name,
        manifest_path=Path(graphical_artifact.manifest_path).name,
        tests_json_path="tests.json",
        tests_py_path="tests.py",
        gold_solution_path="gold_solution.py",
        broken_solution_path="broken_solution.py",
        framework_path="framework.yaml",
        metadata_path="metadata.json",
        supported_input_modes=["structured", "graphical"],
        metadata={
            "entry_point": source_task.entry_point,
            "entry_point_adapter_spec": tests.entry_point_adapter_spec.to_dict(),
            "abms_oracle_spec": tests.abms_oracle_spec.to_dict(),
            "required_method_names": mag.metadata.get("required_method_names", []),
            "split": split,
            "variant_spec_path": "variant_spec.json",
            "variant_id": variant_spec.variant_id if variant_spec is not None else "",
            "template_family": (
                variant_spec.template_family if variant_spec is not None else "single"
            ),
        },
        abms_oracle_path="abms_oracle.json",
        abms_tests_py_path="abms_tests.py",
        gold_abms_solution_path="gold_abms_solution.py",
        broken_abms_solution_path="broken_abms_solution.py",
    )
    _write_yaml(case.to_dict(), case_yaml_path)
    return case


def _save_package(package: FullBenchmarkPackage, path: Path) -> None:
    _write_yaml(package.to_dict(), path)


def _write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def _preserve_existing_artifact_dirs(
    root: Path,
    names: Sequence[str],
) -> dict[str, Path]:
    preserved: dict[str, Path] = {}
    for name in names:
        artifact_dir = root / name
        if not artifact_dir.exists():
            continue
        preserved_path = root.with_name(f".{root.name}_{name}_preserved")
        if preserved_path.exists():
            shutil.rmtree(preserved_path)
        shutil.move(str(artifact_dir), preserved_path)
        preserved[name] = preserved_path
    return preserved


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def _write_readme(root: Path, package: FullBenchmarkPackage) -> None:
    case_lines = "\n".join(
        f"- `{case.case_id}`: {case.task_type} / {case.data_type}"
        for case in package.cases
    )
    content = f"""# {package.name}

Version: `{package.version}`

This package is generated from normalized HumanEval/MBPP-style source tasks.
When built from official normalized files, it is the paper-scale package
candidate for validation and fake/real experiment runners.

## Cases

{case_lines}

## Validate

```bash
python scripts/validate_full_mag_bench.py --bench {root}
```
"""
    (root / "README.md").write_text(content, encoding="utf-8")


def _normalize_formats(formats: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for item in formats:
        for part in str(item).split(","):
            value = part.strip().lower()
            if not value:
                continue
            if value not in {"structured", "graphical", "dot", "svg", "png"}:
                raise ValueError(f"Unsupported full benchmark format: {value!r}.")
            if value not in normalized:
                normalized.append(value)
    if not normalized:
        normalized = ["structured", "graphical"]
    if "structured" not in normalized:
        normalized.insert(0, "structured")
    return normalized


def _validate_source_task_for_oracle(source_task: SourceTask) -> None:
    if source_task.canonical_solution is None or not source_task.canonical_solution:
        raise ValueError("canonical solution is required for functional oracle")
    if not source_task.tests:
        raise ValueError("at least one executable test is required")
    if source_task.entry_point is None or not source_task.entry_point:
        raise ValueError("entry point is required for functional oracle")


def _record_variant_skip(
    skipped: list[dict[str, str]],
    filter_rows: list[dict[str, Any]],
    *,
    source_task: SourceTask,
    variant_spec: TemplateVariantSpec,
    variant_index: int,
    reason: str,
) -> None:
    skipped.append(
        {
            "task_id": source_task.task_id,
            "variant_index": str(variant_index),
            "reason": reason,
        }
    )
    filter_rows.append(
        {
            "accepted": False,
            "source_task_id": source_task.task_id,
            "variant_id": variant_spec.variant_id,
            "variant_index": variant_index,
            "reason": reason,
            "template_family": variant_spec.template_family,
            "task_type": variant_spec.task_type,
            "data_type": variant_spec.data_type,
        }
    )


def _skip_reason_counts(skipped: list[dict[str, str]]) -> dict[str, int]:
    return dict(Counter(item["reason"].split(":", 1)[0] for item in skipped))


def _graphviz_available(warnings: list[str]) -> bool:
    return not any("Graphviz executable" in warning for warning in warnings)


def _graphical_formats(formats: list[str]) -> list[str]:
    output = ["dot"]
    for item in formats:
        if item in {"svg", "png"}:
            output.append(item)
    return output


def _unique_case_id(candidate: str, used: set[str]) -> str:
    if candidate not in used:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in used:
        suffix += 1
    return f"{candidate}_{suffix}"


def _assign_split(index: int, split: str) -> str:
    if split in {"train", "dev", "test"}:
        return split
    if split != "auto":
        return "none"
    bucket = index % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "dev"
    return "test"


def _write_distribution_files(
    root: Path,
    cases: list[FullBenchmarkCase],
    report: FullBenchmarkBuildReport,
) -> None:
    case_index = [
        {
            "case_id": case.case_id,
            "source_task_id": case.source_task_id,
            "source": case.source,
            "task_type": case.task_type,
            "data_type": case.data_type,
            "split": case.metadata.get("split", "none"),
            "variant_id": case.metadata.get("variant_id", ""),
            "template_family": case.metadata.get("template_family", "single"),
            "case_dir": case.case_dir,
        }
        for case in cases
    ]
    _write_json(
        {"cases": case_index, "case_count": len(cases)},
        root / "case_index.json",
    )
    _write_json(
        {
            "task_type_distribution": report.task_type_distribution,
            "data_type_distribution": report.data_type_distribution,
            "source_distribution": report.source_distribution,
            "source_task_distribution": report.source_task_distribution,
            "split_distribution": report.split_distribution,
            "template_family_distribution": report.template_family_distribution,
        },
        root / "task_distribution.json",
    )
    split_cases: dict[str, list[str]] = {}
    for case in cases:
        split_name = str(case.metadata.get("split", "none"))
        split_cases.setdefault(split_name, []).append(case.case_id)
    _write_json(
        {
            "split_distribution": report.split_distribution,
            "splits": split_cases,
        },
        root / "split_manifest.json",
    )


def _write_paper2500_notes(
    root: Path,
    report: FullBenchmarkBuildReport,
    variant_policy: str,
) -> None:
    if variant_policy != "paper2500" and report.target_cases is None:
        return
    target_note = (
        "Exact target reached."
        if report.target_reached
        else (
            "Exact target not reached; accepted count is lower than the requested "
            "target because deterministic filters or source-task availability "
            "limited the usable variants."
        )
    )
    variants_per_task = report.metadata.get("variant_generation_report", {}).get(
        "variants_per_task",
        "n/a",
    )
    content = f"""# Paper-scale ~2500 MAG-Bench Reproduction Notes

The paper describes an about-2,500-template MAG-Bench construction procedure
but does not publish a complete hidden template list. This package is a
deterministic reproduction of that construction procedure, not a claim to be
the original authors' exact private benchmark.

- Official normalized source tasks: {report.total_source_tasks}
- Candidate variants generated: {report.candidate_variants_generated}
- Accepted MAG-test pairs: {report.accepted_cases}
- Requested target: {report.target_cases}
- Target reached: {report.target_reached}
- Variants per task: {variants_per_task}
- Variant policy: `{variant_policy}`
- Deterministic filtering replaces manual duplicate/illogical filtering.
- Source distribution: `{report.source_distribution}`
- Task type distribution: `{report.task_type_distribution}`
- Data type distribution: `{report.data_type_distribution}`
- Template family distribution: `{report.template_family_distribution}`

{target_note}

Every accepted case is expected to contain a structured MAG, DOT + manifest
graphical artifact, source oracle, adapter spec, and ABMS oracle. Unsupported
variants are recorded in `variant_filter_report.json` and excluded from the
accepted benchmark package.
"""
    (root / "paper2500_reproduction_notes.md").write_text(
        content,
        encoding="utf-8",
    )


def _copy_official_reports(root: Path, source_paths: Sequence[str | Path]) -> None:
    for source_path in source_paths:
        parent = Path(source_path).parent
        for filename in [
            "official_download_manifest.json",
            "official_normalization_report.json",
        ]:
            candidate = parent / filename
            if candidate.exists():
                shutil.copy2(candidate, root / filename)


def _official_datasets(root: Path) -> list[str]:
    manifest = _load_optional_json(root / "official_download_manifest.json")
    if not isinstance(manifest, dict):
        return []
    rows = manifest.get("downloads", [])
    if not isinstance(rows, list):
        return []
    return sorted({str(row.get("dataset")) for row in rows if isinstance(row, dict)})


def _official_source_urls(root: Path) -> dict[str, str]:
    manifest = _load_optional_json(root / "official_download_manifest.json")
    if not isinstance(manifest, dict):
        return {}
    rows = manifest.get("downloads", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("dataset")): str(row.get("source_url"))
        for row in rows
        if isinstance(row, dict) and row.get("dataset") and row.get("source_url")
    }


def _official_sha256(root: Path, field_name: str) -> dict[str, str]:
    manifest = _load_optional_json(root / "official_download_manifest.json")
    if not isinstance(manifest, dict):
        return {}
    rows = manifest.get("downloads", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("dataset")): str(row.get(field_name, row.get("sha256", "")))
        for row in rows
        if isinstance(row, dict) and row.get("dataset")
    }


def _normalized_sha256(source_paths: Sequence[str | Path]) -> dict[str, str]:
    return {
        Path(path).stem: _sha256_file(Path(path))
        for path in source_paths
        if Path(path).exists()
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_optional_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def _reason_prefix_count(skipped: list[dict[str, str]], prefix: str) -> int:
    return sum(1 for item in skipped if str(item.get("reason", "")).startswith(prefix))
