"""Full MAG-Bench v1 builder utilities."""

from magcoder.benchmark.full_builder.abms_oracle_spec import (
    ABMSOracleSpec,
    build_abms_oracle_spec,
)
from magcoder.benchmark.full_builder.adapter_spec import (
    EntryPointAdapterSpec,
    build_entry_point_adapter_spec,
)
from magcoder.benchmark.full_builder.benchmark_package import (
    FullBenchmarkBuildReport,
    build_full_mag_bench,
    load_full_benchmark_package,
)
from magcoder.benchmark.full_builder.mag_synthesizer import (
    synthesize_mag_from_objectified_task,
)
from magcoder.benchmark.full_builder.manifest import (
    FullBenchmarkCase,
    FullBenchmarkPackage,
)
from magcoder.benchmark.full_builder.objectifier import (
    ObjectifiedTask,
    objectify_source_task,
)
from magcoder.benchmark.full_builder.source_task import SourceTask
from magcoder.benchmark.full_builder.task_loader import (
    SourceTaskLoadReport,
    load_source_tasks,
    load_source_tasks_from_jsonl,
    load_source_tasks_from_jsonl_with_report,
    load_source_tasks_with_report,
)
from magcoder.benchmark.full_builder.task_taxonomy import (
    TaskClassification,
    classify_source_task,
)
from magcoder.benchmark.full_builder.template_variants import (
    TemplateVariantSpec,
    VariantGenerationConfig,
    VariantGenerationReport,
    generate_template_variant_specs,
)
from magcoder.benchmark.full_builder.test_synthesizer import (
    SynthesizedTestBundle,
    synthesize_tests,
)
from magcoder.benchmark.full_builder.validators import (
    FullBenchmarkValidationReport,
    save_validation_report,
    validate_full_mag_bench,
)
from magcoder.benchmark.full_builder.variant_filters import (
    VariantFilterDecision,
    VariantFilterReport,
    VariantFilterState,
    evaluate_variant,
)

__all__ = [
    "ABMSOracleSpec",
    "FullBenchmarkBuildReport",
    "FullBenchmarkCase",
    "FullBenchmarkPackage",
    "FullBenchmarkValidationReport",
    "EntryPointAdapterSpec",
    "ObjectifiedTask",
    "SourceTask",
    "SourceTaskLoadReport",
    "SynthesizedTestBundle",
    "TaskClassification",
    "TemplateVariantSpec",
    "VariantFilterDecision",
    "VariantFilterReport",
    "VariantFilterState",
    "VariantGenerationConfig",
    "VariantGenerationReport",
    "build_abms_oracle_spec",
    "build_full_mag_bench",
    "build_entry_point_adapter_spec",
    "classify_source_task",
    "evaluate_variant",
    "generate_template_variant_specs",
    "load_full_benchmark_package",
    "load_source_tasks",
    "load_source_tasks_from_jsonl",
    "load_source_tasks_from_jsonl_with_report",
    "load_source_tasks_with_report",
    "objectify_source_task",
    "save_validation_report",
    "synthesize_mag_from_objectified_task",
    "synthesize_tests",
    "validate_full_mag_bench",
]
