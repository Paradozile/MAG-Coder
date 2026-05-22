"""pass@k runner integration for generated Full MAG-Bench packages."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from magcoder.benchmark.full_runner.abms_functional_evaluator import (
    run_abms_case_tests,
)
from magcoder.benchmark.full_runner.case_loader import (
    FullBenchRunCase,
    load_full_bench_run_cases,
)
from magcoder.benchmark.full_runner.evaluation_adapter import (
    evaluate_full_bench_code,
)
from magcoder.benchmark.full_runner.functional_evaluator import run_case_tests
from magcoder.benchmark.full_runner.input_modes import (
    SUPPORTED_FULL_INPUT_MODES,
    baseline_family,
    build_full_bench_prompt_input,
    mode_family,
    mode_nl_condition,
    mode_uses_nl,
)
from magcoder.benchmark.full_runner.result_schema import (
    FullBenchPassKConfig,
    FullBenchPassKResult,
    FullBenchSampleResult,
)
from magcoder.evaluation.passk import (
    estimate_pass_at_k,
    evaluate_success_criterion,
    observed_pass_at_k,
)
from magcoder.generation import (
    DEFAULT_GENERATION_CACHE_DIR,
    GenerationConfig,
    GenerationRequest,
    generate_with_provider,
    save_generation_result,
)
from magcoder.refinement import (
    BalancedRefinementConfig,
    analyze_python_source,
    check_consistency,
    constraints_with_interaction_contracts,
    extract_model_constraints,
    load_framework_spec_from_yaml,
    refine_code,
)
from magcoder.refinement.contracts import build_interaction_contracts

SUPPORTED_FULL_METHODS = (
    "full_with_refinement",
    "no_refinement",
    "mag_coder_full",
    "without_ast",
)
SUPPORTED_EVALUATION_MODES = ("contract", "functional", "both", "abms", "all")


class FullMagBenchPassKRunner:
    """Run Full MAG-Bench cases through generation/refinement/evaluation."""

    def run(self, config: FullBenchPassKConfig) -> FullBenchPassKResult:
        """Run pass@k over a generated Full MAG-Bench package."""

        _validate_config(config)
        output_dir = Path(config.output_dir)
        if output_dir.exists() and config.overwrite:
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_yaml(config.to_dict(), output_dir / "run_config.yaml")
        _save_json(config.to_dict(), output_dir / "run_config.json")
        _save_provider_metadata(config, output_dir / "provider_metadata.json")
        cases = _select_cases(load_full_bench_run_cases(config.bench_path), config)
        _save_selected_cases(cases, output_dir / "selected_cases.json")
        sample_results: list[FullBenchSampleResult] = []
        for case in cases:
            for input_mode in config.modes:
                for method in config.methods:
                    for sample_index in range(config.samples_per_case):
                        sample = _run_or_resume_sample(
                            case=case,
                            input_mode=input_mode,
                            method=method,
                            sample_index=sample_index,
                            config=config,
                            output_dir=output_dir,
                        )
                        sample_results.append(sample)
        sample_results_path = output_dir / "sample_results.jsonl"
        _write_jsonl(
            [sample.to_dict() for sample in sample_results],
            sample_results_path,
        )
        summary_rows = build_full_bench_summary_rows(sample_results, config)
        _write_summary_csv(summary_rows, output_dir / "summary.csv")
        _write_sample_summary_csv(sample_results, output_dir / "sample_summary.csv")
        _write_report(summary_rows, sample_results, output_dir / "report.md", config)
        return FullBenchPassKResult(
            sample_results=sample_results,
            summary_rows=summary_rows,
            output_dir=str(output_dir),
            metadata={
                "sample_results_path": str(sample_results_path),
                "evaluation_mode": config.evaluation,
            },
        )


def build_full_bench_summary_rows(
    samples: list[FullBenchSampleResult],
    config: FullBenchPassKConfig,
) -> list[dict[str, Any]]:
    """Aggregate Full MAG-Bench sample results into summary rows."""

    grouped: dict[tuple[str, str, str], list[FullBenchSampleResult]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.case_id, sample.input_mode, sample.method)].append(sample)
    rows: list[dict[str, Any]] = []
    for (case_id, input_mode, method), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: item.sample_index)
        metadata = ordered[0].metadata if ordered else {}
        successes = [sample.success for sample in ordered]
        contract_successes = [sample.contract_pass for sample in ordered]
        functional_successes = [sample.functional_pass for sample in ordered]
        abms_successes = [sample.abms_pass for sample in ordered]
        num_samples = len(ordered)
        num_correct = sum(1 for success in successes if success)
        for k in config.k_values:
            rows.append(
                {
                    "case_id": case_id,
                    "split": metadata.get("split", ""),
                    "source": metadata.get("source", ""),
                    "task_type": metadata.get("task_type", ""),
                    "data_type": metadata.get("data_type", ""),
                    "template_family": metadata.get("template_family", ""),
                    "mode": input_mode,
                    "mode_family": metadata.get("mode_family", ""),
                    "nl_condition": metadata.get("nl_condition", ""),
                    "input_mode": input_mode,
                    "method": method,
                    "baseline_family": metadata.get("baseline_family", ""),
                    "refinement_strategy": metadata.get(
                        "refinement_strategy",
                        config.refinement_strategy,
                    ),
                    "provider": config.provider,
                    "model_name": config.model_name,
                    "num_samples": num_samples,
                    "num_correct": num_correct,
                    "k": k,
                    "estimated_pass_at_k": _pass_at_k_value(
                        num_samples,
                        num_correct,
                        k,
                    ),
                    "observed_pass_at_k": _observed_pass_at_k_value(successes, k),
                    "contract_num_correct": sum(
                        1 for success in contract_successes if success
                    ),
                    "contract_estimated_pass_at_k": _metric_pass_at_k(
                        contract_successes,
                        k,
                    ),
                    "contract_observed_pass_at_k": _metric_observed_pass_at_k(
                        contract_successes,
                        k,
                    ),
                    "functional_num_correct": _metric_num_correct(
                        functional_successes
                    ),
                    "functional_estimated_pass_at_k": _metric_pass_at_k(
                        functional_successes,
                        k,
                    ),
                    "functional_observed_pass_at_k": _metric_observed_pass_at_k(
                        functional_successes,
                        k,
                    ),
                    "abms_num_correct": _metric_num_correct(abms_successes),
                    "abms_estimated_pass_at_k": _metric_pass_at_k(
                        abms_successes,
                        k,
                    ),
                    "abms_observed_pass_at_k": _metric_observed_pass_at_k(
                        abms_successes,
                        k,
                    ),
                    "checkpoint_accuracy": _format_float(
                        _average([sample.checkpoint_accuracy for sample in ordered])
                    ),
                    "mandatory_success_rate": _format_float(
                        _rate(sample.mandatory_success for sample in ordered)
                    ),
                    "framework_pass_rate": _format_float(
                        _rate(sample.framework_passed for sample in ordered)
                    ),
                    "parse_success_rate": _format_float(
                        _rate(sample.parse_success for sample in ordered)
                    ),
                    "generation_error_rate": _format_float(
                        _rate(
                            sample.generation_error is not None
                            for sample in ordered
                        )
                    ),
                    "cache_hit_rate": _format_float(
                        _rate(sample.cache_hit for sample in ordered)
                    ),
                    "generation_time_seconds": _format_latency(ordered),
                    "sample_id": "aggregate",
                    "final_content_empty": _format_float(
                        _rate(
                            _metadata_bool(sample, "final_content_empty")
                            for sample in ordered
                        )
                    ),
                    "syntax_parse_failure": _format_float(
                        _rate(not sample.parse_success for sample in ordered)
                    ),
                    "extraction_failure": _format_float(
                        _rate(
                            not _metadata_bool(sample, "extracted_code_non_empty")
                            for sample in ordered
                        )
                    ),
                    "backend_error": _format_float(
                        _rate(
                            sample.generation_error is not None
                            for sample in ordered
                        )
                    ),
                    "refinement_error": _format_float(
                        _rate(
                            _metadata_bool(sample, "refinement_error")
                            for sample in ordered
                        )
                    ),
                    "contract_pass": _format_float(
                        _rate(sample.contract_pass for sample in ordered)
                    ),
                    "functional_pass": _format_optional_rate(
                        [sample.functional_pass for sample in ordered]
                    ),
                    "evaluation_mode": config.evaluation,
                    "functional_error_type": _join_nonempty(
                        sample.functional_error_type for sample in ordered
                    ),
                    "functional_error_message": _first_nonempty(
                        sample.functional_error_message for sample in ordered
                    ),
                    "entrypoint_mode": _join_nonempty(
                        sample.entrypoint_mode for sample in ordered
                    ),
                    "adapter_used": _format_optional_rate(
                        [sample.adapter_used for sample in ordered]
                    ),
                    "adapter_success": _format_optional_rate(
                        [sample.adapter_success for sample in ordered]
                    ),
                    "adapter_error_type": _join_nonempty(
                        sample.adapter_error_type for sample in ordered
                    ),
                    "adapter_error_message": _first_nonempty(
                        sample.adapter_error_message for sample in ordered
                    ),
                    "direct_entrypoint_present": _format_optional_rate(
                        [sample.direct_entrypoint_present for sample in ordered]
                    ),
                    "required_candidate_symbols_present": _format_optional_rate(
                        [
                            sample.required_candidate_symbols_present
                            for sample in ordered
                        ]
                    ),
                    "abms_supported": _format_optional_rate(
                        [sample.abms_supported for sample in ordered]
                    ),
                    "abms_pass": _format_optional_rate(
                        [sample.abms_pass for sample in ordered]
                    ),
                    "abms_error_type": _join_nonempty(
                        sample.abms_error_type for sample in ordered
                    ),
                    "abms_error_message": _first_nonempty(
                        sample.abms_error_message for sample in ordered
                    ),
                    "abms_interaction_calls_executed": _format_float(
                        _average(
                            [
                                float(sample.abms_interaction_calls_executed or 0)
                                for sample in ordered
                            ]
                        )
                    ),
                    "abms_required_symbols_present": _format_optional_rate(
                        [sample.abms_required_symbols_present for sample in ordered]
                    ),
                    "evaluation_level": config.evaluation,
                }
            )
    return rows


def _run_or_resume_sample(
    *,
    case: FullBenchRunCase,
    input_mode: str,
    method: str,
    sample_index: int,
    config: FullBenchPassKConfig,
    output_dir: Path,
) -> FullBenchSampleResult:
    sample_dir = (
        output_dir
        / "cases"
        / case.case_id
        / input_mode
        / method
        / f"sample_{sample_index:03d}"
    )
    result_path = sample_dir / "sample_result.json"
    if config.resume and result_path.exists() and not config.overwrite:
        return FullBenchSampleResult.from_dict(_load_json(result_path))
    sample_dir.mkdir(parents=True, exist_ok=True)
    seed = config.base_seed + sample_index
    try:
        sample = _run_sample(
            case=case,
            input_mode=input_mode,
            method=method,
            sample_index=sample_index,
            seed=seed,
            config=config,
            sample_dir=sample_dir,
        )
    except Exception as error:  # noqa: BLE001 - keep matrix running.
        sample = _failed_sample(
            case=case,
            input_mode=input_mode,
            method=method,
            sample_index=sample_index,
            seed=seed,
            sample_dir=sample_dir,
            error=error,
            evaluation_mode=config.evaluation,
            refinement_strategy=config.refinement_strategy,
        )
    _save_json(sample.to_dict(), result_path)
    return sample


def _run_sample(
    *,
    case: FullBenchRunCase,
    input_mode: str,
    method: str,
    sample_index: int,
    seed: int,
    config: FullBenchPassKConfig,
    sample_dir: Path,
) -> FullBenchSampleResult:
    prompt_input = build_full_bench_prompt_input(case, input_mode)
    framework = load_framework_spec_from_yaml(case.framework_path)
    constraints = constraints_with_interaction_contracts(
        extract_model_constraints(prompt_input.mag),
        build_interaction_contracts(prompt_input.mag, framework),
    )
    request = GenerationRequest(
        hybrid_input=prompt_input.hybrid_input,
        config=GenerationConfig(
            provider=config.provider,
            model_name=config.model_name,
            api_base_url=config.api_base_url,
            api_key_env=config.api_key_env,
            timeout_seconds=config.timeout_seconds,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            prompt_profile_id=config.prompt_profile_id,
            prompt_profile_path=config.prompt_profile_path,
            require_non_empty_code=config.require_non_empty_code,
            think=config.think,
            native_endpoint=config.native_endpoint,
            num_predict=config.num_predict,
            seed=seed,
            prompt_variant=method,
            input_mode=input_mode,
            cache_enabled=config.cache_enabled,
            dry_run=config.provider == "dry_run",
            mode=prompt_input.hybrid_mode,
            metadata={
                "benchmark_id": "full_mag_bench_v1",
                "case_id": case.case_id,
                "split": case.package_case.metadata.get("split", "none"),
                "source": case.package_case.source,
                "task_type": case.package_case.task_type,
                "data_type": case.package_case.data_type,
                "template_family": case.package_case.metadata.get(
                    "template_family",
                    "single",
                ),
                "mode_family": mode_family(input_mode),
                "nl_condition": mode_nl_condition(input_mode),
                "input_mode": input_mode,
                "method": method,
                "baseline_family": baseline_family(input_mode, method),
                "sample_index": sample_index,
                "seed": seed,
                "refinement_strategy": config.refinement_strategy,
            },
        ),
        prompt_text=prompt_input.prompt_text,
    )
    generation = generate_with_provider(
        request,
        cache_dir=str(config.metadata.get("cache_dir", DEFAULT_GENERATION_CACHE_DIR)),
    )
    generation_path = sample_dir / "generation_result.json"
    save_generation_result(generation, generation_path)
    code_initial_path = sample_dir / "code_initial.py"
    code_initial_path.write_text(generation.code_initial, encoding="utf-8")
    initial_analysis = analyze_python_source(generation.code_initial)
    initial_report = check_consistency(initial_analysis, constraints, framework)
    initial_report_path = sample_dir / "initial_report.json"
    _save_json(initial_report.to_dict(), initial_report_path)

    if _method_uses_refinement(method):
        refined = refine_code(
            generation.code_initial,
            prompt_input.mag,
            framework,
            refinement_config=BalancedRefinementConfig(
                strategy=config.refinement_strategy,
            ),
        )
        code_refined_path = sample_dir / "code_refined.py"
        refined_report_path = sample_dir / "refined_report.json"
        code_refined_path.write_text(refined.code_refined, encoding="utf-8")
        _save_json(refined.refined_report.to_dict(), refined_report_path)
        active_code = refined.code_refined
        active_code_path = code_refined_path
        active_analysis = analyze_python_source(refined.code_refined)
        active_report = refined.refined_report
    else:
        code_refined_path = None
        refined_report_path = None
        active_code = generation.code_initial
        active_code_path = code_initial_path
        active_analysis = initial_analysis
        active_report = initial_report

    evaluation = evaluate_full_bench_code(
        case=case,
        code=active_code,
        analysis=active_analysis,
        report=active_report,
        constraints=constraints,
        framework=framework,
        mag=prompt_input.mag,
    )
    functional_report = None
    functional_report_path = None
    if config.evaluation in {"functional", "both", "all"}:
        functional_report = run_case_tests(
            case.case_dir,
            active_code_path,
            timeout_seconds=int(config.metadata.get("functional_timeout_seconds", 5)),
        )
        functional_report_path = sample_dir / "functional_report.json"
        _save_json(functional_report.to_dict(), functional_report_path)
    abms_report = None
    abms_report_path = None
    if config.evaluation in {"abms", "all"}:
        abms_report = run_abms_case_tests(
            case.case_dir,
            active_code_path,
            timeout_seconds=int(config.metadata.get("abms_timeout_seconds", 5)),
        )
        abms_report_path = sample_dir / "abms_report.json"
        _save_json(abms_report.to_dict(), abms_report_path)

    evaluation_path = sample_dir / "evaluation.json"
    _save_json(
        {
            "evaluation_mode": config.evaluation,
            "contract": evaluation.to_dict(),
            "functional": (
                None if functional_report is None else functional_report.to_dict()
            ),
            "abms": None if abms_report is None else abms_report.to_dict(),
        },
        evaluation_path,
    )
    contract_success = evaluate_success_criterion(
        criterion=config.success_criterion,
        parse_success=evaluation.parse_success,
        framework_passed=evaluation.framework_passed,
        mandatory_success=evaluation.mandatory_success,
        checkpoint_accuracy=evaluation.checkpoint_accuracy,
    )
    functional_success = (
        None if functional_report is None else bool(functional_report.passed)
    )
    abms_success = None if abms_report is None else bool(abms_report.abms_pass)
    success = _combined_success(
        config.evaluation,
        contract_success,
        functional_success,
        abms_success,
    )
    return FullBenchSampleResult(
        case_id=case.case_id,
        input_mode=input_mode,
        method=method,
        sample_index=sample_index,
        seed=seed,
        generation_result_path=str(generation_path),
        code_initial_path=str(code_initial_path),
        code_refined_path=None if code_refined_path is None else str(code_refined_path),
        initial_report_path=str(initial_report_path),
        refined_report_path=(
            None if refined_report_path is None else str(refined_report_path)
        ),
        evaluation_path=str(evaluation_path),
        parse_success=evaluation.parse_success,
        framework_passed=evaluation.framework_passed,
        mandatory_success=evaluation.mandatory_success,
        checkpoint_accuracy=evaluation.checkpoint_accuracy,
        success=success,
        cache_hit=bool(generation.metadata.get("cache_hit", False)),
        contract_pass=contract_success,
        functional_pass=functional_success,
        evaluation_mode=config.evaluation,
        functional_error_type=(
            None if functional_report is None else functional_report.error_type
        ),
        functional_error_message=(
            None if functional_report is None else functional_report.error_message
        ),
        entrypoint_mode=(
            None if functional_report is None else functional_report.entrypoint_mode
        ),
        adapter_used=(
            None if functional_report is None else functional_report.adapter_used
        ),
        adapter_success=(
            None if functional_report is None else functional_report.adapter_success
        ),
        adapter_error_type=(
            None if functional_report is None else functional_report.adapter_error_type
        ),
        adapter_error_message=(
            None
            if functional_report is None
            else functional_report.adapter_error_message
        ),
        direct_entrypoint_present=(
            None
            if functional_report is None
            else functional_report.direct_entrypoint_present
        ),
        required_candidate_symbols_present=(
            None
            if functional_report is None
            else functional_report.required_candidate_symbols_present
        ),
        abms_supported=None if abms_report is None else abms_report.abms_supported,
        abms_pass=abms_success,
        abms_error_type=None if abms_report is None else abms_report.abms_error_type,
        abms_error_message=(
            None if abms_report is None else abms_report.abms_error_message
        ),
        abms_interaction_calls_executed=(
            None if abms_report is None else abms_report.interaction_calls_executed
        ),
        abms_required_symbols_present=(
            None
            if abms_report is None
            else abms_report.required_candidate_symbols_present
        ),
        latency_seconds=float(generation.metadata.get("latency_seconds", 0.0)),
        metadata={
            "source_task_id": case.package_case.source_task_id,
            "split": case.package_case.metadata.get("split", "none"),
            "source": case.package_case.source,
            "task_type": case.package_case.task_type,
            "data_type": case.package_case.data_type,
            "template_family": case.package_case.metadata.get(
                "template_family",
                "single",
            ),
            "source_kind": prompt_input.source_kind,
            "mode_family": mode_family(input_mode),
            "nl_condition": mode_nl_condition(input_mode),
            "use_nl": mode_uses_nl(input_mode),
            "use_refinement": _method_uses_refinement(method),
            "baseline_family": baseline_family(input_mode, method),
            "refinement_strategy": config.refinement_strategy,
            "evaluation_level": evaluation.evaluation_level,
            "functional_report_path": (
                None if functional_report_path is None else str(functional_report_path)
            ),
            "abms_report_path": (
                None if abms_report_path is None else str(abms_report_path)
            ),
            "prompt_hash": generation.metadata.get("prompt_hash"),
            "request_hash": generation.metadata.get("request_hash"),
            "provider_metadata": _sample_provider_metadata(generation.metadata),
        },
    )


def _failed_sample(
    *,
    case: FullBenchRunCase,
    input_mode: str,
    method: str,
    sample_index: int,
    seed: int,
    sample_dir: Path,
    error: Exception,
    evaluation_mode: str,
    refinement_strategy: str,
) -> FullBenchSampleResult:
    return FullBenchSampleResult(
        case_id=case.case_id,
        input_mode=input_mode,
        method=method,
        sample_index=sample_index,
        seed=seed,
        generation_result_path=str(sample_dir / "generation_result.json"),
        code_initial_path=str(sample_dir / "code_initial.py"),
        initial_report_path=str(sample_dir / "initial_report.json"),
        evaluation_path=str(sample_dir / "evaluation.json"),
        parse_success=False,
        framework_passed=False,
        mandatory_success=False,
        checkpoint_accuracy=0.0,
        success=False,
        cache_hit=False,
        contract_pass=False,
        functional_pass=(
            False if evaluation_mode in {"functional", "both", "all"} else None
        ),
        evaluation_mode=evaluation_mode,
        entrypoint_mode=(
            "missing" if evaluation_mode in {"functional", "both", "all"} else None
        ),
        adapter_used=(
            False if evaluation_mode in {"functional", "both", "all"} else None
        ),
        adapter_success=(
            False if evaluation_mode in {"functional", "both", "all"} else None
        ),
        direct_entrypoint_present=(
            False if evaluation_mode in {"functional", "both", "all"} else None
        ),
        required_candidate_symbols_present=(
            False if evaluation_mode in {"functional", "both", "all"} else None
        ),
        abms_supported=False if evaluation_mode in {"abms", "all"} else None,
        abms_pass=False if evaluation_mode in {"abms", "all"} else None,
        abms_error_type=(
            type(error).__name__ if evaluation_mode in {"abms", "all"} else None
        ),
        abms_error_message=str(error) if evaluation_mode in {"abms", "all"} else None,
        abms_interaction_calls_executed=0
        if evaluation_mode in {"abms", "all"}
        else None,
        abms_required_symbols_present=False
        if evaluation_mode in {"abms", "all"}
        else None,
        generation_error=f"{type(error).__name__}: {error}",
        metadata={
            "evaluation_level": evaluation_mode,
            "split": case.package_case.metadata.get("split", "none"),
            "source": case.package_case.source,
            "task_type": case.package_case.task_type,
            "data_type": case.package_case.data_type,
            "template_family": case.package_case.metadata.get(
                "template_family",
                "single",
            ),
            "mode_family": mode_family(input_mode),
            "nl_condition": mode_nl_condition(input_mode),
            "baseline_family": baseline_family(input_mode, method),
            "refinement_strategy": refinement_strategy,
        },
    )


def _validate_config(config: FullBenchPassKConfig) -> None:
    if config.samples_per_case <= 0:
        raise ValueError("samples_per_case must be positive.")
    if not config.k_values or any(k <= 0 for k in config.k_values):
        raise ValueError("k_values must contain positive integers.")
    missing_modes = [
        mode for mode in config.modes if mode not in SUPPORTED_FULL_INPUT_MODES
    ]
    if missing_modes:
        raise ValueError(f"Unsupported modes: {', '.join(missing_modes)}")
    missing_methods = [
        method for method in config.methods if method not in SUPPORTED_FULL_METHODS
    ]
    if missing_methods:
        raise ValueError(f"Unsupported methods: {', '.join(missing_methods)}")
    if config.evaluation not in SUPPORTED_EVALUATION_MODES:
        raise ValueError(f"Unsupported evaluation mode: {config.evaluation}")
    if config.split not in {"all", "none", "train", "dev", "test"}:
        raise ValueError(f"Unsupported split: {config.split}")
    if config.refinement_strategy not in {
        "abms_focused",
        "balanced",
        "conservative",
    }:
        raise ValueError(
            f"Unsupported refinement strategy: {config.refinement_strategy}"
        )
    evaluate_success_criterion(
        criterion=config.success_criterion,
        parse_success=False,
        framework_passed=False,
        mandatory_success=False,
        checkpoint_accuracy=0.0,
    )


def _select_cases(
    cases: list[FullBenchRunCase],
    config: FullBenchPassKConfig,
) -> list[FullBenchRunCase]:
    selected = list(cases)
    if config.split not in {"all", "none"}:
        selected = [
            case
            for case in selected
            if str(case.package_case.metadata.get("split", "none")) == config.split
        ]
    case_ids = _load_case_list(config.case_list_path)
    if case_ids is not None:
        allowed = set(case_ids)
        selected = [case for case in selected if case.case_id in allowed]
        order = {case_id: index for index, case_id in enumerate(case_ids)}
        selected.sort(key=lambda case: order.get(case.case_id, len(order)))
    if config.max_cases is not None:
        selected = selected[: config.max_cases]
    return selected


def _load_case_list(path: str | None) -> list[str] | None:
    if path is None:
        return None
    case_path = Path(path)
    if case_path.suffix.lower() == ".json":
        data = _load_json(case_path)
        raw_items: Any
        if isinstance(data.get("case_ids"), list):
            raw_items = data["case_ids"]
        elif isinstance(data.get("selected_cases"), list):
            raw_items = data["selected_cases"]
        elif isinstance(data.get("cases"), list):
            raw_items = data["cases"]
        else:
            raise ValueError(
                "JSON case-list must contain case_ids, selected_cases, or cases."
            )
        case_ids: list[str] = []
        for item in raw_items:
            if isinstance(item, str):
                case_ids.append(item)
            elif isinstance(item, dict) and item.get("case_id"):
                case_ids.append(str(item["case_id"]))
        return case_ids
    case_ids = [
        line.strip()
        for line in case_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return case_ids


def _save_selected_cases(cases: list[FullBenchRunCase], path: Path) -> None:
    _save_json(
        {
            "case_count": len(cases),
            "case_ids": [case.case_id for case in cases],
            "cases": [
                {
                    "case_id": case.case_id,
                    "source_task_id": case.package_case.source_task_id,
                    "source": case.package_case.source,
                    "split": case.package_case.metadata.get("split", "none"),
                    "task_type": case.package_case.task_type,
                    "data_type": case.package_case.data_type,
                    "template_family": case.package_case.metadata.get(
                        "template_family",
                        "single",
                    ),
                    "variant_id": case.package_case.metadata.get("variant_id", ""),
                    "case_dir": case.package_case.case_dir,
                }
                for case in cases
            ],
        },
        path,
    )


def _write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "split",
        "source",
        "task_type",
        "data_type",
        "template_family",
        "mode",
        "mode_family",
        "nl_condition",
        "input_mode",
        "method",
        "baseline_family",
        "refinement_strategy",
        "provider",
        "model_name",
        "num_samples",
        "num_correct",
        "k",
        "estimated_pass_at_k",
        "observed_pass_at_k",
        "contract_num_correct",
        "contract_estimated_pass_at_k",
        "contract_observed_pass_at_k",
        "functional_num_correct",
        "functional_estimated_pass_at_k",
        "functional_observed_pass_at_k",
        "abms_num_correct",
        "abms_estimated_pass_at_k",
        "abms_observed_pass_at_k",
        "checkpoint_accuracy",
        "mandatory_success_rate",
        "framework_pass_rate",
        "parse_success_rate",
        "generation_error_rate",
        "cache_hit_rate",
        "generation_time_seconds",
        "sample_id",
        "final_content_empty",
        "syntax_parse_failure",
        "extraction_failure",
        "backend_error",
        "refinement_error",
        "contract_pass",
        "functional_pass",
        "evaluation_mode",
        "functional_error_type",
        "functional_error_message",
        "entrypoint_mode",
        "adapter_used",
        "adapter_success",
        "adapter_error_type",
        "adapter_error_message",
        "direct_entrypoint_present",
        "required_candidate_symbols_present",
        "abms_supported",
        "abms_pass",
        "abms_error_type",
        "abms_error_message",
        "abms_interaction_calls_executed",
        "abms_required_symbols_present",
        "evaluation_level",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_sample_summary_csv(samples: list[FullBenchSampleResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "split",
        "source",
        "task_type",
        "data_type",
        "template_family",
        "mode",
        "mode_family",
        "nl_condition",
        "input_mode",
        "method",
        "baseline_family",
        "refinement_strategy",
        "sample_id",
        "sample_index",
        "seed",
        "contract_pass",
        "functional_pass",
        "evaluation_mode",
        "success",
        "final_content_empty",
        "syntax_parse_failure",
        "extraction_failure",
        "backend_error",
        "refinement_error",
        "functional_error_type",
        "functional_error_message",
        "entrypoint_mode",
        "adapter_used",
        "adapter_success",
        "adapter_error_type",
        "adapter_error_message",
        "direct_entrypoint_present",
        "required_candidate_symbols_present",
        "abms_supported",
        "abms_pass",
        "abms_error_type",
        "abms_error_message",
        "abms_interaction_calls_executed",
        "abms_required_symbols_present",
        "generation_error",
        "cache_hit",
        "latency_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "case_id": sample.case_id,
                    "split": sample.metadata.get("split", ""),
                    "source": sample.metadata.get("source", ""),
                    "task_type": sample.metadata.get("task_type", ""),
                    "data_type": sample.metadata.get("data_type", ""),
                    "template_family": sample.metadata.get("template_family", ""),
                    "mode": sample.input_mode,
                    "mode_family": sample.metadata.get("mode_family", ""),
                    "nl_condition": sample.metadata.get("nl_condition", ""),
                    "input_mode": sample.input_mode,
                    "method": sample.method,
                    "baseline_family": sample.metadata.get("baseline_family", ""),
                    "refinement_strategy": sample.metadata.get(
                        "refinement_strategy",
                        "",
                    ),
                    "sample_id": f"sample_{sample.sample_index:03d}",
                    "sample_index": sample.sample_index,
                    "seed": sample.seed,
                    "contract_pass": sample.contract_pass,
                    "functional_pass": sample.functional_pass,
                    "evaluation_mode": sample.evaluation_mode,
                    "success": sample.success,
                    "final_content_empty": _metadata_bool(
                        sample,
                        "final_content_empty",
                    ),
                    "syntax_parse_failure": not sample.parse_success,
                    "extraction_failure": not _metadata_bool(
                        sample,
                        "extracted_code_non_empty",
                    ),
                    "backend_error": sample.generation_error is not None,
                    "refinement_error": _metadata_bool(sample, "refinement_error"),
                    "functional_error_type": sample.functional_error_type,
                    "functional_error_message": sample.functional_error_message,
                    "entrypoint_mode": sample.entrypoint_mode,
                    "adapter_used": sample.adapter_used,
                    "adapter_success": sample.adapter_success,
                    "adapter_error_type": sample.adapter_error_type,
                    "adapter_error_message": sample.adapter_error_message,
                    "direct_entrypoint_present": sample.direct_entrypoint_present,
                    "required_candidate_symbols_present": (
                        sample.required_candidate_symbols_present
                    ),
                    "abms_supported": sample.abms_supported,
                    "abms_pass": sample.abms_pass,
                    "abms_error_type": sample.abms_error_type,
                    "abms_error_message": sample.abms_error_message,
                    "abms_interaction_calls_executed": (
                        sample.abms_interaction_calls_executed
                    ),
                    "abms_required_symbols_present": (
                        sample.abms_required_symbols_present
                    ),
                    "generation_error": sample.generation_error,
                    "cache_hit": sample.cache_hit,
                    "latency_seconds": sample.latency_seconds,
                }
            )


def _write_report(
    rows: list[dict[str, Any]],
    samples: list[FullBenchSampleResult],
    path: Path,
    config: FullBenchPassKConfig,
) -> None:
    failures = [sample for sample in samples if not sample.success]
    backend_errors = sum(1 for sample in samples if sample.generation_error)
    syntax_failures = sum(1 for sample in samples if not sample.parse_success)
    extraction_failures = sum(
        1
        for sample in samples
        if not _metadata_bool(sample, "extracted_code_non_empty")
    )
    final_content_empty = sum(
        1 for sample in samples if _metadata_bool(sample, "final_content_empty")
    )
    refinement_errors = sum(
        1 for sample in samples if _metadata_bool(sample, "refinement_error")
    )
    missing_entrypoints = sum(
        1
        for sample in samples
        if sample.functional_error_type == "MissingEntryPoint"
    )
    abms_supported = sum(1 for sample in samples if sample.abms_supported)
    abms_passed = sum(1 for sample in samples if sample.abms_pass)
    abms_failures = sum(
        1
        for sample in samples
        if sample.abms_supported and sample.abms_pass is False
    )
    lines = [
        "# Full MAG-Bench pass@k Report",
        "",
        "## Run Configuration",
        "",
        f"- Provider: `{config.provider}`",
        f"- Model: `{config.model_name}`",
        f"- Evaluation mode: `{config.evaluation}`",
        f"- Refinement strategy: `{config.refinement_strategy}`",
        "- Frozen benchmark bundle: "
        f"`{config.metadata.get('frozen_benchmark_path', '')}`",
        f"- Split/subset: `{config.split}`",
        f"- Max cases: `{config.max_cases}`",
        f"- Samples per case: {config.samples_per_case}",
        f"- Modes: `{', '.join(config.modes)}`",
        f"- Methods: `{', '.join(config.methods)}`",
        f"- Average generation time seconds: `{_average_latency(samples)}`",
        "",
        "## Selected Cases",
        "",
        f"- Selected case count: {len({sample.case_id for sample in samples})}",
        "- Selected case ids: "
        f"`{', '.join(sorted({sample.case_id for sample in samples}))}`",
        f"- Source distribution: `{_metadata_distribution(samples, 'source')}`",
        f"- Task type distribution: `{_metadata_distribution(samples, 'task_type')}`",
        f"- Data type distribution: `{_metadata_distribution(samples, 'data_type')}`",
        "- Template family distribution: "
        f"`{_metadata_distribution(samples, 'template_family')}`",
        "",
        "## Results",
        "",
        f"- Evaluation mode: `{rows[0]['evaluation_mode'] if rows else 'unknown'}`",
        f"- Samples: {len(samples)}",
        f"- Summary rows: {len(rows)}",
        f"- Failed samples: {len(failures)}",
        f"- Backend errors: {backend_errors}",
        f"- Extraction failures: {extraction_failures}",
        f"- Syntax parse failures: {syntax_failures}",
        f"- Refinement errors: {refinement_errors}",
        f"- Final content empty: {final_content_empty}",
        f"- MissingEntryPoint: {missing_entrypoints}",
        f"- Interface failures: {_interface_failure_count(samples)}",
        f"- Behavior/runtime failures: {_behavior_failure_count(samples)}",
        f"- ABMS supported samples: {abms_supported}",
        f"- ABMS passed samples: {abms_passed}",
        f"- ABMS failed samples: {abms_failures}",
        "",
        "| case_id | input_mode | method | k | estimated_pass_at_k | "
        "observed_pass_at_k | contract_pass@k | functional_pass@k | "
        "abms_pass@k | mandatory_success_rate | contract_pass | "
        "functional_pass | abms_pass |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | "
        "---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {input_mode} | {method} | {k} | "
            "{estimated_pass_at_k} | {observed_pass_at_k} | "
            "{contract_estimated_pass_at_k} | "
            "{functional_estimated_pass_at_k} | {abms_estimated_pass_at_k} | "
            "{mandatory_success_rate} | {contract_pass} | {functional_pass} | "
            "{abms_pass} |".format(
                **row
            )
        )
    lines.extend(["", "## Contract-Level pass@k", ""])
    lines.extend(_summary_lines(rows, "contract_pass"))
    lines.extend(["", "## Functional-Level pass@k", ""])
    lines.extend(_summary_lines(rows, "functional_pass"))
    lines.extend(["", "## ABMS-Level pass@k", ""])
    lines.extend(_summary_lines(rows, "abms_pass"))
    lines.extend(["", "## Method Comparison", ""])
    lines.extend(_comparison_lines(samples, "method"))
    lines.extend(["", "## Input Mode Comparison", ""])
    lines.extend(_comparison_lines(samples, "input_mode"))
    lines.extend(["", "## RQ2 Mode Family Comparison", ""])
    lines.extend(_comparison_lines(samples, "mode_family"))
    lines.extend(["", "## RQ2 NL Condition Comparison", ""])
    lines.extend(_comparison_lines(samples, "nl_condition"))
    lines.extend(["", "## Per-Case Contract/Functional Summary", ""])
    lines.extend(_per_case_lines(samples))
    lines.extend(["", "## Functional Error Type Distribution", ""])
    lines.extend(_functional_error_distribution_lines(samples))
    lines.extend(["", "## ABMS Error Type Distribution", ""])
    lines.extend(_abms_error_distribution_lines(samples))
    lines.extend(["", "## Interface vs Behavior Failure Taxonomy", ""])
    lines.extend(_failure_taxonomy_lines(samples))
    lines.extend(["", "## Known Failure Patterns", ""])
    lines.extend(_failure_pattern_lines(samples))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _combined_success(
    evaluation_mode: str,
    contract_success: bool,
    functional_success: bool | None,
    abms_success: bool | None,
) -> bool:
    if evaluation_mode == "contract":
        return contract_success
    if evaluation_mode == "functional":
        return bool(functional_success)
    if evaluation_mode == "abms":
        return bool(abms_success)
    if evaluation_mode == "all":
        return contract_success and bool(functional_success) and bool(abms_success)
    return contract_success and bool(functional_success)


def _summary_lines(rows: list[dict[str, Any]], metric: str) -> list[str]:
    passk_field = f"{metric}_estimated_pass_at_k"
    observed_field = f"{metric}_observed_pass_at_k"
    lines = [
        "| case_id | input_mode | method | k | estimated_pass_at_k | "
        f"observed_pass_at_k | {metric}_rate |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('case_id', '')} | {row.get('input_mode', '')} | "
            f"{row.get('method', '')} | {row.get('k', '')} | "
            f"{row.get(passk_field, '')} | {row.get(observed_field, '')} | "
            f"{row.get(metric, '')} |"
        )
    return lines


def _comparison_lines(samples: list[FullBenchSampleResult], key: str) -> list[str]:
    grouped: dict[str, list[FullBenchSampleResult]] = defaultdict(list)
    for sample in samples:
        grouped[_sample_group_value(sample, key)].append(sample)
    lines = [
        f"| {key} | samples | contract_pass_rate | functional_pass_rate | "
        "abms_pass_rate | backend_error_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for value, group in sorted(grouped.items()):
        functional_values = [sample.functional_pass for sample in group]
        abms_values = [sample.abms_pass for sample in group]
        lines.append(
            f"| {value} | {len(group)} | "
            f"{_format_float(_rate(sample.contract_pass for sample in group))} | "
            f"{_format_optional_rate(functional_values)} | "
            f"{_format_optional_rate(abms_values)} | "
            f"{_format_float(_rate(sample.generation_error for sample in group))} |"
        )
    return lines


def _sample_group_value(sample: FullBenchSampleResult, key: str) -> str:
    if key == "input_mode":
        return sample.input_mode
    if key == "method":
        return sample.method
    value = sample.metadata.get(key, "")
    return str(value or "unknown")


def _failure_pattern_lines(samples: list[FullBenchSampleResult]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        if sample.generation_error:
            counts["backend_error"] += 1
        if not sample.parse_success:
            counts["syntax_parse_failure"] += 1
        if not _metadata_bool(sample, "extracted_code_non_empty"):
            counts["extraction_failure"] += 1
        if sample.functional_error_type:
            counts[f"functional:{sample.functional_error_type}"] += 1
        if sample.adapter_error_type:
            counts[f"adapter:{sample.adapter_error_type}"] += 1
        if sample.abms_error_type:
            counts[f"abms:{sample.abms_error_type}"] += 1
    if not counts:
        return ["No structured failure patterns were recorded."]
    return [f"- {key}: {count}" for key, count in sorted(counts.items())]


def _metadata_distribution(
    samples: list[FullBenchSampleResult],
    key: str,
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    seen: set[str] = set()
    for sample in samples:
        unique_key = f"{sample.case_id}:{key}"
        if unique_key in seen:
            continue
        seen.add(unique_key)
        value = str(sample.metadata.get(key, ""))
        counts[value or "unknown"] += 1
    return dict(sorted(counts.items()))


def _per_case_lines(samples: list[FullBenchSampleResult]) -> list[str]:
    grouped: dict[str, list[FullBenchSampleResult]] = defaultdict(list)
    for sample in samples:
        grouped[sample.case_id].append(sample)
    lines = [
        "| case_id | samples | contract_pass | functional_pass | "
        "missing_entrypoint |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for case_id, group in sorted(grouped.items()):
        missing_entrypoints = sum(
            1
            for sample in group
            if sample.functional_error_type == "MissingEntryPoint"
        )
        lines.append(
            f"| {case_id} | {len(group)} | "
            f"{sum(1 for sample in group if sample.contract_pass)} | "
            f"{sum(1 for sample in group if sample.functional_pass)} | "
            f"{missing_entrypoints} |"
        )
    return lines


def _functional_error_distribution_lines(
    samples: list[FullBenchSampleResult],
) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        key = sample.functional_error_type or "none"
        counts[key] += 1
    return [f"- {key}: {count}" for key, count in sorted(counts.items())]


def _abms_error_distribution_lines(
    samples: list[FullBenchSampleResult],
) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        key = sample.abms_error_type or "none"
        counts[key] += 1
    return [f"- {key}: {count}" for key, count in sorted(counts.items())]


def _failure_taxonomy_lines(samples: list[FullBenchSampleResult]) -> list[str]:
    interface_failures = _interface_failure_count(samples)
    behavior_failures = _behavior_failure_count(samples)
    backend_failures = sum(1 for sample in samples if sample.generation_error)
    lines = [
        f"- interface_errors: {interface_failures}",
        f"- behavior_or_runtime_errors: {behavior_failures}",
        f"- backend_errors: {backend_failures}",
    ]
    if interface_failures == 0 and behavior_failures:
        lines.append(
            "- Functional failures are behavior/runtime failures, not entry-point "
            "interface failures."
        )
    return lines


def _interface_failure_count(samples: list[FullBenchSampleResult]) -> int:
    interface_errors = {
        "MissingEntryPoint",
        "AdapterConstructionFailure",
    }
    return sum(
        1 for sample in samples if sample.functional_error_type in interface_errors
    )


def _behavior_failure_count(samples: list[FullBenchSampleResult]) -> int:
    interface_errors = {
        "MissingEntryPoint",
        "AdapterConstructionFailure",
    }
    return sum(
        1
        for sample in samples
        if sample.functional_error_type
        and sample.functional_error_type not in interface_errors
    )


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def _save_provider_metadata(config: FullBenchPassKConfig, path: Path) -> None:
    _save_json(
        {
            "provider": config.provider,
            "model_name": config.model_name,
            "api_base_url": config.api_base_url,
            "api_key_env": config.api_key_env,
            "think": config.think,
            "native_endpoint": config.native_endpoint,
            "prompt_profile_id": config.prompt_profile_id,
            "prompt_profile_path": config.prompt_profile_path,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "num_predict": config.num_predict,
            "timeout_seconds": config.timeout_seconds,
        },
        path,
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return data


def _save_yaml(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _rate(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(1 for value in items if value) / len(items)


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _format_optional_rate(values: list[bool | None]) -> str:
    known = [value for value in values if value is not None]
    if not known:
        return ""
    return _format_float(_rate(known))


def _pass_at_k_value(num_samples: int, num_correct: int, k: int) -> str:
    if k > num_samples:
        return "N/A"
    return _format_float(estimate_pass_at_k(num_samples, num_correct, k))


def _observed_pass_at_k_value(successes: list[bool], k: int) -> str:
    if k > len(successes):
        return "N/A"
    return str(observed_pass_at_k(successes, k))


def _metric_num_correct(values: Sequence[bool | None]) -> str:
    known = [value for value in values if value is not None]
    if not known:
        return ""
    return str(sum(1 for value in known if value))


def _metric_pass_at_k(values: Sequence[bool | None], k: int) -> str:
    known = [bool(value) for value in values if value is not None]
    if not known or k > len(known):
        return "N/A"
    return _format_float(estimate_pass_at_k(len(known), sum(known), k))


def _metric_observed_pass_at_k(values: Sequence[bool | None], k: int) -> str:
    known = [bool(value) for value in values if value is not None]
    if not known or k > len(known):
        return "N/A"
    return str(observed_pass_at_k(known, k))


def _format_latency(samples: list[FullBenchSampleResult]) -> str:
    known = [
        sample.latency_seconds
        for sample in samples
        if sample.latency_seconds is not None
    ]
    if not known:
        return "N/A"
    return _format_float(sum(known) / len(known))


def _average_latency(samples: list[FullBenchSampleResult]) -> str:
    return _format_latency(samples)


def _method_uses_refinement(method: str) -> bool:
    return method in {"full_with_refinement", "mag_coder_full"}


def _metadata_bool(sample: FullBenchSampleResult, key: str) -> bool:
    metadata = sample.metadata.get("provider_metadata")
    if isinstance(metadata, dict):
        return bool(metadata.get(key, False))
    generation_metadata = _load_generation_metadata(sample)
    return bool(generation_metadata.get(key, False))


def _load_generation_metadata(sample: FullBenchSampleResult) -> dict[str, Any]:
    path = Path(sample.generation_result_path)
    if not path.exists():
        return {}
    try:
        data = _load_json(path)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    metadata = data.get("metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _sample_provider_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "provider",
        "model_name",
        "final_content_empty",
        "extracted_code_non_empty",
        "extraction_strategy_used",
        "reasoning_present",
        "reasoning_may_contain_code",
        "require_non_empty_code_failed",
        "latency_seconds",
        "cache_hit",
    ]
    return {key: metadata.get(key) for key in keys if key in metadata}


def _join_nonempty(values: Any) -> str:
    items = sorted({str(value) for value in values if value})
    return "; ".join(items)


def _first_nonempty(values: Any) -> str:
    for value in values:
        if value:
            return str(value)
    return ""
