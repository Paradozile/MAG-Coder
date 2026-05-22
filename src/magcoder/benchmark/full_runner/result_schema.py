"""Result schemas for Full MAG-Bench pass@k runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class FullBenchSampleResult:
    """Result metadata for one Full MAG-Bench generated sample."""

    case_id: str
    input_mode: str
    method: str
    sample_index: int
    seed: int
    generation_result_path: str
    code_initial_path: str
    initial_report_path: str
    evaluation_path: str
    parse_success: bool
    framework_passed: bool
    mandatory_success: bool
    checkpoint_accuracy: float
    success: bool
    cache_hit: bool
    contract_pass: bool = False
    functional_pass: bool | None = None
    evaluation_mode: str = "contract"
    functional_error_type: str | None = None
    functional_error_message: str | None = None
    entrypoint_mode: str | None = None
    adapter_used: bool | None = None
    adapter_success: bool | None = None
    adapter_error_type: str | None = None
    adapter_error_message: str | None = None
    direct_entrypoint_present: bool | None = None
    required_candidate_symbols_present: bool | None = None
    abms_supported: bool | None = None
    abms_pass: bool | None = None
    abms_error_type: str | None = None
    abms_error_message: str | None = None
    abms_interaction_calls_executed: int | None = None
    abms_required_symbols_present: bool | None = None
    code_refined_path: str | None = None
    refined_report_path: str | None = None
    generation_error: str | None = None
    latency_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this sample result as JSON-compatible containers."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FullBenchSampleResult:
        """Build a sample result from JSON-compatible containers."""

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("FullBenchSampleResult metadata must be an object.")
        return cls(
            case_id=str(data["case_id"]),
            input_mode=str(data["input_mode"]),
            method=str(data["method"]),
            sample_index=int(data["sample_index"]),
            seed=int(data["seed"]),
            generation_result_path=str(data["generation_result_path"]),
            code_initial_path=str(data["code_initial_path"]),
            code_refined_path=_optional_str(data.get("code_refined_path")),
            initial_report_path=str(data["initial_report_path"]),
            refined_report_path=_optional_str(data.get("refined_report_path")),
            evaluation_path=str(data["evaluation_path"]),
            parse_success=bool(data["parse_success"]),
            framework_passed=bool(data["framework_passed"]),
            mandatory_success=bool(data["mandatory_success"]),
            checkpoint_accuracy=float(data["checkpoint_accuracy"]),
            success=bool(data["success"]),
            cache_hit=bool(data["cache_hit"]),
            contract_pass=bool(data.get("contract_pass", data["success"])),
            functional_pass=(
                None
                if data.get("functional_pass") is None
                else bool(data["functional_pass"])
            ),
            evaluation_mode=str(data.get("evaluation_mode", "contract")),
            functional_error_type=_optional_str(data.get("functional_error_type")),
            functional_error_message=_optional_str(
                data.get("functional_error_message")
            ),
            entrypoint_mode=_optional_str(data.get("entrypoint_mode")),
            adapter_used=_optional_bool(data.get("adapter_used")),
            adapter_success=_optional_bool(data.get("adapter_success")),
            adapter_error_type=_optional_str(data.get("adapter_error_type")),
            adapter_error_message=_optional_str(data.get("adapter_error_message")),
            direct_entrypoint_present=_optional_bool(
                data.get("direct_entrypoint_present")
            ),
            required_candidate_symbols_present=_optional_bool(
                data.get("required_candidate_symbols_present")
            ),
            abms_supported=_optional_bool(data.get("abms_supported")),
            abms_pass=_optional_bool(data.get("abms_pass")),
            abms_error_type=_optional_str(data.get("abms_error_type")),
            abms_error_message=_optional_str(data.get("abms_error_message")),
            abms_interaction_calls_executed=(
                None
                if data.get("abms_interaction_calls_executed") is None
                else int(data["abms_interaction_calls_executed"])
            ),
            abms_required_symbols_present=_optional_bool(
                data.get("abms_required_symbols_present")
            ),
            generation_error=_optional_str(data.get("generation_error")),
            latency_seconds=(
                None
                if data.get("latency_seconds") is None
                else float(data["latency_seconds"])
            ),
            metadata=dict(metadata),
        )


@dataclass(slots=True)
class FullBenchPassKConfig:
    """Configuration for a Full MAG-Bench pass@k smoke/evaluation run."""

    bench_path: str
    output_dir: str
    provider: str = "fake"
    model_name: str = "fake-generator"
    samples_per_case: int = 1
    k_values: list[int] = field(default_factory=lambda: [1])
    modes: list[str] = field(default_factory=lambda: ["structured_graph_nl"])
    methods: list[str] = field(default_factory=lambda: ["full_with_refinement"])
    max_cases: int | None = None
    overwrite: bool = False
    resume: bool = True
    base_seed: int = 7000
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int | None = None
    timeout_seconds: int = 60
    api_base_url: str | None = None
    api_key_env: str | None = None
    prompt_profile_id: str | None = None
    prompt_profile_path: str | None = None
    require_non_empty_code: bool = False
    think: bool | None = None
    native_endpoint: str | None = None
    num_predict: int | None = None
    cache_enabled: bool = False
    success_criterion: str = "mandatory_success"
    evaluation: str = "contract"
    split: str = "all"
    case_list_path: str | None = None
    refinement_strategy: str = "abms_focused"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this config as JSON/YAML-compatible containers."""

        return asdict(self)


@dataclass(slots=True)
class FullBenchPassKResult:
    """Run result bundle for Full MAG-Bench pass@k."""

    sample_results: list[FullBenchSampleResult]
    summary_rows: list[dict[str, Any]]
    output_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)
