"""Run pass@k over a generated Full MAG-Bench v1 package."""

from __future__ import annotations

import argparse
from pathlib import Path

from magcoder.benchmark.full_runner import (
    FullBenchPassKConfig,
    FullMagBenchPassKRunner,
)


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Run Full MAG-Bench v1 through generation/refinement/evaluation.",
    )
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--provider", default="fake")
    parser.add_argument("--model-name", "--model", dest="model_name", default=None)
    parser.add_argument("--samples-per-case", type=int, default=1)
    parser.add_argument(
        "--k-values",
        default=None,
        help="Comma-separated k values. Defaults to supported paper k values.",
    )
    parser.add_argument(
        "--modes",
        default="structured_graph_nl",
        help=(
            "Comma-separated modes: structured_graph_only, structured_graph_nl, "
            "graphical_graph_only, graphical_graph_nl, zero_shot_direct, "
            "few_shot_direct, without_nl."
        ),
    )
    parser.add_argument(
        "--methods",
        default="full_with_refinement",
        help=(
            "Comma-separated methods: full_with_refinement,no_refinement,"
            "mag_coder_full,without_ast."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-benchmark", type=Path, default=None)
    parser.add_argument(
        "--evaluation",
        choices=["contract", "functional", "both", "abms", "all"],
        default="contract",
        help="Evaluation mode: contract, functional, both, abms, or all.",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all", "none"],
        default="all",
    )
    parser.add_argument("--case-list", type=Path, default=None)
    parser.add_argument(
        "--refinement-strategy",
        choices=["abms_focused", "balanced", "conservative"],
        default="abms_focused",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--base-seed", type=int, default=7000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--prompt-profile-id", default=None)
    parser.add_argument("--prompt-profile-path", "--prompt-profile", default=None)
    parser.add_argument("--require-non-empty-code", action="store_true")
    parser.add_argument("--think", choices=["true", "false"], default=None)
    parser.add_argument("--native-endpoint", default=None)
    parser.add_argument("--num-predict", type=int, default=None)
    args = parser.parse_args()

    provider_defaults = _provider_defaults(args.provider)
    model_name = args.model_name or provider_defaults["model_name"]
    api_base_url = args.api_base_url or provider_defaults["api_base_url"]
    native_endpoint = args.native_endpoint or provider_defaults["native_endpoint"]
    prompt_profile_path = (
        args.prompt_profile_path or provider_defaults["prompt_profile_path"]
    )
    think = _parse_optional_bool(args.think, provider_defaults["think"])
    timeout_seconds = (
        args.timeout_seconds
        if args.timeout_seconds != 60
        else int(provider_defaults["timeout_seconds"])
    )
    max_tokens = (
        args.max_tokens
        if args.max_tokens is not None
        else provider_defaults["max_tokens"]
    )
    num_predict = (
        args.num_predict
        if args.num_predict is not None
        else provider_defaults["num_predict"]
    )
    k_values = (
        _parse_int_list(args.k_values)
        if args.k_values
        else _default_k_values(args.samples_per_case)
    )
    config = FullBenchPassKConfig(
        bench_path=str(args.bench),
        output_dir=str(args.output),
        provider=args.provider,
        model_name=str(model_name),
        samples_per_case=args.samples_per_case,
        k_values=k_values,
        modes=_parse_str_list(args.modes),
        methods=_parse_str_list(args.methods),
        max_cases=args.max_cases,
        split=args.split,
        case_list_path=None if args.case_list is None else str(args.case_list),
        refinement_strategy=args.refinement_strategy,
        overwrite=args.overwrite,
        resume=args.resume,
        base_seed=args.base_seed,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        api_base_url=api_base_url,
        api_key_env=args.api_key_env,
        prompt_profile_id=args.prompt_profile_id,
        prompt_profile_path=prompt_profile_path,
        require_non_empty_code=(
            args.require_non_empty_code
            or bool(provider_defaults["require_non_empty_code"])
        ),
        think=think,
        native_endpoint=native_endpoint,
        num_predict=num_predict,
        evaluation=args.evaluation,
        metadata={
            "frozen_benchmark_path": (
                None if args.frozen_benchmark is None else str(args.frozen_benchmark)
            ),
        },
    )
    result = FullMagBenchPassKRunner().run(config)
    print(f"Full MAG-Bench pass@k output: {args.output}")
    print(f"Samples: {len(result.sample_results)}")
    print(f"Summary CSV: {args.output / 'summary.csv'}")
    print(f"Report: {args.output / 'report.md'}")


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in _parse_str_list(value)]


def _default_k_values(samples_per_case: int) -> list[int]:
    """Return paper-friendly default k values for the requested sample count."""

    return [k for k in [1, 3, 5, 10, 50, 100] if k <= samples_per_case]


def _parse_optional_bool(value: str | None, default: bool | None) -> bool | None:
    if value is None:
        return default
    return value.lower() == "true"


def _provider_defaults(provider: str) -> dict[str, object]:
    if provider == "ollama_native":
        return {
            "model_name": "qwen3.5:9b",
            "api_base_url": "http://127.0.0.1:11434",
            "native_endpoint": "/api/chat",
            "think": False,
            "prompt_profile_path": (
                "configs/prompt_profiles/ollama_qwen_contract_compact.yaml"
            ),
            "require_non_empty_code": True,
            "timeout_seconds": 300,
            "max_tokens": 2048,
            "num_predict": 2048,
        }
    return {
        "model_name": "fake-generator",
        "api_base_url": None,
        "native_endpoint": None,
        "think": None,
        "prompt_profile_path": None,
        "require_non_empty_code": False,
        "timeout_seconds": 60,
        "max_tokens": None,
        "num_predict": None,
    }


if __name__ == "__main__":
    main()
