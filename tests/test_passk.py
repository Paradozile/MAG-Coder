import json
from pathlib import Path

import pytest

from magcoder.benchmark import build_mini_mag_bench
from magcoder.evaluation.multisample import (
    MultiSamplePassKRunner,
    MultiSampleRunConfig,
)
from magcoder.evaluation.passk import estimate_pass_at_k, observed_pass_at_k
from magcoder.generation import (
    FakeCodeGenerator,
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
)
from magcoder.generation.errors import MissingAPIKeyError


def _config(
    tmp_path: Path,
    *,
    provider: str = "fake",
    model_name: str = "fake-generator",
    output_name: str = "passk",
    num_samples: int = 2,
    api_base_url: str | None = None,
    api_key_env: str | None = None,
) -> MultiSampleRunConfig:
    return MultiSampleRunConfig(
        run_id=f"{provider}_passk_test",
        benchmark_path=str(tmp_path / "mini_mag_bench"),
        provider=provider,
        model_name=model_name,
        api_base_url=api_base_url,
        api_key_env=api_key_env,
        base_seed=100,
        num_samples=num_samples,
        k_values=[1, num_samples],
        input_modes=["structured"],
        ablation_ids=["graph_nl_without_refinement"],
        output_dir=str(tmp_path / output_name),
        metadata={"cache_dir": str(tmp_path / "generation_cache")},
    )


def test_passk_estimator_returns_zero_when_no_correct() -> None:
    assert estimate_pass_at_k(num_samples=10, num_correct=0, k=5) == 0.0


def test_passk_estimator_returns_one_when_k_covers_failed_samples() -> None:
    assert estimate_pass_at_k(num_samples=10, num_correct=1, k=10) == 1.0


def test_observed_pass_at_k_is_correct() -> None:
    assert not observed_pass_at_k([False, False, True], 2)
    assert observed_pass_at_k([False, False, True], 3)


def test_passk_estimator_and_observed_support_5_and_10() -> None:
    assert estimate_pass_at_k(num_samples=10, num_correct=2, k=5) > 0.0
    assert estimate_pass_at_k(num_samples=10, num_correct=2, k=10) == 1.0
    assert not observed_pass_at_k([False, False, False, False, True], 4)
    assert observed_pass_at_k([False, False, False, False, True], 5)


def test_fake_provider_multi_sample_run_completes(tmp_path: Path) -> None:
    package = build_mini_mag_bench(tmp_path / "mini_mag_bench")

    result = MultiSamplePassKRunner().run(package, _config(tmp_path))

    assert len(result.sample_results) == 4
    assert result.summary_rows
    assert all(sample.generation_error is None for sample in result.sample_results)


def test_dry_run_provider_does_not_call_external_apis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = build_mini_mag_bench(tmp_path / "mini_mag_bench")

    def fail_urlopen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry_run should not call urlopen")

    monkeypatch.setattr(
        "magcoder.generation.openai_compatible.urllib.request.urlopen",
        fail_urlopen,
    )

    result = MultiSamplePassKRunner().run(
        package,
        _config(
            tmp_path,
            provider="dry_run",
            model_name="dry-run-model",
            output_name="dry_run",
            num_samples=1,
        ),
    )

    assert len(result.sample_results) == 2
    assert all(sample.generation_error is None for sample in result.sample_results)


def test_sample_index_and_seed_enter_metadata_or_cache_key(tmp_path: Path) -> None:
    package = build_mini_mag_bench(tmp_path / "mini_mag_bench")

    result = MultiSamplePassKRunner().run(package, _config(tmp_path))
    robot_samples = [
        sample for sample in result.sample_results if sample.task_id == "robot_rescue"
    ]
    generation_payloads = [
        json.loads(Path(sample.generation_result_path).read_text(encoding="utf-8"))
        for sample in robot_samples
    ]

    assert [sample.seed for sample in robot_samples] == [100, 101]
    assert generation_payloads[0]["metadata"]["config"]["seed"] == 100
    assert generation_payloads[1]["metadata"]["config"]["seed"] == 101
    assert (
        generation_payloads[0]["metadata"]["request_hash"]
        != generation_payloads[1]["metadata"]["request_hash"]
    )


def test_one_failed_sample_does_not_stop_full_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = build_mini_mag_bench(tmp_path / "mini_mag_bench")
    calls = {"count": 0}

    def flaky_generate_with_provider(
        request: GenerationRequest,
        *,
        cache_dir: str | Path,
    ) -> GenerationResult:
        calls["count"] += 1
        if calls["count"] == 1:
            raise MissingAPIKeyError("simulated missing key")
        request.config = GenerationConfig(provider="fake", model_name="fake-generator")
        return FakeCodeGenerator().generate(request)

    monkeypatch.setattr(
        "magcoder.evaluation.multisample.generate_with_provider",
        flaky_generate_with_provider,
    )

    result = MultiSamplePassKRunner().run(package, _config(tmp_path))

    assert len(result.sample_results) == 4
    assert any(sample.generation_error is not None for sample in result.sample_results)
    assert any(sample.generation_error is None for sample in result.sample_results)


def test_passk_summary_csv_and_markdown_are_generated(tmp_path: Path) -> None:
    package = build_mini_mag_bench(tmp_path / "mini_mag_bench")

    MultiSamplePassKRunner().run(package, _config(tmp_path))

    assert (tmp_path / "passk" / "passk_summary.csv").exists()
    assert (tmp_path / "passk" / "passk_summary.md").exists()
    assert (tmp_path / "passk" / "failure_summary.md").exists()


def test_cache_hit_is_recorded(tmp_path: Path) -> None:
    package = build_mini_mag_bench(tmp_path / "mini_mag_bench")
    first_config = _config(tmp_path, output_name="cache_first")
    second_config = _config(tmp_path, output_name="cache_second")

    MultiSamplePassKRunner().run(package, first_config)
    second_result = MultiSamplePassKRunner().run(package, second_config)

    assert second_result.sample_results
    assert all(sample.cache_hit for sample in second_result.sample_results)


def test_api_keys_are_not_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = build_mini_mag_bench(tmp_path / "mini_mag_bench")
    token = "-".join(["mock", "passk", "token"])
    monkeypatch.setenv("MAGCODER_PASSK_KEY", token)

    class MockHTTPResponse:
        def __enter__(self) -> object:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"print(1)"}}]}'

    monkeypatch.setattr(
        "magcoder.generation.openai_compatible.urllib.request.urlopen",
        lambda *_args, **_kwargs: MockHTTPResponse(),
    )

    MultiSamplePassKRunner().run(
        package,
        _config(
            tmp_path,
            provider="openai_compatible",
            model_name="mock-model",
            output_name="secret_check",
            num_samples=1,
            api_base_url="https://example.test/v1",
            api_key_env="MAGCODER_PASSK_KEY",
        ),
    )

    output_text = (tmp_path / "secret_check").read_text(
        encoding="utf-8",
        errors="ignore",
    ) if (tmp_path / "secret_check").is_file() else ""
    combined = output_text + "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "secret_check").rglob("*.json")
    )
    assert token not in combined
    assert "MAGCODER_PASSK_KEY" in combined
