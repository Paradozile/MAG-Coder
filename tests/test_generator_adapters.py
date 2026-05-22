import json
from pathlib import Path
from typing import Any

import pytest

from magcoder.generation import (
    DryRunGenerator,
    GenerationConfig,
    GenerationRequest,
    MissingAPIKeyError,
    OpenAICompatibleGenerator,
    generate_with_provider,
    generation_cache_key,
)
from magcoder.generation.openai_compatible import extract_openai_compatible_text
from magcoder.models import load_mag_from_yaml
from magcoder.semantic import build_hybrid_input

ROOT = Path(__file__).resolve().parents[1]
ROBOT_MAG = ROOT / "examples" / "robot_rescue_mag.yaml"


def _request(config: GenerationConfig | None = None) -> GenerationRequest:
    mag = load_mag_from_yaml(ROBOT_MAG)
    hybrid_input = build_hybrid_input(mag)
    return GenerationRequest(
        hybrid_input=hybrid_input,
        config=config or GenerationConfig(),
        prompt_text=hybrid_input.to_prompt_text(),
    )


def test_fake_provider_still_works() -> None:
    result = generate_with_provider(_request(GenerationConfig(provider="fake")))

    assert result.generator_name == "fake-generator"
    assert result.metadata["provider"] == "fake"
    assert "class SonarDetection" in result.code_initial


def test_dry_run_provider_does_not_call_external_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_urlopen(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("dry_run should not call urlopen")

    monkeypatch.setattr(
        "magcoder.generation.openai_compatible.urllib.request.urlopen",
        fail_urlopen,
    )

    result = DryRunGenerator().generate(
        _request(GenerationConfig(provider="dry_run", model_name="dry-run-model")),
    )

    assert result.metadata["dry_run"] is True
    assert result.code_initial == "# Dry run: no code generated.\n"


def test_dry_run_result_contains_request_payload_metadata() -> None:
    result = generate_with_provider(
        _request(GenerationConfig(provider="dry_run", model_name="dry-run-model")),
    )

    payload = result.metadata["request_payload"]
    assert payload["model"] == "dry-run-model"
    assert payload["messages"][1]["content"] == result.prompt_text


def test_cache_key_generation_is_stable() -> None:
    request = _request(GenerationConfig(provider="fake", cache_enabled=True))

    assert generation_cache_key(request) == generation_cache_key(request)


def test_cache_hit_returns_existing_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(GenerationConfig(provider="fake", cache_enabled=True))
    first_result = generate_with_provider(request, cache_dir=tmp_path)

    def fail_create_generator(_provider: str) -> None:
        raise AssertionError("cache hit should not create a generator")

    monkeypatch.setattr(
        "magcoder.generation.registry.create_generator",
        fail_create_generator,
    )
    second_result = generate_with_provider(request, cache_dir=tmp_path)

    assert first_result.code_initial == second_result.code_initial
    assert second_result.metadata["cache_hit"] is True


def test_openai_compatible_adapter_works_with_mocked_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "-".join(["mock", "token", "value"])
    monkeypatch.setenv("MAGCODER_TEST_KEY", token)

    class MockHTTPResponse:
        def __enter__(self) -> object:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "```python\nresult = 'ok'\n```",
                            },
                        },
                    ],
                },
            ).encode("utf-8")

    def mock_urlopen(_request: object, *, timeout: int) -> MockHTTPResponse:
        assert timeout == 60
        return MockHTTPResponse()

    monkeypatch.setattr(
        "magcoder.generation.openai_compatible.urllib.request.urlopen",
        mock_urlopen,
    )
    request = _request(
        GenerationConfig(
            provider="openai_compatible",
            model_name="mock-model",
            api_base_url="https://example.test/v1",
            api_key_env="MAGCODER_TEST_KEY",
        ),
    )

    result = OpenAICompatibleGenerator().generate(request)

    assert result.metadata["provider"] == "openai_compatible"
    assert result.code_initial == "result = 'ok'\n"
    assert result.metadata["response_schema_summary"]["choices_count"] == 1


def test_openai_response_parser_falls_back_to_choice_text() -> None:
    extraction = extract_openai_compatible_text(
        json.dumps({"choices": [{"text": "print('fallback')"}]}),
    )

    assert extraction.text == "print('fallback')"
    assert extraction.schema_summary["extracted_field"] == "choices[0].text"
    assert extraction.schema_summary["fallback_used"] is True
    assert extraction.warnings


def test_openai_response_parser_records_reasoning_without_using_it() -> None:
    extraction = extract_openai_compatible_text(
        json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "def add(a, b): return a + b",
                        }
                    }
                ]
            }
        ),
    )

    assert extraction.text == ""
    assert extraction.schema_summary["reasoning_content_present"] is True
    assert "reasoning_content" in extraction.warnings[0]


def test_openai_response_parser_records_ollama_reasoning_without_using_it() -> None:
    extraction = extract_openai_compatible_text(
        json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "reasoning": "thinking text",
                            "content": "",
                        }
                    }
                ]
            }
        ),
    )

    assert extraction.text == ""
    assert extraction.schema_summary["reasoning_present"] is True
    assert "message.reasoning" in extraction.warnings[0]


def test_openai_response_parser_falls_back_to_output_text() -> None:
    extraction = extract_openai_compatible_text(
        json.dumps({"output_text": "print('output')"}),
    )

    assert extraction.text == "print('output')"
    assert extraction.schema_summary["extracted_field"] == "output_text"


def test_openai_response_parser_falls_back_to_top_level_content() -> None:
    extraction = extract_openai_compatible_text(
        json.dumps({"content": "print('content')"}),
    )

    assert extraction.text == "print('content')"
    assert extraction.schema_summary["extracted_field"] == "content"


def test_api_key_is_not_saved_into_generation_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "-".join(["mock", "token", "value"])
    monkeypatch.setenv("MAGCODER_TEST_KEY", token)

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
    result = OpenAICompatibleGenerator().generate(
        _request(
            GenerationConfig(
                provider="openai_compatible",
                model_name="mock-model",
                api_base_url="https://example.test/v1",
                api_key_env="MAGCODER_TEST_KEY",
            ),
        ),
    )

    persisted = json.dumps(result.to_dict(), sort_keys=True)
    assert token not in persisted
    assert "MAGCODER_TEST_KEY" in persisted


def test_missing_api_key_raises_clear_error() -> None:
    request = _request(
        GenerationConfig(
            provider="openai_compatible",
            model_name="mock-model",
            api_base_url="https://example.test/v1",
            api_key_env="MISSING_MAGCODER_TEST_KEY",
        ),
    )

    with pytest.raises(MissingAPIKeyError, match="MISSING_MAGCODER_TEST_KEY"):
        OpenAICompatibleGenerator().generate(request)
