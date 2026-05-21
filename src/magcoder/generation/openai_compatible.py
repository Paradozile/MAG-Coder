"""OpenAI-compatible HTTP generation adapter."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

from magcoder.generation.adapters import build_chat_completion_payload
from magcoder.generation.core import (
    GenerationRequest,
    GenerationResult,
    build_generation_metadata,
    extract_python_code_with_diagnostics,
)
from magcoder.generation.errors import (
    GeneratorConfigError,
    GeneratorHTTPError,
    MissingAPIKeyError,
)


@dataclass(slots=True)
class ResponseExtraction:
    """Text and schema diagnostics extracted from an OpenAI-compatible response."""

    text: str
    schema_summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render this extraction as plain Python containers."""

        return asdict(self)


class OpenAICompatibleGenerator:
    """Generic HTTP generator for OpenAI-style chat completion endpoints."""

    name = "openai-compatible-generator"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Call a configured OpenAI-compatible endpoint."""

        config = request.config
        endpoint = _resolve_endpoint(config.api_base_url)
        api_key = _read_api_key(config.api_key_env)
        payload = build_chat_completion_payload(request)
        request_bytes = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            endpoint,
            data=request_bytes,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started = perf_counter()
        try:
            with urllib.request.urlopen(
                http_request,
                timeout=config.timeout_seconds,
            ) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise GeneratorHTTPError(
                f"HTTP generation failed with status {error.code}: {body}"
            ) from error
        except urllib.error.URLError as error:
            raise GeneratorHTTPError(
                f"HTTP generation failed: {error.reason}"
            ) from error
        except TimeoutError as error:
            raise GeneratorHTTPError("HTTP generation timed out.") from error

        latency_seconds = perf_counter() - started
        extraction = extract_openai_compatible_text(response_body)
        code_extraction = extract_python_code_with_diagnostics(extraction.text)
        return GenerationResult(
            prompt_text=request.prompt_text,
            raw_response=extraction.text,
            code_initial=code_extraction.code,
            generator_name=self.name,
            metadata=build_generation_metadata(
                request,
                provider="openai_compatible",
                model_name=config.model_name,
                cache_hit=False,
                latency_seconds=latency_seconds,
                extra={
                    "api_base_url": config.api_base_url,
                    "api_key_env": config.api_key_env,
                    "response_body": response_body,
                    "response_schema_summary": extraction.schema_summary,
                    "final_content_empty": not bool(extraction.text.strip()),
                    "reasoning_present": bool(
                        extraction.schema_summary.get("reasoning_present")
                        or extraction.schema_summary.get(
                            "reasoning_content_present"
                        )
                    ),
                    "reasoning_may_contain_code": _reasoning_may_contain_code(
                        response_body
                    ),
                    "extraction_strategy_used": (
                        code_extraction.extraction_strategy_used
                    ),
                    "extracted_code_non_empty": (
                        code_extraction.extracted_code_non_empty
                    ),
                    "require_non_empty_code_failed": (
                        config.require_non_empty_code
                        and not code_extraction.extracted_code_non_empty
                    ),
                    "source": "openai_compatible_http_generator",
                },
            ),
            warnings=[
                *extraction.warnings,
                *code_extraction.warnings,
                *(
                    ["Final content did not contain non-empty Python code."]
                    if config.require_non_empty_code
                    and not code_extraction.extracted_code_non_empty
                    else []
                ),
            ],
        )


def _resolve_endpoint(api_base_url: str | None) -> str:
    if not api_base_url:
        raise GeneratorConfigError(
            "api_base_url is required for openai_compatible generation."
        )
    normalized = api_base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _read_api_key(api_key_env: str | None) -> str:
    if not api_key_env:
        raise MissingAPIKeyError(
            "api_key_env is required for openai_compatible generation."
        )
    value = os.environ.get(api_key_env)
    if not value:
        raise MissingAPIKeyError(
            f"API key environment variable {api_key_env!r} is not set."
        )
    return value


def extract_openai_compatible_text(response_body: str) -> ResponseExtraction:
    """Extract assistant text and response diagnostics from an HTTP response."""

    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        return ResponseExtraction(
            text=response_body,
            schema_summary={
                "root_type": "non_json",
                "root_keys": [],
                "choices_count": 0,
                "choice_visible_fields": [],
                "message_keys": [],
                "extracted_field": "raw_text",
                "fallback_used": False,
            },
        )

    if not isinstance(parsed, dict):
        return ResponseExtraction(
            text=response_body,
            schema_summary={
                "root_type": type(parsed).__name__,
                "root_keys": [],
                "choices_count": 0,
                "choice_visible_fields": [],
                "message_keys": [],
                "extracted_field": "raw_json",
                "fallback_used": True,
            },
            warnings=["Response JSON root was not an object; using raw response."],
        )

    schema = _response_schema_summary(parsed)
    warnings: list[str] = []

    choices = parsed.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content:
                    schema["extracted_field"] = "choices[0].message.content"
                    schema["fallback_used"] = False
                    return ResponseExtraction(content, schema, warnings)
                if isinstance(message.get("reasoning_content"), str):
                    schema["reasoning_content_present"] = True
                    warnings.append(
                        "choices[0].message.reasoning_content was present but was "
                        "not used as final code content."
                    )
                if isinstance(message.get("reasoning"), str):
                    schema["reasoning_present"] = True
                    warnings.append(
                        "choices[0].message.reasoning was present but was not used "
                        "as final code content."
                    )
                if isinstance(message.get("tool_calls"), list):
                    schema["tool_calls_present"] = True
                    warnings.append(
                        "choices[0].message.tool_calls was present but cannot be "
                        "extracted as Python code."
                    )
            if isinstance(first_choice.get("text"), str):
                schema["extracted_field"] = "choices[0].text"
                schema["fallback_used"] = True
                warnings.append("Fell back to choices[0].text.")
                return ResponseExtraction(str(first_choice["text"]), schema, warnings)

    output_text = parsed.get("output_text")
    if isinstance(output_text, str):
        schema["extracted_field"] = "output_text"
        schema["fallback_used"] = True
        warnings.append("Fell back to response.output_text.")
        return ResponseExtraction(output_text, schema, warnings)

    content = parsed.get("content")
    if isinstance(content, str):
        schema["extracted_field"] = "content"
        schema["fallback_used"] = True
        warnings.append("Fell back to response.content.")
        return ResponseExtraction(content, schema, warnings)

    schema["extracted_field"] = None
    schema["fallback_used"] = False
    return ResponseExtraction("", schema, warnings)


def _response_schema_summary(parsed: dict[str, Any]) -> dict[str, Any]:
    choices = parsed.get("choices")
    choice_visible_fields: list[list[str]] = []
    message_keys: list[list[str]] = []
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                choice_visible_fields.append(sorted(str(key) for key in choice))
                message = choice.get("message")
                if isinstance(message, dict):
                    message_keys.append(sorted(str(key) for key in message))
                else:
                    message_keys.append([])
            else:
                choice_visible_fields.append([])
                message_keys.append([])

    return {
        "root_type": "dict",
        "root_keys": sorted(str(key) for key in parsed),
        "choices_count": len(choices) if isinstance(choices, list) else 0,
        "choice_visible_fields": choice_visible_fields,
        "message_keys": message_keys,
        "output_text_present": isinstance(parsed.get("output_text"), str),
        "content_present": isinstance(parsed.get("content"), str),
        "reasoning_content_present": any(
            "reasoning_content" in keys for keys in message_keys
        ),
        "reasoning_present": any("reasoning" in keys for keys in message_keys),
        "tool_calls_present": any("tool_calls" in keys for keys in message_keys),
        "extracted_field": None,
        "fallback_used": False,
    }


def _extract_message_content(response_body: str) -> str:
    """Backward-compatible wrapper for older tests/imports."""

    return extract_openai_compatible_text(response_body).text


def _reasoning_may_contain_code(response_body: str) -> bool:
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False
    choices = parsed.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        for key in ("reasoning", "reasoning_content"):
            value = message.get(key)
            if isinstance(value, str) and _looks_like_code(value):
                return True
    return False


def _looks_like_code(text: str) -> bool:
    return any(marker in text for marker in ("```", "def ", "class ", "return "))
