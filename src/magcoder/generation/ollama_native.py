"""Ollama native generation adapter."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, cast

from magcoder.generation.adapters import build_chat_completion_payload
from magcoder.generation.core import (
    GenerationRequest,
    GenerationResult,
    build_generation_metadata,
    extract_python_code_with_diagnostics,
)
from magcoder.generation.errors import GeneratorConfigError, GeneratorHTTPError


@dataclass(slots=True)
class OllamaNativeExtraction:
    """Text and diagnostics extracted from an Ollama native response."""

    text: str
    schema_summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render this extraction as plain Python containers."""

        return asdict(self)


class OllamaNativeGenerator:
    """Generator that calls Ollama's native non-streaming API."""

    name = "ollama-native-generator"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Call Ollama /api/chat with think disabled when configured."""

        config = request.config
        endpoint_name = config.native_endpoint or "/api/chat"
        endpoint = _resolve_endpoint(config.api_base_url, endpoint_name)
        payload = build_ollama_chat_payload(request)
        started = perf_counter()
        response_body: str
        warnings: list[str] = []
        native_endpoint = endpoint_name
        try:
            response_body = _post_json(
                endpoint,
                payload,
                timeout_seconds=config.timeout_seconds,
            )
        except urllib.error.HTTPError as error:
            if endpoint_name == "/api/chat" and error.code == 404:
                native_endpoint = "/api/generate"
                warnings.append(
                    "/api/chat returned 404; fell back to /api/generate."
                )
                response_body = _post_json(
                    _resolve_endpoint(config.api_base_url, native_endpoint),
                    build_ollama_generate_payload(request),
                    timeout_seconds=config.timeout_seconds,
                )
            else:
                body = error.read().decode("utf-8", errors="replace")
                raise GeneratorHTTPError(
                    f"Ollama native generation failed with status "
                    f"{error.code}: {body}"
                ) from error
        except urllib.error.URLError as error:
            raise GeneratorHTTPError(
                f"Ollama native generation failed: {error.reason}"
            ) from error
        except TimeoutError as error:
            raise GeneratorHTTPError("Ollama native generation timed out.") from error

        latency_seconds = perf_counter() - started
        extraction = extract_ollama_native_text(response_body)
        code_extraction = extract_python_code_with_diagnostics(extraction.text)
        return GenerationResult(
            prompt_text=request.prompt_text,
            raw_response=extraction.text,
            code_initial=code_extraction.code,
            generator_name=self.name,
            metadata=build_generation_metadata(
                request,
                provider="ollama_native",
                model_name=config.model_name,
                cache_hit=False,
                latency_seconds=latency_seconds,
                extra={
                    "api_base_url": config.api_base_url,
                    "native_endpoint": native_endpoint,
                    "think": config.think,
                    "response_body": response_body,
                    "response_schema_summary": extraction.schema_summary,
                    "final_content_empty": not bool(extraction.text.strip()),
                    "reasoning_present": _reasoning_present(response_body),
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
                    "source": "ollama_native_generator",
                },
            ),
            warnings=[
                *warnings,
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


def build_ollama_chat_payload(request: GenerationRequest) -> dict[str, Any]:
    """Build a non-streaming Ollama /api/chat payload."""

    config = request.config
    chat_payload = build_chat_completion_payload(request)
    options: dict[str, Any] = {"temperature": config.temperature}
    if config.top_p is not None:
        options["top_p"] = config.top_p
    num_predict = (
        config.num_predict if config.num_predict is not None else config.max_tokens
    )
    if num_predict is not None:
        options["num_predict"] = num_predict
    return {
        "model": config.model_name,
        "messages": chat_payload["messages"],
        "stream": False,
        "think": False if config.think is None else config.think,
        "options": options,
    }


def build_ollama_generate_payload(request: GenerationRequest) -> dict[str, Any]:
    """Build a fallback non-streaming Ollama /api/generate payload."""

    config = request.config
    chat_payload = build_chat_completion_payload(request)
    prompt = "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}"
        for message in chat_payload["messages"]
    )
    options: dict[str, Any] = {"temperature": config.temperature}
    if config.top_p is not None:
        options["top_p"] = config.top_p
    num_predict = (
        config.num_predict if config.num_predict is not None else config.max_tokens
    )
    if num_predict is not None:
        options["num_predict"] = num_predict
    return {
        "model": config.model_name,
        "prompt": prompt,
        "stream": False,
        "think": False if config.think is None else config.think,
        "options": options,
    }


def extract_ollama_native_text(response_body: str) -> OllamaNativeExtraction:
    """Extract final content from an Ollama native response."""

    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        return OllamaNativeExtraction(
            text=response_body,
            schema_summary={
                "root_type": "non_json",
                "root_keys": [],
                "extracted_field": "raw_text",
                "fallback_used": False,
            },
        )
    if not isinstance(parsed, dict):
        return OllamaNativeExtraction(
            text=response_body,
            schema_summary={
                "root_type": type(parsed).__name__,
                "root_keys": [],
                "extracted_field": "raw_json",
                "fallback_used": True,
            },
            warnings=["Ollama response JSON root was not an object; using raw JSON."],
        )

    schema = _schema_summary(parsed)
    warnings: list[str] = []
    message = parsed.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content:
            schema["extracted_field"] = "message.content"
            schema["fallback_used"] = False
            return OllamaNativeExtraction(content, schema, warnings)
        for key in ("thinking", "reasoning", "reasoning_content"):
            if isinstance(message.get(key), str):
                warnings.append(
                    f"message.{key} was present but was not used as final code."
                )

    response = parsed.get("response")
    if isinstance(response, str) and response:
        schema["extracted_field"] = "response"
        schema["fallback_used"] = True
        warnings.append("Fell back to top-level response.")
        return OllamaNativeExtraction(response, schema, warnings)

    for key in ("thinking", "reasoning", "reasoning_content"):
        if isinstance(parsed.get(key), str):
            warnings.append(f"{key} was present but was not used as final code.")
    if isinstance(parsed.get("error"), str):
        warnings.append(f"Ollama returned error: {parsed['error']}")
    schema["extracted_field"] = None
    schema["fallback_used"] = False
    return OllamaNativeExtraction("", schema, warnings)


def _resolve_endpoint(api_base_url: str | None, native_endpoint: str) -> str:
    if not api_base_url:
        raise GeneratorConfigError("api_base_url is required for ollama_native.")
    base = api_base_url.rstrip("/")
    endpoint = (
        native_endpoint
        if native_endpoint.startswith("/")
        else f"/{native_endpoint}"
    )
    return f"{base}{endpoint}"


def _post_json(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: int,
) -> str:
    http_request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=timeout_seconds) as response:
        body = cast(bytes, response.read())
        return body.decode("utf-8")


def _schema_summary(parsed: dict[str, Any]) -> dict[str, Any]:
    message = parsed.get("message")
    message_keys = (
        sorted(str(key) for key in message)
        if isinstance(message, dict)
        else []
    )
    return {
        "root_type": "dict",
        "root_keys": sorted(str(key) for key in parsed),
        "message_keys": message_keys,
        "message_content_present": (
            isinstance(message, dict) and isinstance(message.get("content"), str)
        ),
        "response_present": isinstance(parsed.get("response"), str),
        "done": parsed.get("done"),
        "error_present": isinstance(parsed.get("error"), str),
        "reasoning_present": _reasoning_present_in_parsed(parsed),
        "extracted_field": None,
        "fallback_used": False,
    }


def _reasoning_present(response_body: str) -> bool:
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and _reasoning_present_in_parsed(parsed)


def _reasoning_present_in_parsed(parsed: dict[str, Any]) -> bool:
    message = parsed.get("message")
    if isinstance(message, dict) and any(
        isinstance(message.get(key), str)
        for key in ("thinking", "reasoning", "reasoning_content")
    ):
        return True
    return any(
        isinstance(parsed.get(key), str)
        for key in ("thinking", "reasoning", "reasoning_content")
    )


def _reasoning_may_contain_code(response_body: str) -> bool:
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False
    values: list[str] = []
    message = parsed.get("message")
    if isinstance(message, dict):
        values.extend(
            str(message[key])
            for key in ("thinking", "reasoning", "reasoning_content")
            if isinstance(message.get(key), str)
        )
    values.extend(
        str(parsed[key])
        for key in ("thinking", "reasoning", "reasoning_content")
        if isinstance(parsed.get(key), str)
    )
    return any(_looks_like_code(value) for value in values)


def _looks_like_code(text: str) -> bool:
    return any(marker in text for marker in ("```", "def ", "class ", "return "))
