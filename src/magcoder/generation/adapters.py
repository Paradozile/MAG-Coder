"""Local generation adapters that do not call external services."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from magcoder.generation.core import (
    GenerationRequest,
    GenerationResult,
    build_generation_metadata,
)


def build_chat_completion_payload(request: GenerationRequest) -> dict[str, Any]:
    """Build a generic chat-completion style payload without secrets."""

    config = request.config
    profile = config.metadata.get("prompt_profile")
    profile_system_message = (
        str(profile["system_message"])
        if isinstance(profile, dict) and isinstance(profile.get("system_message"), str)
        else str(
            config.metadata.get(
                "prompt_profile_system_message",
                (
                    "Generate Python interaction code from the provided "
                    "MAG-Coder hybrid semantic input. Return parseable Python."
                ),
            )
        )
    )
    payload: dict[str, Any] = {
        "model": config.model_name,
        "messages": [
            {
                "role": "system",
                "content": profile_system_message,
            },
            {"role": "user", "content": request.prompt_text},
        ],
        "temperature": config.temperature,
    }
    if config.top_p is not None:
        payload["top_p"] = config.top_p
    if config.max_tokens is not None:
        payload["max_tokens"] = config.max_tokens
    if config.seed is not None:
        payload["seed"] = config.seed
    if config.stop is not None:
        payload["stop"] = config.stop
    return payload


class DryRunGenerator:
    """Generator that records the would-be HTTP payload without calling an API."""

    name = "dry-run-generator"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return a dry-run result with request payload metadata."""

        started = perf_counter()
        payload = build_chat_completion_payload(request)
        raw_response = json.dumps(
            {
                "dry_run": True,
                "provider": request.config.provider,
                "request_payload": payload,
            },
            indent=2,
            sort_keys=True,
        )
        latency_seconds = perf_counter() - started
        return GenerationResult(
            prompt_text=request.prompt_text,
            raw_response=raw_response,
            code_initial="# Dry run: no code generated.\n",
            generator_name=self.name,
            metadata=build_generation_metadata(
                request,
                provider="dry_run",
                model_name=request.config.model_name,
                cache_hit=False,
                latency_seconds=latency_seconds,
                extra={
                    "dry_run": True,
                    "request_payload": payload,
                    "source": "dry_run_generator",
                },
            ),
            warnings=["Dry run completed without calling an external API."],
        )
