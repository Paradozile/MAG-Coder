"""Apply prompt profiles to generator requests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from magcoder.generation.core import GenerationRequest
from magcoder.generation.prompt_profile import (
    PromptProfile,
    load_prompt_profile,
    load_prompt_profile_by_id,
)


def apply_prompt_profile(prompt_text: str, profile: PromptProfile) -> str:
    """Wrap a prompt with a profile while preserving the original prompt."""

    parts = []
    if profile.user_prefix.strip():
        parts.append(profile.user_prefix.strip())
    parts.append(prompt_text.strip())
    if profile.user_suffix.strip():
        parts.append(profile.user_suffix.strip())
    if (
        profile.extraction_strategy == "begin_end_python"
        and "BEGIN_PYTHON" not in "\n".join(parts)
    ):
        parts.append(
            "Output must use exactly this envelope:\n"
            "BEGIN_PYTHON\n"
            "<valid Python code>\n"
            "END_PYTHON"
        )
    return "\n\n".join(parts).strip() + "\n"


def resolve_prompt_profile(request: GenerationRequest) -> PromptProfile | None:
    """Resolve the configured profile for a request, if any."""

    config = request.config
    if config.prompt_profile_path:
        return load_prompt_profile(config.prompt_profile_path)
    if config.prompt_profile_id:
        return load_prompt_profile_by_id(config.prompt_profile_id, root=Path("."))
    return None


def apply_generation_prompt_profile(request: GenerationRequest) -> GenerationRequest:
    """Return a request whose prompt has been wrapped by its prompt profile."""

    profile = resolve_prompt_profile(request)
    if profile is None:
        return request

    wrapped_prompt = apply_prompt_profile(request.prompt_text, profile)
    metadata = {
        **request.config.metadata,
        "prompt_profile": profile.to_dict(),
        "prompt_profile_system_message": profile.system_message,
        "prompt_profile_output_format": profile.output_format,
        "prompt_profile_extraction_strategy": profile.extraction_strategy,
    }
    config = replace(
        request.config,
        prompt_profile_id=profile.profile_id,
        metadata=metadata,
    )
    return GenerationRequest(
        hybrid_input=request.hybrid_input,
        config=config,
        prompt_text=wrapped_prompt,
    )
