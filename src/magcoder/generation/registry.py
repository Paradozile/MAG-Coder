"""Provider registry for generation backends."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from magcoder.generation.adapters import DryRunGenerator
from magcoder.generation.cache import (
    DEFAULT_GENERATION_CACHE_DIR,
    load_cached_generation_result,
    save_cache_index_record,
    save_cached_generation_result,
)
from magcoder.generation.core import (
    CodeGenerator,
    FakeCodeGenerator,
    GenerationRequest,
    GenerationResult,
)
from magcoder.generation.errors import GeneratorConfigError
from magcoder.generation.ollama_native import OllamaNativeGenerator
from magcoder.generation.openai_compatible import OpenAICompatibleGenerator
from magcoder.generation.prompt_wrappers import apply_generation_prompt_profile

GeneratorFactory = Callable[[], CodeGenerator]
GeneratorProvider = str

_GENERATOR_FACTORIES: dict[str, GeneratorFactory] = {
    "fake": FakeCodeGenerator,
    "dry_run": DryRunGenerator,
    "http": OpenAICompatibleGenerator,
    "ollama_native": OllamaNativeGenerator,
    "openai_compatible": OpenAICompatibleGenerator,
}


def available_providers() -> list[str]:
    """Return registered provider names."""

    return sorted(_GENERATOR_FACTORIES)


def register_generator_provider(
    provider: str,
    factory: GeneratorFactory,
) -> None:
    """Register or replace a generator provider factory."""

    _GENERATOR_FACTORIES[provider] = factory


def create_generator(provider: str) -> CodeGenerator:
    """Create a generator for a provider name."""

    try:
        return _GENERATOR_FACTORIES[provider]()
    except KeyError as error:
        supported = ", ".join(available_providers())
        raise GeneratorConfigError(
            f"Unknown generator provider {provider!r}. Supported: {supported}."
        ) from error


def generate_with_provider(
    request: GenerationRequest,
    *,
    cache_dir: str | Path = DEFAULT_GENERATION_CACHE_DIR,
) -> GenerationResult:
    """Generate using request.config.provider with optional file caching."""

    request = apply_generation_prompt_profile(request)
    if request.config.cache_enabled:
        cached = load_cached_generation_result(request, cache_dir=cache_dir)
        if cached is not None:
            return cached

    provider = "dry_run" if request.config.dry_run else request.config.provider
    generator = create_generator(provider)
    result = generator.generate(request)

    if request.config.cache_enabled:
        save_cached_generation_result(result, request, cache_dir=cache_dir)
        save_cache_index_record(result=result, request=request, cache_dir=cache_dir)
    return result
