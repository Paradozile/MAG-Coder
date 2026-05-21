"""Code generation interfaces."""

from magcoder.generation.adapters import DryRunGenerator, build_chat_completion_payload
from magcoder.generation.cache import (
    DEFAULT_GENERATION_CACHE_DIR,
    generation_cache_key,
    generation_cache_path,
    load_cached_generation_result,
    save_cached_generation_result,
)
from magcoder.generation.core import (
    CodeExtractionResult,
    CodeGenerator,
    FakeCodeGenerator,
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    build_generation_metadata,
    extract_python_code,
    extract_python_code_with_diagnostics,
    load_generation_result,
    prompt_hash,
    request_hash,
    sanitized_generation_config,
    save_generation_result,
)
from magcoder.generation.errors import (
    GenerationError,
    GeneratorConfigError,
    GeneratorHTTPError,
    MissingAPIKeyError,
)
from magcoder.generation.ollama_native import (
    OllamaNativeGenerator,
    build_ollama_chat_payload,
    extract_ollama_native_text,
)
from magcoder.generation.openai_compatible import OpenAICompatibleGenerator
from magcoder.generation.prompt_profile import (
    PromptProfile,
    find_prompt_profile_path,
    load_prompt_profile,
    load_prompt_profile_by_id,
)
from magcoder.generation.prompt_wrappers import (
    apply_generation_prompt_profile,
    apply_prompt_profile,
    resolve_prompt_profile,
)
from magcoder.generation.registry import (
    available_providers,
    create_generator,
    generate_with_provider,
    register_generator_provider,
)

__all__ = [
    "CodeGenerator",
    "CodeExtractionResult",
    "DEFAULT_GENERATION_CACHE_DIR",
    "DryRunGenerator",
    "FakeCodeGenerator",
    "GenerationConfig",
    "GenerationError",
    "GenerationRequest",
    "GenerationResult",
    "GeneratorConfigError",
    "GeneratorHTTPError",
    "MissingAPIKeyError",
    "OpenAICompatibleGenerator",
    "OllamaNativeGenerator",
    "PromptProfile",
    "apply_generation_prompt_profile",
    "apply_prompt_profile",
    "available_providers",
    "build_chat_completion_payload",
    "build_generation_metadata",
    "build_ollama_chat_payload",
    "create_generator",
    "extract_python_code",
    "extract_ollama_native_text",
    "extract_python_code_with_diagnostics",
    "find_prompt_profile_path",
    "generate_with_provider",
    "generation_cache_key",
    "generation_cache_path",
    "load_cached_generation_result",
    "load_generation_result",
    "load_prompt_profile",
    "load_prompt_profile_by_id",
    "prompt_hash",
    "register_generator_provider",
    "resolve_prompt_profile",
    "request_hash",
    "sanitized_generation_config",
    "save_cached_generation_result",
    "save_generation_result",
]
