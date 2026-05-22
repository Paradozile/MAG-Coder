"""Code generation interfaces and deterministic fake implementation."""

from __future__ import annotations

import ast
import hashlib
import json
import keyword
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from magcoder.semantic import HybridEdgeSemantics, HybridInput


@dataclass(slots=True)
class GenerationConfig:
    """Configuration for an interaction-code generation request."""

    model_name: str = "fake-generator"
    provider: str = "fake"
    api_base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int = 60
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    think: bool | None = None
    native_endpoint: str | None = None
    num_predict: int | None = None
    seed: int | None = None
    prompt_variant: str = "default"
    prompt_profile_id: str | None = None
    prompt_profile_path: str | None = None
    require_non_empty_code: bool = False
    input_mode: str = ""
    cache_enabled: bool = False
    dry_run: bool = False
    mode: str = "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationRequest:
    """Input passed to a code generator."""

    hybrid_input: HybridInput
    config: GenerationConfig
    prompt_text: str


@dataclass(slots=True)
class GenerationResult:
    """Initial code generation result."""

    prompt_text: str
    raw_response: str
    code_initial: str
    generator_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render this result as plain Python containers."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationResult:
        """Build a generation result from persisted JSON data."""

        metadata = data.get("metadata", {})
        warnings = data.get("warnings", [])
        if not isinstance(metadata, dict):
            raise TypeError("Generation result field 'metadata' must be a dict.")
        if not isinstance(warnings, list):
            raise TypeError("Generation result field 'warnings' must be a list.")
        return cls(
            prompt_text=str(data["prompt_text"]),
            raw_response=str(data["raw_response"]),
            code_initial=str(data["code_initial"]),
            generator_name=str(data["generator_name"]),
            metadata=dict(metadata),
            warnings=[str(warning) for warning in warnings],
        )


class CodeGenerator(Protocol):
    """Common interface for code generators."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate initial interaction code from a request."""


class FakeCodeGenerator:
    """Deterministic generator used for tests and Phase 4 input fixtures."""

    name = "fake-generator"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate parseable Python code without calling any external API."""

        started = perf_counter()
        code = self._generate_code(request.hybrid_input)
        raw_response = (
            "Fake generator response. No external LLM API was called.\n\n"
            f"```python\n{code}\n```"
        )
        warnings = _generation_warnings(request.hybrid_input)
        latency_seconds = perf_counter() - started
        return GenerationResult(
            prompt_text=request.prompt_text,
            raw_response=raw_response,
            code_initial=extract_python_code(raw_response),
            generator_name=self.name,
            metadata=build_generation_metadata(
                request,
                provider="fake",
                model_name=request.config.model_name,
                cache_hit=False,
                latency_seconds=latency_seconds,
                extra={
                    "generation_interface_version": 1,
                    "hybrid_input": request.hybrid_input.to_dict(),
                    "mag_id": request.hybrid_input.mag_id,
                    "mode": request.hybrid_input.mode,
                    "source": "deterministic_fake_generator",
                },
            ),
            warnings=warnings,
        )

    def _generate_code(self, hybrid_input: HybridInput) -> str:
        used_agent_names: set[str] = set()
        agent_names = {
            agent.agent_id: _unique_identifier(
                _to_class_name(agent.agent_id),
                used_agent_names,
            )
            for agent in hybrid_input.agents
        }
        used_filter_names: set[str] = set()
        filter_names = {
            attribute_filter.filter_id: _unique_identifier(
                f"_filter_{_to_snake_case(attribute_filter.filter_id)}",
                used_filter_names,
            )
            for attribute_filter in hybrid_input.filters
        }
        used_edge_names: set[str] = set()

        lines = [
            "from __future__ import annotations",
            "",
            "from typing import Any",
            "",
            f'MAG_ID = "{_escape_string(hybrid_input.mag_id)}"',
            "",
            "",
        ]

        for agent in hybrid_input.agents:
            class_name = agent_names[agent.agent_id]
            lines.extend(_render_agent_class(class_name, agent.to_dict()))

        lines.extend(_render_lookup_helper())

        for attribute_filter in hybrid_input.filters:
            function_name = filter_names[attribute_filter.filter_id]
            lines.extend(
                _render_filter_function(
                    function_name=function_name,
                    filter_id=attribute_filter.filter_id,
                    expressions=attribute_filter.original_expressions,
                )
            )

        for edge in hybrid_input.edges:
            function_name = _unique_identifier(
                f"interact_{_to_snake_case(edge.edge_id)}",
                used_edge_names,
            )
            lines.extend(
                _render_interaction_function(
                    edge=edge,
                    function_name=function_name,
                    source_class=agent_names[edge.source["agent_id"]],
                    target_class=agent_names[edge.target["agent_id"]],
                    filter_function=(
                        filter_names.get(edge.filter_id)
                        if edge.filter_id is not None
                        else None
                    ),
                )
            )

        return "\n".join(lines).rstrip() + "\n"


def prompt_hash(prompt_text: str) -> str:
    """Return a stable SHA-256 hash for prompt text."""

    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def request_hash(request: GenerationRequest) -> str:
    """Return a stable SHA-256 hash for request identity."""

    payload = {
        "config": sanitized_generation_config(request.config),
        "hybrid_input": request.hybrid_input.to_dict(),
        "prompt_hash": prompt_hash(request.prompt_text),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sanitized_generation_config(config: GenerationConfig) -> dict[str, Any]:
    """Render generation config without runtime-only or secret values."""

    data = asdict(config)
    data.pop("metadata", None)
    return data


def build_generation_metadata(
    request: GenerationRequest,
    *,
    provider: str,
    model_name: str,
    cache_hit: bool,
    latency_seconds: float,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build standardized non-secret metadata for generation results."""

    config = request.config
    metadata: dict[str, Any] = {
        "provider": provider,
        "model_name": model_name,
        "prompt_variant": config.prompt_variant,
        "input_mode": config.input_mode or request.hybrid_input.mode,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "stop": config.stop,
        "think": config.think,
        "native_endpoint": config.native_endpoint,
        "num_predict": config.num_predict,
        "seed": config.seed,
        "prompt_profile_id": config.prompt_profile_id,
        "prompt_profile_path": config.prompt_profile_path,
        "require_non_empty_code": config.require_non_empty_code,
        "prompt_hash": prompt_hash(request.prompt_text),
        "prompt_text_sha256": prompt_hash(request.prompt_text),
        "request_hash": request_hash(request),
        "cache_hit": cache_hit,
        "latency_seconds": latency_seconds,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
    }
    if error is not None:
        metadata["error"] = error
    if extra:
        metadata.update(extra)
    return metadata


@dataclass(slots=True)
class CodeExtractionResult:
    """Python code extracted from generator final content."""

    code: str
    extraction_strategy_used: str
    extracted_code_non_empty: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render this result as plain Python containers."""

        return asdict(self)


def extract_python_code_with_diagnostics(raw_response: str) -> CodeExtractionResult:
    """Extract Python code and record which extraction strategy was used."""

    if not raw_response.strip():
        return CodeExtractionResult(
            code="",
            extraction_strategy_used="empty_final_content",
            extracted_code_non_empty=False,
        )

    json_code = _extract_json_code_field(raw_response)
    if json_code is not None:
        return _code_extraction_result(json_code, "json_code_field")

    begin_end_blocks = [
        match.group("code")
        for match in re.finditer(
            r"BEGIN_PYTHON\s*(?P<code>.*?)\s*END_PYTHON",
            raw_response,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]
    if begin_end_blocks:
        return _best_parseable_code_block(begin_end_blocks, "begin_end_python")

    python_blocks = [
        match.group("code")
        for match in re.finditer(
            r"```(?:python|py)\s*\n(?P<code>.*?)```",
            raw_response,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]
    if python_blocks:
        return _best_parseable_code_block(python_blocks, "python_code_fence")

    generic_blocks = [
        match.group("code")
        for match in re.finditer(
            r"```\s*\n(?P<code>.*?)```",
            raw_response,
            flags=re.DOTALL,
        )
    ]
    if generic_blocks:
        return _best_parseable_code_block(generic_blocks, "generic_code_fence")

    stripped = raw_response.strip()
    if _looks_like_python_code(stripped):
        return _code_extraction_result(stripped, "bare_python")

    result = _code_extraction_result(raw_response, "raw_text")
    if result.extracted_code_non_empty and not _can_parse_python(result.code):
        result.warnings.append("Extracted raw text is not parseable Python.")
    return result


def extract_python_code(raw_response: str) -> str:
    """Extract Python code from generator final content."""

    return extract_python_code_with_diagnostics(raw_response).code


def _extract_json_code_field(raw_response: str) -> str | None:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("code"), str):
        return str(parsed["code"])
    return None


def _code_extraction_result(code: str, strategy: str) -> CodeExtractionResult:
    normalized = code.strip()
    if normalized:
        normalized += "\n"
    return CodeExtractionResult(
        code=normalized,
        extraction_strategy_used=strategy,
        extracted_code_non_empty=bool(normalized),
    )


def _best_parseable_code_block(
    blocks: list[str],
    strategy: str,
) -> CodeExtractionResult:
    warnings: list[str] = []
    normalized_blocks = [
        block.strip() + "\n" if block.strip() else "" for block in blocks
    ]
    for index, block in enumerate(normalized_blocks):
        if block and _can_parse_python(block):
            result = CodeExtractionResult(
                code=block,
                extraction_strategy_used=strategy,
                extracted_code_non_empty=True,
                warnings=warnings,
            )
            if index > 0:
                result.warnings.append(
                    f"Selected parseable {strategy} block at index {index}."
                )
            return result
        if block:
            warnings.append(f"Skipped non-parseable {strategy} block at index {index}.")

    first = normalized_blocks[0] if normalized_blocks else ""
    result = CodeExtractionResult(
        code=first,
        extraction_strategy_used=strategy,
        extracted_code_non_empty=bool(first),
        warnings=warnings,
    )
    if first and not _can_parse_python(first):
        result.warnings.append(f"No parseable {strategy} block was found.")
    return result


def _can_parse_python(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def _looks_like_python_code(text: str) -> bool:
    return bool(
        re.match(
            r"^(from\s+__future__|from\s+\w+|import\s+\w+|class\s+\w+|def\s+\w+)",
            text,
        )
    )


def save_generation_result(result: GenerationResult, path: str | Path) -> None:
    """Persist a generation result as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")


def load_generation_result(path: str | Path) -> GenerationResult:
    """Load a generation result from JSON."""

    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError("Generation result JSON root must be an object.")
    return GenerationResult.from_dict(data)


def _generation_warnings(hybrid_input: HybridInput) -> list[str]:
    warnings: list[str] = []
    if not hybrid_input.edges:
        warnings.append("Hybrid input has no interaction edges.")
    if not hybrid_input.filters:
        warnings.append("Hybrid input has no attribute filters.")
    return warnings


def _render_agent_class(class_name: str, agent_data: dict[str, Any]) -> list[str]:
    attributes = agent_data["attributes"]
    used_attribute_names: set[str] = set()
    attribute_pairs = [
        (
            str(attribute["name"]),
            _unique_identifier(
                _to_snake_case(str(attribute["name"])),
                used_attribute_names,
            ),
        )
        for attribute in attributes
    ]

    lines = [
        f"class {class_name}:",
        f'    """Agent class generated for MAG agent {agent_data["agent_id"]!r}."""',
    ]
    if attribute_pairs:
        params = ", ".join(f"{name}: Any = None" for _, name in attribute_pairs)
        lines.append(f"    def __init__(self, {params}) -> None:")
        for original_name, safe_name in attribute_pairs:
            lines.append(f"        self.{safe_name} = {safe_name}")
            if safe_name != original_name:
                lines.append(
                    f"        setattr(self, {_quote(original_name)}, {safe_name})"
                )
    else:
        lines.extend(
            [
                "    def __init__(self) -> None:",
                "        pass",
            ]
        )
    lines.extend(["", ""])
    return lines


def _render_lookup_helper() -> list[str]:
    return [
        "def _lookup_state(source: Any, target: Any, name: str) -> Any:",
        "    if hasattr(source, name):",
        "        return getattr(source, name)",
        "    if hasattr(target, name):",
        "        return getattr(target, name)",
        "    return None",
        "",
        "",
    ]


def _render_filter_function(
    function_name: str,
    filter_id: str,
    expressions: list[str],
) -> list[str]:
    lines = [
        f"def {function_name}(source: Any, target: Any) -> bool:",
        f'    """Guard for MAG filter {filter_id!r}."""',
    ]
    if not expressions:
        lines.extend(["    return True", "", ""])
        return lines

    names = sorted(
        {
            name
            for expression in expressions
            for name in _expression_names(expression)
            if name not in {"source", "target", "True", "False", "None"}
        }
    )
    for name in names:
        lines.append(f"    {name} = _lookup_state(source, target, {_quote(name)})")

    valid_expressions = [
        expression for expression in expressions if _is_python_expression(expression)
    ]
    if not valid_expressions:
        lines.extend(
            [
                "    # Original expressions could not be parsed as Python:",
                *[f"    # - {expression}" for expression in expressions],
                "    return True",
                "",
                "",
            ]
        )
        return lines

    if len(valid_expressions) == 1:
        return_expression = f"bool({valid_expressions[0]})"
    else:
        return_expression = " and ".join(
            f"bool({expression})" for expression in valid_expressions
        )
    lines.extend(
        [
            "    try:",
            f"        return {return_expression}",
            "    except (AttributeError, TypeError, ValueError):",
            "        return False",
            "",
            "",
        ]
    )
    return lines


def _render_interaction_function(
    edge: HybridEdgeSemantics,
    function_name: str,
    source_class: str,
    target_class: str,
    filter_function: str | None,
) -> list[str]:
    source_agent = edge.source["agent_id"]
    source_port = edge.source["port"]
    target_agent = edge.target["agent_id"]
    target_port = edge.target["port"]
    target_message_attribute = f"last_{_to_snake_case(target_port)}"
    lines = [
        f"def {function_name}(",
        f"    source: {source_class},",
        f"    target: {target_class},",
        ") -> dict[str, Any]:",
        f'    """Initial interaction for MAG edge {edge.edge_id!r}."""',
    ]
    if filter_function is not None:
        lines.extend(
            [
                f"    if not {filter_function}(source, target):",
                "        return {",
                f'            "edge_id": {_quote(edge.edge_id)},',
                '            "delivered": False,',
                '            "reason": "filter_failed",',
                "        }",
            ]
        )
    lines.extend(
        [
            "    message = {",
            f'        "edge_id": {_quote(edge.edge_id)},',
            f'        "source": {_quote(f"{source_agent}.{source_port}")},',
            f'        "target": {_quote(f"{target_agent}.{target_port}")},',
            '        "delivered": True,',
            f"        \"payload\": getattr(source, {_quote(source_port)}, None),",
            "    }",
            f"    target.{target_message_attribute} = message",
            "    return message",
            "",
            "",
        ]
    )
    return lines


def _expression_names(expression: str) -> set[str]:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        return set()
    return {
        node.id
        for node in ast.walk(parsed)
        if isinstance(node, ast.Name) and _is_identifier(node.id)
    }


def _is_python_expression(expression: str) -> bool:
    try:
        ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    return True


def _to_class_name(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value)
    if not words:
        return "GeneratedAgent"
    class_name = "".join(word[:1].upper() + word[1:] for word in words)
    if class_name[0].isdigit():
        return f"Agent{class_name}"
    if keyword.iskeyword(class_name):
        return f"{class_name}Agent"
    return class_name


def _to_snake_case(value: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    if not value:
        value = "value"
    if value[0].isdigit():
        value = f"value_{value}"
    if keyword.iskeyword(value):
        value = f"{value}_"
    return value


def _unique_identifier(base_name: str, used_names: set[str]) -> str:
    candidate = base_name
    index = 2
    while candidate in used_names:
        candidate = f"{base_name}_{index}"
        index += 1
    used_names.add(candidate)
    return candidate


def _is_identifier(value: str) -> bool:
    return value.isidentifier() and not keyword.iskeyword(value)


def _quote(value: str) -> str:
    return repr(value)


def _escape_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
