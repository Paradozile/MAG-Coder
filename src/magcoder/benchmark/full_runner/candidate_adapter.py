"""Thin source-entry wrappers for generated Full MAG-Bench candidates."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any

from magcoder.benchmark.full_builder.adapter_spec import EntryPointAdapterSpec


@dataclass(slots=True)
class AdaptedCandidate:
    """Candidate code plus entry-point adapter diagnostics."""

    code: str
    direct_entrypoint: bool
    adapter_used: bool
    adapter_success: bool
    required_candidate_symbols_present: bool
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this result as JSON-compatible containers."""

        return asdict(self)


def build_candidate_with_entrypoint_adapter(
    candidate_code: str,
    adapter_spec: EntryPointAdapterSpec,
) -> AdaptedCandidate:
    """Append a thin source-compatible entry point when possible."""

    try:
        tree = ast.parse(candidate_code)
    except SyntaxError as error:
        return AdaptedCandidate(
            code=candidate_code,
            direct_entrypoint=False,
            adapter_used=False,
            adapter_success=False,
            required_candidate_symbols_present=False,
            error_type="SyntaxError",
            error_message=str(error),
        )

    symbols = _module_symbols(tree)
    if adapter_spec.source_entry_point in symbols["functions"]:
        return AdaptedCandidate(
            code=candidate_code,
            direct_entrypoint=True,
            adapter_used=False,
            adapter_success=True,
            required_candidate_symbols_present=True,
            metadata={"entrypoint_mode": "direct"},
        )

    source_class = adapter_spec.candidate_agent_class
    target_class = str(adapter_spec.input_mapping.get("target_agent_class", ""))
    method_name = adapter_spec.candidate_interaction_method
    source_class_methods = symbols["class_methods"].get(source_class, set())
    has_method = (
        method_name in source_class_methods or method_name in symbols["functions"]
    )
    required_present = bool(source_class and source_class in symbols["classes"]) and (
        not target_class or target_class in symbols["classes"]
    ) and bool(method_name and has_method)
    if not required_present:
        return AdaptedCandidate(
            code=candidate_code,
            direct_entrypoint=False,
            adapter_used=False,
            adapter_success=False,
            required_candidate_symbols_present=False,
            error_type="AdapterConstructionFailure",
            error_message=(
                "Required candidate symbols were missing: "
                + ", ".join(adapter_spec.required_candidate_symbols)
            ),
            metadata={
                "entrypoint_mode": "missing",
                "available_classes": sorted(symbols["classes"]),
                "available_functions": sorted(symbols["functions"]),
            },
        )

    wrapper = _render_adapter_wrapper(adapter_spec)
    return AdaptedCandidate(
        code=candidate_code.rstrip() + "\n\n\n" + wrapper,
        direct_entrypoint=False,
        adapter_used=True,
        adapter_success=True,
        required_candidate_symbols_present=True,
        metadata={"entrypoint_mode": "adapter"},
    )


def _module_symbols(tree: ast.Module) -> dict[str, Any]:
    classes: set[str] = set()
    functions: set[str] = set()
    class_methods: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.add(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.add(node.name)
            class_methods[node.name] = {
                item.name for item in node.body if isinstance(item, ast.FunctionDef)
            }
    return {
        "classes": classes,
        "functions": functions,
        "class_methods": class_methods,
    }


def _render_adapter_wrapper(adapter_spec: EntryPointAdapterSpec) -> str:
    source_entry_point = adapter_spec.source_entry_point
    source_class = adapter_spec.candidate_agent_class
    target_class = str(adapter_spec.input_mapping.get("target_agent_class", ""))
    method_name = adapter_spec.candidate_interaction_method
    source_attribute = str(
        adapter_spec.input_mapping.get("source_agent_attribute", "input_data")
    )
    preferred_output = str(
        adapter_spec.output_mapping.get("preferred_target_attribute", "last_result")
    )
    fallback_keys = adapter_spec.output_mapping.get("fallback_return_keys", [])
    if not isinstance(fallback_keys, list):
        fallback_keys = []
    return f'''def {source_entry_point}(*args, **kwargs):
    """Thin MAG-Bench adapter generated from entry_point_adapter_spec.

    This wrapper delegates to candidate agent interaction code and does not
    copy the oracle implementation.
    """
    source_payload = {{"args": args, "kwargs": kwargs}}
    source_agent = {source_class}(**{{{source_attribute!r}: source_payload}})
    target_agent = {target_class}()
    if hasattr(source_agent, {method_name!r}):
        adapter_result = getattr(source_agent, {method_name!r})(target_agent)
    elif {method_name!r} in globals():
        adapter_result = globals()[{method_name!r}](source_agent, target_agent)
    else:
        raise AttributeError({method_name!r})
    if hasattr(target_agent, {preferred_output!r}):
        target_value = getattr(target_agent, {preferred_output!r})
        if target_value is not None:
            return target_value
    if isinstance(adapter_result, dict):
        for key in {fallback_keys!r}:
            if key in adapter_result:
                return adapter_result[key]
    return adapter_result
'''
