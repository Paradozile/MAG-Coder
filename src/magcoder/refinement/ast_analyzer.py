"""AST analysis helpers for generated Python code."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ASTAnalysisResult:
    """Summary of parseable structures found in generated Python source."""

    class_names: list[str] = field(default_factory=list)
    methods_by_class: dict[str, list[str]] = field(default_factory=dict)
    attributes_by_class: dict[str, list[str]] = field(default_factory=dict)
    if_conditions: list[str] = field(default_factory=list)
    function_calls: list[str] = field(default_factory=list)
    function_calls_by_class_method: dict[str, dict[str, list[str]]] = field(
        default_factory=dict
    )
    function_calls_by_function: dict[str, list[str]] = field(default_factory=dict)
    parse_errors: list[str] = field(default_factory=list)
    function_names: list[str] = field(default_factory=list)
    base_classes_by_class: dict[str, list[str]] = field(default_factory=dict)
    raw_ast: ast.Module | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render the analysis result without the non-serializable raw AST."""

        data = asdict(self)
        data.pop("raw_ast", None)
        return data


def analyze_python_source(
    source_code: str,
    *,
    include_raw_ast: bool = False,
) -> ASTAnalysisResult:
    """Parse and summarize generated Python source code."""

    try:
        tree = ast.parse(source_code)
    except SyntaxError as error:
        return ASTAnalysisResult(parse_errors=[_format_syntax_error(error)])

    result = ASTAnalysisResult(raw_ast=tree if include_raw_ast else None)
    result.class_names = [
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    ]
    result.function_names = [
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    result.function_calls_by_function = {
        node.name: sorted(_function_call_names(node))
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        result.methods_by_class[node.name] = [
            item.name for item in node.body if isinstance(item, ast.FunctionDef)
        ]
        result.function_calls_by_class_method[node.name] = {
            item.name: sorted(_function_call_names(item))
            for item in node.body
            if isinstance(item, ast.FunctionDef)
        }
        result.attributes_by_class[node.name] = sorted(_class_attributes(node))
        result.base_classes_by_class[node.name] = [
            _unparse(base) for base in node.bases
        ]

    result.if_conditions = [
        _unparse(node.test) for node in ast.walk(tree) if isinstance(node, ast.If)
    ]
    result.function_calls = sorted(
        {
            call_name
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            if (call_name := _call_name(node)) is not None
        }
    )
    return result


def _class_attributes(class_node: ast.ClassDef) -> set[str]:
    attributes: set[str] = set()
    for node in ast.walk(class_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (attribute_name := _self_attribute_name(target)) is not None:
                    attributes.add(attribute_name)
        elif (
            isinstance(node, ast.AnnAssign)
            and (attribute_name := _self_attribute_name(node.target)) is not None
        ):
            attributes.add(attribute_name)
    return attributes


def _self_attribute_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Attribute):
        return None
    if isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _function_call_names(function_node: ast.FunctionDef) -> set[str]:
    return {
        call_name
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        if (call_name := _call_name(node)) is not None
    }


def _unparse(node: ast.AST) -> str:
    return ast.unparse(node)


def _format_syntax_error(error: SyntaxError) -> str:
    location = f"line {error.lineno}" if error.lineno is not None else "unknown line"
    return f"{location}: {error.msg}"
