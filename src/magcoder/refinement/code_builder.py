"""Build small deterministic code snippets for refinement."""

from __future__ import annotations

import ast
import keyword
from dataclasses import dataclass, field

from magcoder.refinement.constraints import InteractionConstraint
from magcoder.refinement.framework import FrameworkSpec


@dataclass(slots=True)
class BuiltMethod:
    """Rendered method source and non-fatal warnings."""

    source: str
    warnings: list[str] = field(default_factory=list)


def build_interaction_method_source(
    *,
    edge: InteractionConstraint,
    method_name: str,
    filter_expressions: list[str],
    framework: FrameworkSpec,
) -> BuiltMethod:
    """Build a framework-style interaction method for one MAG edge."""

    warnings: list[str] = []
    lines = [
        f"def {method_name}(self, target):",
        f'    """Refined interaction for MAG edge {edge.edge_id!r}."""',
        (
            f"    # Direction: {edge.source_agent}.{edge.source_port} -> "
            f"{edge.target_agent}.{edge.target_port}"
        ),
    ]
    valid_expressions: list[str] = []
    invalid_expressions: list[str] = []
    for expression in filter_expressions:
        if _is_python_expression(expression):
            valid_expressions.append(expression)
        else:
            invalid_expressions.append(expression)

    names = sorted(
        {
            name
            for expression in valid_expressions
            for name in _expression_names(expression)
            if name not in {"self", "target", "True", "False", "None"}
        }
    )
    for name in names:
        lines.extend(_state_lookup_lines(name))

    message_line = _message_api_line(framework)
    if valid_expressions:
        for expression in valid_expressions:
            lines.extend(
                [
                    "    try:",
                    f"        if {expression}:",
                    f"            {message_line}",
                    "    except (AttributeError, TypeError, ValueError):",
                    "        return None",
                ]
            )
    elif invalid_expressions:
        for expression in invalid_expressions:
            lines.append(
                f"    # Original filter expression is not valid Python: {expression}"
            )
            warnings.append(
                f"Filter expression for edge {edge.edge_id!r} is not valid Python: "
                f"{expression!r}."
            )
        lines.extend(
            [
                "    if True:",
                f"        {message_line}",
            ]
        )
    else:
        lines.append(f"    {message_line}")

    lines.append("    return None")
    return BuiltMethod(source="\n".join(lines), warnings=warnings)


def build_missing_class_source(
    *,
    class_name: str,
    attributes: list[str],
) -> str:
    """Build a minimal missing agent class."""

    lines = [f"class {class_name}:"]
    if attributes:
        params = ", ".join(f"{attribute}=None" for attribute in attributes)
        lines.append(f"    def __init__(self, {params}):")
        for attribute in attributes:
            lines.append(f"        self.{attribute} = {attribute}")
    else:
        lines.extend(
            [
                "    def __init__(self):",
                "        pass",
            ]
        )
    return "\n".join(lines)


def _state_lookup_lines(name: str) -> list[str]:
    line = f"    {name} = getattr(self, {name!r}, getattr(target, {name!r}, None))"
    if len(line) <= 88:
        return [line]
    return [
        f"    {name} = getattr(",
        "        self,",
        f"        {name!r},",
        f"        getattr(target, {name!r}, None),",
        "    )",
    ]


def _message_api_line(framework: FrameworkSpec) -> str:
    if framework.message_api_name:
        return (
            f"return self.{framework.message_api_name}"
            '(target, "interaction_triggered")'
        )
    return 'return {"delivered": True, "message": "interaction_triggered"}'


def _is_python_expression(expression: str) -> bool:
    try:
        ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    return True


def _expression_names(expression: str) -> set[str]:
    parsed = ast.parse(expression, mode="eval")
    return {
        node.id
        for node in ast.walk(parsed)
        if isinstance(node, ast.Name)
        and node.id.isidentifier()
        and not keyword.iskeyword(node.id)
    }
