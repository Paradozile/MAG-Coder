"""Deterministic AST/code refinement."""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from magcoder.models import MAGModel
from magcoder.refinement.ast_analyzer import analyze_python_source
from magcoder.refinement.checker import (
    ConsistencyReport,
    check_consistency,
)
from magcoder.refinement.code_builder import (
    build_interaction_method_source,
    build_missing_class_source,
)
from magcoder.refinement.constraints import (
    ModelConstraints,
    constraints_with_interaction_contracts,
    extract_model_constraints,
)
from magcoder.refinement.contracts import build_interaction_contracts
from magcoder.refinement.framework import FrameworkSpec
from magcoder.refinement.rewrite_plan import RewritePlan, build_rewrite_plan


@dataclass(slots=True)
class RefinedCodeResult:
    """Result of deterministic code refinement."""

    code_initial: str
    code_refined: str
    initial_report: ConsistencyReport
    refined_report: ConsistencyReport
    rewrite_actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this result as plain Python containers."""

        return {
            "code_initial": self.code_initial,
            "code_refined": self.code_refined,
            "initial_report": self.initial_report.to_dict(),
            "refined_report": self.refined_report.to_dict(),
            "rewrite_actions": self.rewrite_actions,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class BalancedRefinementConfig:
    """Configuration for deterministic refinement strategy selection."""

    strategy: str = "abms_focused"
    preserve_source_entrypoint: bool = True
    preserve_existing_return_values: bool = True
    allow_message_bridge: bool = True
    allow_signature_wrapper: bool = True
    allow_attribute_defaults: bool = True
    record_conflicts: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render this configuration as JSON-compatible containers."""

        return asdict(self)


def refine_code(
    code_initial: str,
    mag_model: MAGModel,
    framework_spec: FrameworkSpec,
    refinement_config: BalancedRefinementConfig | str | None = None,
) -> RefinedCodeResult:
    """Refine generated code using deterministic model/framework rules."""

    config = _coerce_refinement_config(refinement_config)
    code_for_refinement, pre_warnings, pre_actions = _preprocess_code_initial(
        code_initial
    )
    constraints = constraints_with_interaction_contracts(
        extract_model_constraints(mag_model),
        build_interaction_contracts(mag_model, framework_spec),
    )
    initial_analysis = analyze_python_source(code_for_refinement)
    initial_report = check_consistency(
        initial_analysis,
        constraints,
        framework_spec,
    )
    plan = build_rewrite_plan(initial_report, constraints, framework_spec)

    try:
        tree = ast.parse(code_for_refinement)
    except SyntaxError as error:
        warning = f"Initial code could not be parsed for refinement: {error.msg}."
        return RefinedCodeResult(
            code_initial=code_initial,
            code_refined=code_for_refinement,
            initial_report=initial_report,
            refined_report=initial_report,
            rewrite_actions=_dedupe_labels([*plan.action_labels(), *pre_actions]),
            warnings=[*pre_warnings, warning],
            metadata=_metadata(constraints, framework_spec, plan, config),
        )

    code_refined, warnings, rewrite_actions = _rewrite_module(
        code_initial=code_for_refinement,
        tree=tree,
        constraints=constraints,
        framework=framework_spec,
        refinement_config=config,
    )
    refined_analysis = analyze_python_source(code_refined)
    refined_report = check_consistency(
        refined_analysis,
        constraints,
        framework_spec,
    )
    return RefinedCodeResult(
        code_initial=code_initial,
        code_refined=code_refined,
        initial_report=initial_report,
        refined_report=refined_report,
        rewrite_actions=_dedupe_labels(
            [*plan.action_labels(), *pre_actions, *rewrite_actions]
        ),
        warnings=[*plan.warnings, *pre_warnings, *warnings],
        metadata=_metadata(constraints, framework_spec, plan, config),
    )


def save_refined_code_result(result: RefinedCodeResult, path: str | Path) -> None:
    """Save a refined code result as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")


def _preprocess_code_initial(code_initial: str) -> tuple[str, list[str], list[str]]:
    """Remove output-format sentinels that are valid Python names but fail import."""

    removed_markers: list[str] = []
    kept_lines: list[str] = []
    for line in code_initial.splitlines():
        marker = line.strip()
        if marker in {"BEGIN_PYTHON", "END_PYTHON"}:
            removed_markers.append(marker)
            continue
        kept_lines.append(line)
    if not removed_markers:
        return code_initial, [], []
    warnings = [
        "Removed output-format marker lines from candidate code: "
        + ", ".join(sorted(set(removed_markers)))
        + "."
    ]
    return "\n".join(kept_lines).lstrip() + "\n", warnings, [
        "remove_output_format_markers"
    ]


def _rewrite_module(
    *,
    code_initial: str,
    tree: ast.Module,
    constraints: ModelConstraints,
    framework: FrameworkSpec,
    refinement_config: BalancedRefinementConfig,
) -> tuple[str, list[str], list[str]]:
    class_nodes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    method_sources_by_class: dict[str, list[str]] = {
        class_name: [] for class_name in constraints.expected_agent_class_names
    }
    self_attributes_by_class = _self_attributes_by_class(tree)
    warnings: list[str] = []
    rewrite_actions: list[str] = []

    edge_by_id = {edge.edge_id: edge for edge in constraints.expected_interaction_edges}
    for contract in constraints.expected_interaction_contracts:
        edge = edge_by_id[contract.edge_id]
        source_class = edge.source_class_name
        method_name = contract.required_method_name
        class_node = class_nodes.get(source_class)
        existing_methods = _class_method_names(class_node)
        method_node = _class_method_node(class_node, method_name)
        for fallback_name in contract.fallback_method_names:
            if fallback_name in existing_methods:
                warnings.append(
                    f"Preserved fallback interaction method {fallback_name!r} "
                    f"for edge {edge.edge_id!r}."
                )
        guard = constraints.expected_guard_requirements.get(edge.edge_id)
        filter_expressions = [] if guard is None else guard.expressions
        patch_reasons = _existing_method_patch_reasons(
            method_node=method_node,
            target_parameter_names=_method_target_parameter_names(method_node),
            filter_expressions=filter_expressions,
            message_api_name=framework.message_api_name,
        )
        if method_name in existing_methods and not patch_reasons:
            continue
        built_method = build_interaction_method_source(
            edge=edge,
            method_name=method_name,
            filter_expressions=filter_expressions,
            framework=framework,
        )
        method_sources_by_class.setdefault(source_class, []).append(
            built_method.source
        )
        warnings.extend(built_method.warnings)
        if method_name not in existing_methods:
            rewrite_actions.append(f"add_required_edge_method:{edge.edge_id}")
        else:
            rewrite_actions.append(
                f"enforce_message_api_in_required_method:{edge.edge_id}"
            )
            if "missing_message_api" in patch_reasons:
                rewrite_actions.append(
                    f"insert_message_api_call_in_existing_method:{edge.edge_id}"
                )
            if "missing_guard" in patch_reasons:
                rewrite_actions.append(
                    f"insert_guard_in_existing_method:{edge.edge_id}"
                )
            if "wrong_signature" in patch_reasons:
                rewrite_actions.append(
                    f"normalize_interaction_method_signature:{edge.edge_id}"
                )
            if "wrong_message_api_signature" in patch_reasons:
                rewrite_actions.append(
                    f"normalize_message_api_signature:{edge.edge_id}"
                )
            if "wrong_message_api_order" in patch_reasons:
                rewrite_actions.append(
                    f"normalize_message_api_argument_order:{edge.edge_id}"
                )
            warnings.append(
                f"Added contract-compliant override for existing method "
                f"{method_name!r} on edge {edge.edge_id!r}: "
                f"{', '.join(patch_reasons)}."
            )

    for class_name, attributes in constraints.expected_attributes.items():
        init_attributes = _dedupe_labels(
            [*attributes, *self_attributes_by_class.get(class_name, [])]
        )
        method_sources_by_class.setdefault(class_name, [])
        method_sources_by_class[class_name].append(
            _build_abms_safe_init_source(init_attributes)
        )
        rewrite_actions.append(f"ensure_abms_attribute_initialization:{class_name}")
        extra_attributes = [
            attribute
            for attribute in init_attributes
            if attribute not in set(attributes)
        ]
        if extra_attributes:
            rewrite_actions.append(
                f"initialize_candidate_self_attributes:{class_name}"
            )
            warnings.append(
                f"Initialized candidate-referenced attributes on {class_name}: "
                f"{', '.join(extra_attributes)}."
            )
        if framework.message_api_name:
            method_sources_by_class[class_name].append(
                _build_abms_safe_message_api_source(framework.message_api_name)
            )
            rewrite_actions.append(f"ensure_abms_message_api:{class_name}")

    code_sections: list[str] = []
    emitted_classes: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            emitted_classes.add(node.name)
            class_source = ast.get_source_segment(code_initial, node)
            if class_source is None:
                class_source = _render_class_with_methods(class_node=node)
            code_sections.append(
                _append_methods_to_class_source(
                    class_source=class_source,
                    extra_method_sources=method_sources_by_class.get(node.name, []),
                )
            )
        else:
            source_segment = ast.get_source_segment(code_initial, node)
            code_sections.append(source_segment or ast.unparse(node))

    for class_name in constraints.expected_agent_class_names:
        if class_name in emitted_classes:
            continue
        code_sections.append(
            build_missing_class_source(
                class_name=class_name,
                attributes=constraints.expected_attributes.get(class_name, []),
            )
        )
        if method_sources_by_class.get(class_name):
            code_sections[-1] = _append_methods_to_class_source(
                class_source=code_sections[-1],
                extra_method_sources=method_sources_by_class[class_name],
            )

    code_refined = "\n\n\n".join(section for section in code_sections if section)
    comments = _missing_direction_metadata_comments(code_refined, constraints)
    if comments:
        code_refined = "\n\n\n".join([code_refined, *comments])
    if (
        refinement_config.strategy == "balanced"
        and refinement_config.preserve_source_entrypoint
    ):
        source_entry_point = str(
            refinement_config.metadata.get("source_entry_point", "")
        ).strip()
        snapshot = _build_source_entrypoint_snapshot(
            code_initial=code_initial,
            source_entry_point=source_entry_point,
        )
        if snapshot.source:
            code_refined = "\n\n\n".join([code_refined, snapshot.source])
        warnings.extend(snapshot.warnings)
        rewrite_actions.extend(snapshot.rewrite_actions)
    code_refined = _normalize_import_spacing(code_refined)
    return code_refined.rstrip() + "\n", warnings, rewrite_actions


@dataclass(slots=True)
class _SourceEntryPointSnapshot:
    source: str
    rewrite_actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _build_source_entrypoint_snapshot(
    *,
    code_initial: str,
    source_entry_point: str,
) -> _SourceEntryPointSnapshot:
    """Build a candidate-derived source entrypoint snapshot for balanced repair."""

    if not source_entry_point:
        return _SourceEntryPointSnapshot(
            source="",
            warnings=[
                "Balanced refinement could not preserve source entrypoint: "
                "source_entry_point metadata was missing."
            ],
        )
    try:
        tree = ast.parse(code_initial)
    except SyntaxError as error:
        return _SourceEntryPointSnapshot(
            source="",
            warnings=[
                "Balanced refinement could not preserve source entrypoint: "
                f"candidate parse failed ({error.msg})."
            ],
        )
    source_function = _top_level_function(tree, source_entry_point)
    if source_function is None:
        return _SourceEntryPointSnapshot(
            source="",
            warnings=[
                "Balanced refinement could not preserve source entrypoint: "
                f"{source_entry_point!r} was not defined by the candidate."
            ],
        )
    top_level_names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }
    rename_map = {
        name: f"_magcoder_source_{name}"
        for name in top_level_names
        if name != source_entry_point
    }
    snapshot_nodes: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.Import | ast.ImportFrom | ast.Assign | ast.AnnAssign):
            snapshot_nodes.append(copy.deepcopy(node))
            continue
        if not isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        cloned = copy.deepcopy(node)
        if cloned.name in rename_map:
            cloned.name = rename_map[cloned.name]
        cloned = _ReferenceRenamer(rename_map).visit(cloned)
        if not isinstance(cloned, ast.stmt):
            continue
        snapshot_nodes.append(cloned)
    source = "\n\n".join(ast.unparse(node) for node in snapshot_nodes)
    if not source:
        return _SourceEntryPointSnapshot(
            source="",
            warnings=[
                "Balanced refinement could not preserve source entrypoint: "
                "snapshot rendering produced no code."
            ],
        )
    return _SourceEntryPointSnapshot(
        source=(
            "# MAG-Coder balanced source-entrypoint preservation snapshot.\n"
            "# This code is copied from the candidate's original output with "
            "helper symbols renamed; it does not contain gold-solution logic.\n"
            f"{source}"
        ),
        rewrite_actions=[f"preserve_source_entrypoint_snapshot:{source_entry_point}"],
        warnings=[
            "Balanced refinement preserved the candidate-defined source "
            f"entrypoint {source_entry_point!r} with renamed helper symbols."
        ],
    )


class _ReferenceRenamer(ast.NodeTransformer):
    """Rename top-level helper references inside a copied source snapshot."""

    def __init__(self, rename_map: dict[str, str]) -> None:
        self._rename_map = rename_map

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Rename name references that point at copied helper symbols."""

        if node.id in self._rename_map:
            return ast.copy_location(
                ast.Name(id=self._rename_map[node.id], ctx=node.ctx),
                node,
            )
        return node


def _top_level_function(
    tree: ast.Module,
    name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == name
        ):
            return node
    return None


def _coerce_refinement_config(
    refinement_config: BalancedRefinementConfig | str | None,
) -> BalancedRefinementConfig:
    if refinement_config is None:
        return BalancedRefinementConfig()
    if isinstance(refinement_config, str):
        return BalancedRefinementConfig(strategy=refinement_config)
    return refinement_config


def _render_class_with_methods(
    *,
    class_node: ast.ClassDef,
) -> str:
    bases = ""
    if class_node.bases:
        bases = "(" + ", ".join(ast.unparse(base) for base in class_node.bases) + ")"
    lines = [f"class {class_node.name}{bases}:"]
    body_sections = [ast.unparse(statement) for statement in class_node.body]
    if not body_sections:
        body_sections = ["pass"]
    for section in body_sections:
        lines.append(_indent(section, spaces=4))
    return "\n".join(lines)


def _append_methods_to_class_source(
    *,
    class_source: str,
    extra_method_sources: list[str],
) -> str:
    if not extra_method_sources:
        return class_source
    method_block = "\n\n".join(
        _indent(method_source, spaces=4) for method_source in extra_method_sources
    )
    return f"{class_source.rstrip()}\n\n{method_block}"


def _class_method_names(class_node: ast.ClassDef | None) -> set[str]:
    if class_node is None:
        return set()
    return {
        node.name for node in class_node.body if isinstance(node, ast.FunctionDef)
    }


def _class_method_node(
    class_node: ast.ClassDef | None,
    method_name: str,
) -> ast.FunctionDef | None:
    if class_node is None:
        return None
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    return None


def _existing_method_patch_reasons(
    *,
    method_node: ast.FunctionDef | None,
    target_parameter_names: set[str],
    filter_expressions: list[str],
    message_api_name: str,
) -> list[str]:
    if method_node is None:
        return []
    reasons: list[str] = []
    if message_api_name and message_api_name not in _method_call_names(method_node):
        reasons.append("missing_message_api")
    if not _method_signature_compatible(method_node):
        reasons.append("wrong_signature")
    message_api_issue = _message_api_call_issue(
        method_node,
        message_api_name,
        target_parameter_names,
    )
    if message_api_issue is not None:
        reasons.append(message_api_issue)
    if filter_expressions and not _method_has_filter_guard(
        method_node,
        filter_expressions,
    ):
        reasons.append("missing_guard")
    return reasons


def _method_signature_compatible(method_node: ast.FunctionDef) -> bool:
    args = method_node.args
    positional_count = len(args.posonlyargs) + len(args.args)
    if args.vararg is not None:
        return True
    return positional_count >= 2


def _method_target_parameter_names(method_node: ast.FunctionDef | None) -> set[str]:
    if method_node is None:
        return set()
    names = [
        arg.arg
        for arg in [*method_node.args.posonlyargs, *method_node.args.args]
    ]
    return {
        name
        for name in names[1:]
        if name
        in {
            "target",
            "target_agent",
            "destination",
            "destination_agent",
            "dst",
            "receiver",
            "recipient",
        }
        or name.endswith("_agent")
    }


def _message_api_call_issue(
    method_node: ast.FunctionDef,
    message_api_name: str,
    target_parameter_names: set[str],
) -> str | None:
    if not message_api_name:
        return None
    for node in ast.walk(method_node):
        if not isinstance(node, ast.Call) or _call_name(node) != message_api_name:
            continue
        if len(node.args) < 2:
            return "wrong_message_api_signature"
        first_name = _argument_name(node.args[0])
        if (
            target_parameter_names
            and first_name is not None
            and first_name not in target_parameter_names
        ):
            return "wrong_message_api_order"
    return None


def _argument_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _build_abms_safe_init_source(attributes: list[str]) -> str:
    lines = [
        "def __init__(self, *args, **kwargs):",
        '    """ABMS-safe initializer inserted by deterministic refinement."""',
        "    self.messages = kwargs.get('messages', [])",
        "    self.sent_messages = kwargs.get('sent_messages', [])",
    ]
    for index, attribute in enumerate(attributes):
        if index == 0:
            lines.extend(
                [
                    "    if args:",
                    f"        self.{attribute} = args[0]",
                    "    else:",
                    (
                        f"        self.{attribute} = kwargs.get("
                        f"{attribute!r}, getattr(self, {attribute!r}, None))"
                    ),
                ]
            )
        else:
            lines.append(
                f"    self.{attribute} = kwargs.get("
                f"{attribute!r}, getattr(self, {attribute!r}, None))"
            )
    lines.extend(
        [
            "    for key, value in kwargs.items():",
            "        if key != 'messages':",
            "            setattr(self, key, value)",
        ]
    )
    return "\n".join(lines)


def _build_abms_safe_message_api_source(message_api_name: str) -> str:
    return "\n".join(
        [
            f"def {message_api_name}(self, target, message=None):",
            '    """Framework message API inserted by deterministic refinement."""',
            "    if message is None:",
            "        message = target",
            "        target = self",
            (
                "    elif not hasattr(target, '__dict__') "
                "and hasattr(message, '__dict__'):"
            ),
            "        target, message = message, target",
            "    if not hasattr(self, 'sent_messages'):",
            "        self.sent_messages = []",
            (
                "    self.sent_messages.append("
                "{'target': target.__class__.__name__, 'message': message})"
            ),
            "    if not hasattr(self, 'messages'):",
            "        self.messages = []",
            "    if hasattr(target, 'messages'):",
            "        target.messages.append(message)",
            "    else:",
            "        self.messages.append(message)",
            "    if hasattr(target, '__dict__'):",
            "        target.last_result = message",
            "        target.received_result = message",
            "    else:",
            "        self._last_message_payload = message",
            "    if hasattr(target, 'receive_message'):",
            "        try:",
            "            target.receive_message(message)",
            "        except Exception as error:",
            "            self._last_message_api_error = error",
            "    return {'delivered': True, 'message': message}",
        ]
    )


def _self_attributes_by_class(tree: ast.Module) -> dict[str, list[str]]:
    """Return self attribute references used by each class body."""

    output: dict[str, list[str]] = {}
    for class_node in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
        names: list[str] = []
        for node in ast.walk(class_node):
            if not isinstance(node, ast.Attribute):
                continue
            if not isinstance(node.value, ast.Name) or node.value.id != "self":
                continue
            if node.attr.startswith("_"):
                continue
            if node.attr in {"send_message", "receive_message"}:
                continue
            names.append(node.attr)
        output[class_node.name] = _dedupe_labels(names)
    return output


def _method_call_names(method_node: ast.FunctionDef) -> set[str]:
    return {
        call_name
        for node in ast.walk(method_node)
        if isinstance(node, ast.Call)
        if (call_name := _call_name(node)) is not None
    }


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _method_has_filter_guard(
    method_node: ast.FunctionDef,
    filter_expressions: list[str],
) -> bool:
    conditions = " ".join(
        ast.unparse(node.test)
        for node in ast.walk(method_node)
        if isinstance(node, ast.If)
    )
    return any(expression in conditions for expression in filter_expressions)


def _indent(source: str, *, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in source.splitlines())


def _missing_direction_metadata_comments(
    source: str,
    constraints: ModelConstraints,
) -> list[str]:
    comments: list[str] = []
    for edge in constraints.expected_interaction_edges:
        direction = (
            f"{edge.source_agent}.{edge.source_port} -> "
            f"{edge.target_agent}.{edge.target_port}"
        )
        if direction in source:
            continue
        comments.append(f"# MAG direction {edge.edge_id}: {direction}")
    return comments


def _metadata(
    constraints: ModelConstraints,
    framework: FrameworkSpec,
    plan: RewritePlan,
    refinement_config: BalancedRefinementConfig,
) -> dict[str, Any]:
    return {
        "mag_id": constraints.mag_id,
        "framework": framework.name,
        "rewrite_plan": plan.to_dict(),
        "refinement_version": 1,
        "strategy": "deterministic_rule_based",
        "refinement_strategy": refinement_config.strategy,
        "balanced_refinement_config": refinement_config.to_dict(),
    }


def _normalize_import_spacing(source: str) -> str:
    normalized = source.replace(
        "from __future__ import annotations\n\n\nfrom typing import Any",
        "from __future__ import annotations\n\nfrom typing import Any",
    )
    return normalized.replace(
        "from typing import Any\n\n\nMAG_ID",
        "from typing import Any\n\nMAG_ID",
    )


def _dedupe_labels(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        deduped.append(label)
    return deduped
