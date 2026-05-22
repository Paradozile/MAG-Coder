"""Hybrid semantic input derived from validated MAG models."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from magcoder.models import AgentClass, AttributeFilter, InteractionEdge, MAGModel
from magcoder.refinement.contracts import edge_method_name, target_method_name

HybridMode = Literal["graph-only", "graph+nl"]


@dataclass(slots=True)
class HybridAgentSemantics:
    """Implementation-relevant semantics for one MAG agent class."""

    agent_id: str
    ports: list[dict[str, str | None]]
    attributes: list[dict[str, str | None]]
    role_summary: str
    behavioral_responsibility: str
    state_implications: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render this semantic record as plain Python containers."""

        return asdict(self)


@dataclass(slots=True)
class HybridEdgeSemantics:
    """Implementation-relevant semantics for one MAG interaction edge."""

    edge_id: str
    source: dict[str, str]
    target: dict[str, str]
    filter_id: str | None
    interaction_intent: str
    communication_direction: str
    implementation_hint: str

    def to_dict(self) -> dict[str, Any]:
        """Render this semantic record as plain Python containers."""

        return asdict(self)


@dataclass(slots=True)
class HybridFilterSemantics:
    """Implementation-relevant semantics for one MAG attribute filter."""

    filter_id: str
    original_expressions: list[str]
    condition_meaning: str
    guard_placement_hint: str
    implementation_hint: str

    def to_dict(self) -> dict[str, Any]:
        """Render this semantic record as plain Python containers."""

        return asdict(self)


@dataclass(slots=True)
class HybridInput:
    """LLM-consumable hybrid semantic representation H=(A',E',F')."""

    mag_id: str
    mode: HybridMode
    agents: list[HybridAgentSemantics]
    edges: list[HybridEdgeSemantics]
    filters: list[HybridFilterSemantics]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render the hybrid input as plain Python containers."""

        return {
            "mag_id": self.mag_id,
            "mode": self.mode,
            "agents": [agent.to_dict() for agent in self.agents],
            "edges": [edge.to_dict() for edge in self.edges],
            "filters": [
                attribute_filter.to_dict()
                for attribute_filter in self.filters
            ],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Render the hybrid input as stable, indented JSON."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_prompt_text(self, render_mode: str = "default") -> str:
        """Render LLM-ready prompt content for later code generation."""

        if render_mode in {"compact_contract", "graph_contract_only", "local_compact"}:
            return self.to_compact_prompt_text()

        lines = [
            f"# Hybrid Semantic Input for MAG: {self.mag_id}",
            f"Mode: {self.mode}",
            "",
            "## Agents",
        ]

        for agent in self.agents:
            lines.extend(
                [
                    f"- Agent: {agent.agent_id}",
                    f"  Role summary: {agent.role_summary}",
                    (
                        "  Behavioral responsibility: "
                        f"{agent.behavioral_responsibility}"
                    ),
                    "  Ports:",
                ]
            )
            lines.extend(f"    - {_format_port(port)}" for port in agent.ports)
            lines.append("  Attributes:")
            if agent.attributes:
                lines.extend(
                    f"    - {_format_attribute(attribute)}"
                    for attribute in agent.attributes
                )
            else:
                lines.append("    - No explicit attributes declared.")
            lines.append("  State implications:")
            lines.extend(
                f"    - {implication}" for implication in agent.state_implications
            )

        lines.extend(["", "## Interactions"])
        for edge in self.edges:
            lines.extend(
                [
                    f"- Edge: {edge.edge_id}",
                    (
                        "  Source: "
                        f"{edge.source['agent_id']}.{edge.source['port']}"
                    ),
                    (
                        "  Target: "
                        f"{edge.target['agent_id']}.{edge.target['port']}"
                    ),
                    f"  Intent: {edge.interaction_intent}",
                    f"  Direction: {edge.communication_direction}",
                    f"  Implementation hint: {edge.implementation_hint}",
                ]
            )

        lines.extend(["", "## Required interaction method contracts"])
        contracts = _prompt_contracts(self)
        if contracts:
            for contract in contracts:
                filter_expression = contract["filter_expression"]
                message_api_name = contract["message_api_name"]
                lines.extend(
                    [
                        (
                            f"- For edge {contract['edge_id']}, implement method "
                            f"{contract['required_method_name']} in class "
                            f"{contract['source_class_name']}."
                        ),
                        (
                            "  Signature: method should take target or "
                            "destination agent as an argument."
                        ),
                        (
                            "  Required method name must be used exactly; do "
                            "not rename it."
                        ),
                        (
                            "  Fallback compatibility names, if already "
                            f"present: {', '.join(contract['fallback_method_names'])}."
                        ),
                        (
                            "  Preserve filter guard: "
                            f"{_filter_expression_label(filter_expression)}."
                        ),
                        (
                            "  Message API requirement: "
                            f"call {message_api_name} when delivering the interaction."
                        ),
                    ]
                )
        else:
            lines.append("- No interaction method contracts declared.")

        lines.extend(["", "## Filters / Conditions"])
        if self.filters:
            for attribute_filter in self.filters:
                lines.extend(
                    [
                        f"- Filter: {attribute_filter.filter_id}",
                        (
                            "  Original expressions: "
                            + "; ".join(attribute_filter.original_expressions)
                        ),
                        f"  Meaning: {attribute_filter.condition_meaning}",
                        (
                            "  Guard placement: "
                            f"{attribute_filter.guard_placement_hint}"
                        ),
                        (
                            "  Implementation hint: "
                            f"{attribute_filter.implementation_hint}"
                        ),
                    ]
                )
        else:
            lines.append("- No filters declared.")

        lines.extend(
            [
                "",
                "## Code generation requirements",
                (
                    "- Generate interaction code from the semantic records, "
                    "not from raw YAML field order."
                ),
                (
                    "- Represent each interaction as a directed transfer from "
                    "the source output port to the target input port."
                ),
                "- Evaluate referenced filters before executing interactions.",
                (
                    "- Keep agent state reads and writes explicit and tied to "
                    "declared attributes."
                ),
                "- Preserve MAG identifiers in generated code comments or metadata.",
                "",
                "## Constraints to preserve",
                "- Do not reverse source and target agents or ports.",
                "- Do not execute an edge when its referenced filter evaluates false.",
                "- Do not invent undeclared agents, ports, filters, or attributes.",
                "- Do not integrate an external LLM API in this phase.",
            ]
        )
        return "\n".join(lines) + "\n"

    def to_compact_prompt_text(self) -> str:
        """Render a compact local-model prompt centered on formal contracts."""

        lines = [
            f"MAG_ID: {self.mag_id}",
            f"MODE: {self.mode}",
            "",
            "AGENTS:",
        ]
        for agent in self.agents:
            attributes = ", ".join(
                str(attribute["name"]) for attribute in agent.attributes
            )
            ports = ", ".join(_format_port(port) for port in agent.ports)
            lines.append(
                f"- {agent.agent_id}; ports=[{ports}]; attributes=[{attributes}]"
            )

        lines.extend(["", "EDGES:"])
        for edge in self.edges:
            lines.append(
                "- "
                f"{edge.edge_id}: {edge.source['agent_id']}.{edge.source['port']} "
                f"-> {edge.target['agent_id']}.{edge.target['port']}; "
                f"filter={edge.filter_id or 'none'}"
            )

        lines.extend(["", "FILTERS:"])
        if self.filters:
            for attribute_filter in self.filters:
                expressions = " and ".join(attribute_filter.original_expressions)
                lines.append(f"- {attribute_filter.filter_id}: {expressions}")
        else:
            lines.append("- none")

        lines.extend(["", "Required interaction method contracts:"])
        for contract in _prompt_contracts(self):
            fallback_names = ", ".join(contract["fallback_method_names"]) or "none"
            lines.extend(
                [
                    (
                        f"- edge_id={contract['edge_id']}; "
                        f"class={contract['source_class_name']}; "
                        f"method={contract['required_method_name']}; "
                        f"fallbacks={fallback_names}"
                    ),
                    (
                        "  source_target="
                        f"{contract['source']} -> {contract['target']}"
                    ),
                    (
                        "  filter_guard="
                        f"{_filter_expression_label(contract['filter_expression'])}"
                    ),
                    f"  message_api={contract['message_api_name']}",
                ]
            )

        lines.extend(
            [
                "",
                "CODE_REQUIREMENTS:",
                "- Return valid Python only.",
                "- Implement every required method name exactly as specified.",
                "- Each required interaction method must call self.send_message(...).",
                "- Preserve filter guards before sending messages.",
                "- Preserve source/target direction in comments or metadata.",
            ]
        )
        return "\n".join(lines) + "\n"


def build_hybrid_input(mag: MAGModel, mode: str = "graph+nl") -> HybridInput:
    """Build a deterministic hybrid semantic input from a validated MAG model."""

    normalized_mode = _normalize_mode(mode)
    include_nl = normalized_mode == "graph+nl"
    mag.validate()

    filter_usage = _filter_usage_by_edge(mag.edges)
    return HybridInput(
        mag_id=mag.id,
        mode=normalized_mode,
        agents=[
            _build_agent_semantics(agent, include_nl) for agent in mag.agents
        ],
        edges=[
            _build_edge_semantics(edge, include_nl) for edge in mag.edges
        ],
        filters=[
            _build_filter_semantics(
                attribute_filter=attribute_filter,
                edge_ids=filter_usage.get(attribute_filter.id, []),
                include_nl=include_nl,
            )
            for attribute_filter in mag.filters
        ],
        metadata={
            "source_mag_metadata": mag.metadata,
            "semantic_version": 1,
            "interaction_contracts": _metadata_contracts(mag),
        },
    )


def _normalize_mode(mode: str) -> HybridMode:
    if mode == "graph":
        return "graph-only"
    if mode == "graph-only":
        return "graph-only"
    if mode == "graph+nl":
        return "graph+nl"
    msg = "Hybrid input mode must be 'graph-only', 'graph', or 'graph+nl'."
    raise ValueError(msg)


def _build_agent_semantics(
    agent: AgentClass,
    include_nl: bool,
) -> HybridAgentSemantics:
    return HybridAgentSemantics(
        agent_id=agent.id,
        ports=[_port_to_semantics(port, include_nl) for port in agent.ports],
        attributes=[
            _attribute_to_semantics(attribute, include_nl)
            for attribute in agent.attributes
        ],
        role_summary=_agent_role_summary(agent, include_nl),
        behavioral_responsibility=_agent_behavioral_responsibility(agent),
        state_implications=_agent_state_implications(agent, include_nl),
    )


def _build_edge_semantics(
    edge: InteractionEdge,
    include_nl: bool,
) -> HybridEdgeSemantics:
    communication_direction = (
        f"{edge.src_agent}.{edge.src_port} -> {edge.dst_agent}.{edge.dst_port}"
    )
    return HybridEdgeSemantics(
        edge_id=edge.id,
        source={"agent_id": edge.src_agent, "port": edge.src_port},
        target={"agent_id": edge.dst_agent, "port": edge.dst_port},
        filter_id=edge.filter_id,
        interaction_intent=_edge_intent(edge, include_nl),
        communication_direction=communication_direction,
        implementation_hint=_edge_implementation_hint(edge, communication_direction),
    )


def _build_filter_semantics(
    attribute_filter: AttributeFilter,
    edge_ids: list[str],
    include_nl: bool,
) -> HybridFilterSemantics:
    return HybridFilterSemantics(
        filter_id=attribute_filter.id,
        original_expressions=list(attribute_filter.expressions),
        condition_meaning=_filter_condition_meaning(attribute_filter, include_nl),
        guard_placement_hint=_filter_guard_placement_hint(edge_ids),
        implementation_hint=_filter_implementation_hint(attribute_filter),
    )


def _port_to_semantics(port: Any, include_nl: bool) -> dict[str, str | None]:
    return {
        "name": port.name,
        "direction": port.direction,
        "description": port.description if include_nl else None,
    }


def _attribute_to_semantics(
    attribute: Any,
    include_nl: bool,
) -> dict[str, str | None]:
    return {
        "name": attribute.name,
        "type": attribute.type,
        "description": attribute.description if include_nl else None,
    }


def _agent_role_summary(agent: AgentClass, include_nl: bool) -> str:
    if include_nl and agent.nl_description:
        return agent.nl_description

    input_ports = _port_names_by_direction(agent, "input")
    output_ports = _port_names_by_direction(agent, "output")
    capabilities: list[str] = []
    if input_ports:
        capabilities.append(f"consumes inputs through {', '.join(input_ports)}")
    if output_ports:
        capabilities.append(f"produces outputs through {', '.join(output_ports)}")
    if not capabilities:
        return f"{agent.id} participates in the MAG without declared ports."
    return f"{agent.id} " + " and ".join(capabilities) + "."


def _agent_behavioral_responsibility(agent: AgentClass) -> str:
    input_ports = _port_names_by_direction(agent, "input")
    output_ports = _port_names_by_direction(agent, "output")
    responsibilities: list[str] = []
    if input_ports:
        responsibilities.append(
            f"handle inbound interactions on {', '.join(input_ports)}"
        )
    if output_ports:
        responsibilities.append(
            f"emit outbound interactions through {', '.join(output_ports)}"
        )
    if not responsibilities:
        return "Maintain declared state and participate only when referenced."
    return "Should " + " and ".join(responsibilities) + "."


def _agent_state_implications(agent: AgentClass, include_nl: bool) -> list[str]:
    if not agent.attributes:
        return ["No explicit state attributes were declared."]

    implications: list[str] = []
    for attribute in agent.attributes:
        type_part = f" ({attribute.type})" if attribute.type else ""
        description_part = (
            f": {_trim_terminal_punctuation(attribute.description)}"
            if include_nl and attribute.description
            else ""
        )
        implications.append(
            f"Maintain state field {attribute.name}{type_part}{description_part}."
        )
    return implications


def _edge_intent(edge: InteractionEdge, include_nl: bool) -> str:
    if include_nl and edge.nl_description:
        return edge.nl_description
    return (
        f"Transfer interaction data from {edge.src_agent}.{edge.src_port} "
        f"to {edge.dst_agent}.{edge.dst_port}."
    )


def _edge_implementation_hint(
    edge: InteractionEdge,
    communication_direction: str,
) -> str:
    filter_hint = (
        f" Evaluate filter {edge.filter_id!r} before delivery."
        if edge.filter_id
        else " No filter guard is declared for this edge."
    )
    return (
        f"Implement a directed handler for {communication_direction}."
        f"{filter_hint} Keep the source and target port names traceable."
    )


def _filter_condition_meaning(
    attribute_filter: AttributeFilter,
    include_nl: bool,
) -> str:
    if include_nl and attribute_filter.nl_description:
        return attribute_filter.nl_description
    if attribute_filter.expressions:
        expressions = " and ".join(attribute_filter.expressions)
        return f"The interaction is allowed only when {expressions}."
    return "No explicit condition was declared."


def _filter_guard_placement_hint(edge_ids: list[str]) -> str:
    if edge_ids:
        joined_edge_ids = ", ".join(edge_ids)
        return f"Apply before executing edges: {joined_edge_ids}."
    return "No edge references this filter yet."


def _filter_implementation_hint(attribute_filter: AttributeFilter) -> str:
    if attribute_filter.expressions:
        return (
            "Translate expressions into boolean guard checks; skip the "
            "interaction when any expression is false."
        )
    return "Treat as an always-true guard until expressions are provided."


def _filter_usage_by_edge(edges: list[InteractionEdge]) -> dict[str, list[str]]:
    usage: dict[str, list[str]] = {}
    for edge in edges:
        if edge.filter_id is None:
            continue
        usage.setdefault(edge.filter_id, []).append(edge.id)
    return usage


def _metadata_contracts(mag: MAGModel) -> list[dict[str, Any]]:
    filters_by_id = {
        attribute_filter.id: list(attribute_filter.expressions)
        for attribute_filter in mag.filters
    }
    return [
        {
            "edge_id": edge.id,
            "required_method_name": edge_method_name(edge.id),
            "fallback_method_names": [
                target_method_name("interact_with", edge.dst_agent)
            ],
                "source_class_name": _class_name(edge.src_agent),
                "target_class_name": _class_name(edge.dst_agent),
                "source": f"{edge.src_agent}.{edge.src_port}",
                "target": f"{edge.dst_agent}.{edge.dst_port}",
                "filter_id": edge.filter_id,
            "filter_expressions": (
                [] if edge.filter_id is None else filters_by_id[edge.filter_id]
            ),
            "message_api_name": "send_message",
            "strategy": "edge_id",
        }
        for edge in mag.edges
    ]


def _prompt_contracts(hybrid_input: HybridInput) -> list[dict[str, Any]]:
    raw_contracts = hybrid_input.metadata.get("interaction_contracts", [])
    if not isinstance(raw_contracts, list):
        return []
    contracts: list[dict[str, Any]] = []
    for contract in raw_contracts:
        if not isinstance(contract, dict):
            continue
        filter_expressions = contract.get("filter_expressions", [])
        fallback_names = contract.get("fallback_method_names", [])
        contracts.append(
            {
                "edge_id": str(contract.get("edge_id", "")),
                "required_method_name": str(
                    contract.get("required_method_name", "")
                ),
                "fallback_method_names": [
                    str(name) for name in fallback_names if isinstance(name, str)
                ],
                "source_class_name": str(contract.get("source_class_name", "")),
                "source": str(contract.get("source", "")),
                "target": str(contract.get("target", "")),
                "filter_expression": " and ".join(
                    str(expression)
                    for expression in filter_expressions
                    if isinstance(expression, str)
                ),
                "message_api_name": str(
                    contract.get("message_api_name", "send_message")
                ),
            }
        )
    return contracts


def _filter_expression_label(filter_expression: str) -> str:
    return filter_expression if filter_expression else "no filter"


def _class_name(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value)
    if not words:
        return "GeneratedAgent"
    class_name = "".join(word[:1].upper() + word[1:] for word in words)
    if class_name[0].isdigit():
        return f"Agent{class_name}"
    return class_name


def _port_names_by_direction(agent: AgentClass, direction: str) -> list[str]:
    return [port.name for port in agent.ports if port.direction == direction]


def _format_port(port: dict[str, str | None]) -> str:
    description = _format_optional_suffix(port.get("description"))
    return f"{port['name']} ({port['direction']}){description}"


def _format_attribute(attribute: dict[str, str | None]) -> str:
    type_part = f" ({attribute['type']})" if attribute.get("type") else ""
    description = _format_optional_suffix(attribute.get("description"))
    return f"{attribute['name']}{type_part}{description}"


def _format_optional_suffix(value: str | None) -> str:
    if not value:
        return ""
    return f": {value}"


def _trim_terminal_punctuation(value: str) -> str:
    return value.rstrip().rstrip(".!?")
