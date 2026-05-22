"""Input mode adapters for Full MAG-Bench cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from magcoder.benchmark.full_runner.case_loader import FullBenchRunCase
from magcoder.graphics import load_mag_from_graphical_artifact
from magcoder.models import MAGModel, load_mag_from_yaml
from magcoder.semantic import HybridInput, build_hybrid_input

FullBenchInputMode = Literal[
    "zero_shot_direct",
    "few_shot_direct",
    "structured_graph_only",
    "structured_graph_nl",
    "graphical_graph_only",
    "graphical_graph_nl",
    "without_nl",
]

SUPPORTED_FULL_INPUT_MODES: tuple[str, ...] = (
    "zero_shot_direct",
    "few_shot_direct",
    "structured_graph_only",
    "structured_graph_nl",
    "graphical_graph_only",
    "graphical_graph_nl",
    "without_nl",
)


@dataclass(slots=True)
class FullBenchPromptInput:
    """Unified prompt input for one case/input-mode pair."""

    case_id: str
    input_mode: str
    mag: MAGModel
    hybrid_input: HybridInput
    prompt_text: str
    source_kind: str
    hybrid_mode: str


def build_full_bench_prompt_input(
    case: FullBenchRunCase,
    input_mode: str,
) -> FullBenchPromptInput:
    """Load MAG and build HybridInput for a Full MAG-Bench input mode."""

    if input_mode not in SUPPORTED_FULL_INPUT_MODES:
        supported = ", ".join(SUPPORTED_FULL_INPUT_MODES)
        raise ValueError(
            f"Unsupported Full MAG-Bench input mode: {input_mode}. {supported}"
        )
    structured = input_mode.startswith("structured_") or input_mode in {
        "zero_shot_direct",
        "few_shot_direct",
        "without_nl",
    }
    use_nl = input_mode.endswith("_graph_nl")
    mag = (
        load_mag_from_yaml(case.mag_yaml_path)
        if structured
        else load_mag_from_graphical_artifact(case.dot_path, case.manifest_path)
    )
    hybrid_mode = "graph+nl" if use_nl else "graph-only"
    hybrid_input = build_hybrid_input(mag, mode=hybrid_mode)
    if input_mode == "zero_shot_direct":
        prompt_text = _direct_prompt(case, few_shot=False)
    elif input_mode == "few_shot_direct":
        prompt_text = _direct_prompt(case, few_shot=True)
    else:
        prompt_text = _append_functional_interface_contract(
            hybrid_input.to_prompt_text(),
            case=case,
            use_nl=use_nl,
        )
    return FullBenchPromptInput(
        case_id=case.case_id,
        input_mode=input_mode,
        mag=mag,
        hybrid_input=hybrid_input,
        prompt_text=prompt_text,
        source_kind=mode_source_kind(input_mode),
        hybrid_mode="direct" if input_mode.endswith("_direct") else hybrid_mode,
    )


def mode_uses_nl(input_mode: str) -> bool:
    """Return whether an input mode uses natural-language descriptions."""

    return input_mode.endswith("_graph_nl") or input_mode == "few_shot_direct"


def mode_source_kind(input_mode: str) -> str:
    """Return structured or graphical source kind for an input mode."""

    if input_mode.startswith("structured_") or input_mode == "without_nl":
        return "structured"
    if input_mode.startswith("graphical_"):
        return "graphical"
    if input_mode.endswith("_direct"):
        return "direct"
    raise ValueError(f"Unsupported Full MAG-Bench input mode: {input_mode}")


def mode_family(input_mode: str) -> str:
    """Return the RQ2 mode family for an input mode."""

    if input_mode.endswith("_direct"):
        return "direct"
    return mode_source_kind(input_mode)


def mode_nl_condition(input_mode: str) -> str:
    """Return graph_nl or graph_only for an input mode."""

    if input_mode.endswith("_graph_nl"):
        return "graph_nl"
    if input_mode.endswith("_graph_only") or input_mode == "without_nl":
        return "graph_only"
    if input_mode == "zero_shot_direct":
        return "zero_shot"
    if input_mode == "few_shot_direct":
        return "few_shot"
    raise ValueError(f"Unsupported Full MAG-Bench input mode: {input_mode}")


def baseline_family(input_mode: str, method: str) -> str:
    """Return a paper-facing baseline family label."""

    if input_mode == "zero_shot_direct":
        return "zero_shot_direct"
    if input_mode == "few_shot_direct":
        return "few_shot_direct"
    if method in {"without_ast", "no_refinement"}:
        return "without_ast"
    if input_mode == "without_nl" or input_mode.endswith("_graph_only"):
        return "without_nl"
    if method in {"mag_coder_full", "full_with_refinement"}:
        return "mag_coder_full"
    return "custom"


def _append_functional_interface_contract(
    prompt_text: str,
    *,
    case: FullBenchRunCase,
    use_nl: bool,
) -> str:
    adapter_spec = case.entry_point_adapter_spec
    if adapter_spec is None:
        return prompt_text
    lines = [
        "",
        "Functional interface requirement:",
        (
            f"- Candidate code MUST expose a source-compatible function named "
            f"{adapter_spec.source_entry_point}(...)."
        ),
        "- The function MUST delegate to generated MAG agent/object interaction code.",
        (
            f"- Required candidate agent class: "
            f"{adapter_spec.candidate_agent_class}."
        ),
        (
            f"- Required interaction method: "
            f"{adapter_spec.candidate_interaction_method}."
        ),
        (
            f"- Objectified entry point: "
            f"{adapter_spec.objectified_entry_point}."
        ),
        "- Initialize every MAG attribute in each agent __init__ method.",
        "- Interaction methods MUST accept self and destination/target agent.",
        "- Do not make source entry points recursively call themselves.",
        "- Do not rely on implicit globals or undefined helper symbols.",
        (
            "- Define send_message(target, message) or the configured framework "
            "message API."
        ),
        "- Message APIs MUST record sent messages in self.sent_messages.",
        (
            "- Interaction methods MUST call send_message when an edge "
            "communication occurs."
        ),
        "- Message APIs should produce observable target state such as last_result.",
        (
            "- Method signatures must accept peer/target when an interaction "
            "uses another agent."
        ),
        "- Return or expose observable outputs required by the ABMS oracle.",
        (
            "- Preserve the source-compatible function return value while adding "
            "message/state tracking for ABMS evaluation."
        ),
        (
            "- If source-entry behavior and ABMS observation need different "
            "paths, keep both paths explicit instead of overwriting one."
        ),
        (
            "- Do not bypass the agent interaction with a standalone copy of the "
            "source solution."
        ),
    ]
    abms_spec = case.abms_oracle_spec
    if abms_spec is not None:
        lines.extend(
            [
                "ABMS oracle interface requirement:",
                (
                    f"- Required candidate symbols: "
                    f"{', '.join(abms_spec.required_candidate_symbols)}."
                ),
                (
                    f"- Expected interaction call: "
                    f"{_format_interaction_call(abms_spec.interaction_sequence)}."
                ),
                "- The interaction must emit at least one framework message.",
            ]
        )
    if use_nl:
        objectified_prompt = str(case.objectified_task.get("objectified_prompt", ""))
        expected_behavior = str(case.objectified_task.get("expected_behavior", ""))
        if objectified_prompt:
            lines.append(f"- Objectified prompt: {objectified_prompt}")
        if expected_behavior:
            lines.append(f"- Expected behavior: {expected_behavior}")
    return prompt_text.rstrip() + "\n" + "\n".join(lines) + "\n"


def _direct_prompt(case: FullBenchRunCase, *, few_shot: bool) -> str:
    source_prompt = str(
        case.source_task.get("prompt")
        or case.source_task.get("text")
        or case.objectified_task.get("source_prompt")
        or case.objectified_task.get("objectified_prompt")
        or ""
    ).strip()
    entry_point = str(
        case.source_task.get("entry_point")
        or case.source_task.get("entrypoint")
        or "the requested function"
    )
    lines = [
        "Generate Python code for the requested programming task.",
        "Return parseable Python only.",
        f"Required source entry point: {entry_point}",
    ]
    if few_shot:
        lines.extend(
            [
                "",
                "Example:",
                "Task: Write add_one(x) that returns x + 1.",
                "Answer:",
                "def add_one(x):",
                "    return x + 1",
            ]
        )
    if source_prompt:
        lines.extend(["", "Task:", source_prompt])
    return "\n".join(lines).rstrip() + "\n"


def _format_interaction_call(sequence: list[dict[str, object]]) -> str:
    if not sequence:
        return "none"
    step = sequence[0]
    raw_args = step.get("args", [])
    args = ", ".join(str(arg) for arg in raw_args) if isinstance(raw_args, list) else ""
    return f"{step.get('caller', 'source')}.{step.get('method', '<method>')}({args})"
