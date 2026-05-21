"""Code refinement interfaces."""

from magcoder.refinement.ast_analyzer import ASTAnalysisResult, analyze_python_source
from magcoder.refinement.checker import (
    ConsistencyReport,
    check_consistency,
    save_consistency_report,
)
from magcoder.refinement.constraints import (
    GuardRequirement,
    InteractionConstraint,
    ModelConstraints,
    constraints_with_interaction_contracts,
    extract_model_constraints,
)
from magcoder.refinement.contracts import (
    InteractionMethodContract,
    build_interaction_contracts,
    build_interaction_contracts_from_constraints,
    edge_method_name,
    target_method_name,
    to_snake_case,
)
from magcoder.refinement.framework import (
    FrameworkSpec,
    load_framework_spec_from_yaml,
    save_framework_spec_to_yaml,
)
from magcoder.refinement.refiner import (
    BalancedRefinementConfig,
    RefinedCodeResult,
    refine_code,
    save_refined_code_result,
)
from magcoder.refinement.rewrite_plan import (
    RewriteAction,
    RewritePlan,
    build_rewrite_plan,
)

__all__ = [
    "ASTAnalysisResult",
    "ConsistencyReport",
    "FrameworkSpec",
    "GuardRequirement",
    "InteractionConstraint",
    "InteractionMethodContract",
    "ModelConstraints",
    "BalancedRefinementConfig",
    "RefinedCodeResult",
    "RewriteAction",
    "RewritePlan",
    "analyze_python_source",
    "build_interaction_contracts",
    "build_interaction_contracts_from_constraints",
    "build_rewrite_plan",
    "check_consistency",
    "constraints_with_interaction_contracts",
    "extract_model_constraints",
    "load_framework_spec_from_yaml",
    "refine_code",
    "save_consistency_report",
    "save_framework_spec_to_yaml",
    "save_refined_code_result",
    "edge_method_name",
    "target_method_name",
    "to_snake_case",
]
