from __future__ import annotations

from importlib import import_module

BOUNDARY_MODULES = (
    ("runtime", "service"),
    ("runtime", "routing"),
    ("runtime", "workflow_graph"),
    ("contracts", "runtime_compatibility"),
    ("contracts", "typed_failure_taxonomy"),
    ("contracts", "agent_contracts"),
    ("audit", "agent_contract_audit"),
    ("audit", "production_release_audit"),
    ("audit", "adaptive_execution_failure_path_audit"),
    ("governance", "budget_policies"),
    ("governance", "escalation_policy"),
    ("governance", "research_governance"),
    ("advisory", "runtime_recommendation_engine"),
    ("advisory", "cost_dashboard"),
    ("advisory", "model_router"),
    ("metadata", "agent_metadata"),
    ("metadata", "creative_director"),
    ("metadata", "multimodal_studio"),
)


def test_legacy_module_imports_alias_canonical_boundary_modules() -> None:
    orchestration = import_module("creative_coding_assistant.orchestration")

    for package, module in BOUNDARY_MODULES:
        legacy = import_module(f"creative_coding_assistant.orchestration.{module}")
        canonical = import_module(
            f"creative_coding_assistant.orchestration.{package}.{module}"
        )

        assert legacy is canonical
        assert getattr(orchestration, module) is canonical


def test_root_public_exports_remain_available_after_decomposition() -> None:
    orchestration = import_module("creative_coding_assistant.orchestration")

    for export_name in (
        "AssistantService",
        "RouteName",
        "derive_runtime_compatibility_profile",
        "evaluate_budget_policies",
        "agent_contract_registry",
        "agent_contract_audit_registry",
    ):
        assert getattr(orchestration, export_name) is not None
