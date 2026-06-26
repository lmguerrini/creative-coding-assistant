"""Passive escalation policy metadata for V3.6 preparation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EscalationPolicyStage = Literal[
    "human_review_advisory",
    "future_agent_advisory",
]

ESCALATION_POLICY_RULE_SERIALIZATION_VERSION = "escalation_policy_rule.v1"
ESCALATION_POLICY_REGISTRY_SERIALIZATION_VERSION = "escalation_policy_registry.v1"
ESCALATION_POLICY_REGISTRY_AUTHORITY_BOUNDARY = (
    "Escalation policy metadata describes advisory future escalation rules, "
    "triggering signals, source registries, and blocked runtime behaviors "
    "only; it does not evaluate policy, trigger escalation, route providers "
    "or models, select runtimes, retry work, invoke agents, execute artifacts, "
    "or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_control",
    "retry_or_refinement_triggering",
    "agent_invocation",
    "artifact_execution",
    "generated_output_modification",
)


class EscalationPolicyRule(BaseModel):
    """Metadata-only future escalation policy rule."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    rule_id: str = Field(min_length=1, max_length=80)
    rule_name: str = Field(min_length=1, max_length=140)
    policy_stage: EscalationPolicyStage
    authority_boundary: str = Field(min_length=1, max_length=900)
    source_contract_registries: tuple[str, ...] = Field(min_length=1, max_length=6)
    trigger_signals: tuple[str, ...] = Field(min_length=1, max_length=12)
    evidence_sources: tuple[str, ...] = Field(min_length=1, max_length=12)
    advisory_outcome: str = Field(min_length=1, max_length=180)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=1, max_length=12)
    serialization_version: Literal["escalation_policy_rule.v1"] = (
        ESCALATION_POLICY_RULE_SERIALIZATION_VERSION
    )


class EscalationPolicyRegistry(BaseModel):
    """Stable metadata registry for future escalation policy preparation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["escalation_policy_registry"] = "escalation_policy_registry"
    serialization_version: Literal["escalation_policy_registry.v1"] = (
        ESCALATION_POLICY_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ESCALATION_POLICY_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    rules: tuple[EscalationPolicyRule, ...] = Field(min_length=5, max_length=5)
    rule_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    rule_count: int = Field(ge=5, le=5)
    source_contract_registries: tuple[str, ...] = Field(min_length=4, max_length=4)
    metadata_only: Literal[True] = True


def escalation_policy_registry() -> EscalationPolicyRegistry:
    """Return the static future escalation policy registry."""

    return ESCALATION_POLICY_REGISTRY


def escalation_policy_by_id(rule_id: str) -> EscalationPolicyRule | None:
    """Return one future escalation policy rule without changing behavior."""

    for rule in ESCALATION_POLICY_RULES:
        if rule.rule_id == rule_id:
            return rule
    return None


def _rule(
    *,
    rule_id: str,
    rule_name: str,
    policy_stage: EscalationPolicyStage,
    source_contract_registries: tuple[str, ...],
    trigger_signals: tuple[str, ...],
    evidence_sources: tuple[str, ...],
    advisory_outcome: str,
) -> EscalationPolicyRule:
    return EscalationPolicyRule(
        rule_id=rule_id,
        rule_name=rule_name,
        policy_stage=policy_stage,
        authority_boundary=(
            "This rule is an advisory escalation policy metadata contract only; "
            "it does not evaluate policy, trigger escalation, invoke agents, "
            "alter routing, select runtimes, retry work, execute artifacts, or "
            "modify generated output."
        ),
        source_contract_registries=source_contract_registries,
        trigger_signals=trigger_signals,
        evidence_sources=evidence_sources,
        advisory_outcome=advisory_outcome,
        blocked_runtime_behaviors=_BLOCKED_RUNTIME_BEHAVIORS,
    )


ESCALATION_POLICY_RULES = (
    _rule(
        rule_id="missing_information_review",
        rule_name="Missing Information Human Review",
        policy_stage="human_review_advisory",
        source_contract_registries=(
            "artifact_engine_contract_registry",
            "evaluation_engine_contract_registry",
        ),
        trigger_signals=(
            "missing_information",
            "missing_dependency_risks",
            "hitl_questions",
        ),
        evidence_sources=(
            "artifact_plan",
            "artifact_dependency_graph",
            "creative_critic",
            "self_evaluation",
        ),
        advisory_outcome=(
            "Recommend human review metadata when missing information affects "
            "artifact or evaluation confidence."
        ),
    ),
    _rule(
        rule_id="artifact_risk_review",
        rule_name="Artifact Risk Human Review",
        policy_stage="human_review_advisory",
        source_contract_registries=("artifact_engine_contract_registry",),
        trigger_signals=(
            "implementation_risks",
            "risk_areas",
            "risk_assessment",
            "escalation_candidates",
        ),
        evidence_sources=(
            "artifact_plan",
            "multi_artifact_strategy",
            "artifact_critic",
            "artifact_intelligence_synthesis",
        ),
        advisory_outcome=(
            "Recommend human review metadata when artifact risks are prominent "
            "or repeated across artifact intelligence profiles."
        ),
    ),
    _rule(
        rule_id="runtime_incompatibility_review",
        rule_name="Runtime Incompatibility Human Review",
        policy_stage="human_review_advisory",
        source_contract_registries=(
            "artifact_engine_contract_registry",
            "workstation_engine_contract_registry",
        ),
        trigger_signals=(
            "unsupported_runtimes",
            "runtime_confidence",
            "capability_risks",
            "runtime_fit_status",
        ),
        evidence_sources=(
            "runtime_compatibility",
            "artifact_capability_matrix",
            "workstation_dashboard",
        ),
        advisory_outcome=(
            "Recommend human review metadata when runtime fit is weak or "
            "compatibility risks affect the requested artifact."
        ),
    ),
    _rule(
        rule_id="evaluation_confidence_review",
        rule_name="Evaluation Confidence Human Review",
        policy_stage="human_review_advisory",
        source_contract_registries=("evaluation_engine_contract_registry",),
        trigger_signals=(
            "confidence_score",
            "confidence_uncertainties",
            "evaluation_integrity",
            "hitl_recommendation",
        ),
        evidence_sources=(
            "creative_confidence",
            "creative_score",
            "consistency_validation",
            "evaluation_reports",
        ),
        advisory_outcome=(
            "Recommend human review metadata when evaluation confidence or "
            "consistency signals indicate uncertainty."
        ),
    ),
    _rule(
        rule_id="future_agent_escalation_readiness",
        rule_name="Future Agent Escalation Readiness",
        policy_stage="future_agent_advisory",
        source_contract_registries=(
            "agent_capability_registry",
            "artifact_engine_contract_registry",
            "evaluation_engine_contract_registry",
        ),
        trigger_signals=(
            "escalation_recommendation",
            "adaptive_multi_agent_escalation",
            "future_agent_hooks",
            "blocked_runtime_behaviors",
        ),
        evidence_sources=(
            "agent_capability_registry",
            "creative_confidence",
            "artifact_engine_contracts",
            "evaluation_engine_contracts",
        ),
        advisory_outcome=(
            "Prepare future escalation context metadata without invoking agents "
            "or changing V3 workflow control."
        ),
    ),
)

ESCALATION_POLICY_REGISTRY = EscalationPolicyRegistry(
    rules=ESCALATION_POLICY_RULES,
    rule_ids=tuple(rule.rule_id for rule in ESCALATION_POLICY_RULES),
    rule_count=len(ESCALATION_POLICY_RULES),
    source_contract_registries=(
        "agent_capability_registry",
        "artifact_engine_contract_registry",
        "evaluation_engine_contract_registry",
        "workstation_engine_contract_registry",
    ),
)
