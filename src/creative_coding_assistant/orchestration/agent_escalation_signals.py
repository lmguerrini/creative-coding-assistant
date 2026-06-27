"""Passive V4.2 escalation policy signal metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

EscalationSignalCategory = Literal[
    "confidence",
    "risk",
    "ambiguity",
    "cost",
    "latency",
    "quality",
    "hitl",
]
EscalationThresholdDirection = Literal["below", "above", "present"]

ESCALATION_SIGNAL_SERIALIZATION_VERSION = "agent_escalation_signal.v1"
ESCALATION_SIGNAL_REGISTRY_SERIALIZATION_VERSION = (
    "agent_escalation_signal_registry.v1"
)
ESCALATION_SIGNAL_REGISTRY_AUTHORITY_BOUNDARY = (
    "Escalation signal metadata describes advisory confidence, risk, "
    "ambiguity, cost, latency, quality, and HITL thresholds as passive "
    "routing metadata only; it does not perform escalation, route providers, "
    "trigger HITL automatically, invoke agents, change workflow control, "
    "execute voting, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "escalation_execution",
    "provider_or_model_routing",
    "automatic_hitl_triggering",
    "agent_invocation",
    "workflow_control",
    "voting_execution",
    "generated_output_modification",
)

_SIGNAL_SPECS: tuple[
    tuple[
        str,
        EscalationSignalCategory,
        EscalationThresholdDirection,
        float,
        tuple[str, ...],
        tuple[str, ...],
    ],
    ...,
] = (
    (
        "confidence_escalation_signal",
        "confidence",
        "below",
        0.55,
        ("consensus_confidence_placeholder", "confidence_uncertainties"),
        ("evaluation_confidence_review",),
    ),
    (
        "risk_escalation_signal",
        "risk",
        "above",
        0.7,
        ("risk_assessment", "unresolved_risks", "implementation_risks"),
        ("artifact_risk_review",),
    ),
    (
        "ambiguity_escalation_signal",
        "ambiguity",
        "above",
        0.6,
        ("missing_information", "planning_gap_summary", "disagreement_points"),
        ("missing_information_review",),
    ),
    (
        "cost_escalation_signal",
        "cost",
        "above",
        0.8,
        ("estimated_cost_metadata", "cost_latency_routing_blocked"),
        ("future_agent_escalation_readiness",),
    ),
    (
        "latency_escalation_signal",
        "latency",
        "above",
        0.8,
        ("estimated_latency_metadata", "blocking_inputs"),
        ("future_agent_escalation_readiness",),
    ),
    (
        "quality_escalation_signal",
        "quality",
        "below",
        0.65,
        ("quality_signal_metadata", "quality_review_signals"),
        ("evaluation_confidence_review",),
    ),
    (
        "hitl_escalation_signal",
        "hitl",
        "present",
        1.0,
        ("hitl_questions", "human_review_signal_declared"),
        ("missing_information_review", "artifact_risk_review"),
    ),
)


class AgentEscalationSignal(BaseModel):
    """Advisory escalation signal metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=120)
    category: EscalationSignalCategory
    threshold_direction: EscalationThresholdDirection
    advisory_threshold: float = Field(ge=0, le=1)
    evidence_sources: tuple[str, ...] = Field(min_length=1, max_length=12)
    policy_rule_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    signal_boundary: str = Field(min_length=1, max_length=700)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    escalation_performed: Literal[False] = False
    provider_routing_implemented: Literal[False] = False
    automatic_hitl_triggering_implemented: Literal[False] = False
    serialization_version: Literal["agent_escalation_signal.v1"] = (
        ESCALATION_SIGNAL_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentEscalationSignalRegistry(BaseModel):
    """Stable passive V4.2 escalation signal registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_escalation_signal_registry"] = (
        "agent_escalation_signal_registry"
    )
    serialization_version: Literal["agent_escalation_signal_registry.v1"] = (
        ESCALATION_SIGNAL_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ESCALATION_SIGNAL_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    signals: tuple[AgentEscalationSignal, ...] = Field(min_length=7, max_length=7)
    signal_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    categories: tuple[EscalationSignalCategory, ...] = Field(min_length=7, max_length=7)
    source_registries: tuple[str, ...] = Field(min_length=3, max_length=5)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    escalation_performed: Literal[False] = False
    provider_routing_implemented: Literal[False] = False
    automatic_hitl_triggering_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_signals(self) -> Self:
        derived_signal_ids = tuple(signal.signal_id for signal in self.signals)
        derived_categories = tuple(signal.category for signal in self.signals)
        if self.signal_ids != derived_signal_ids:
            raise ValueError("signal_ids must match signals")
        if self.categories != derived_categories:
            raise ValueError("categories must match signals")
        if len(set(derived_categories)) != len(derived_categories):
            raise ValueError("categories must be unique")
        for signal in self.signals:
            if signal.escalation_performed:
                raise ValueError("signals must remain advisory")
        return self


def agent_escalation_signal_registry() -> AgentEscalationSignalRegistry:
    """Return passive V4.2 escalation signal metadata."""

    return AGENT_ESCALATION_SIGNAL_REGISTRY


def agent_escalation_signal_by_id(
    signal_id: str,
    registry: AgentEscalationSignalRegistry | None = None,
) -> AgentEscalationSignal | None:
    """Return one escalation signal without performing escalation."""

    source_registry = registry or AGENT_ESCALATION_SIGNAL_REGISTRY
    for signal in source_registry.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def _signal(
    spec: tuple[
        str,
        EscalationSignalCategory,
        EscalationThresholdDirection,
        float,
        tuple[str, ...],
        tuple[str, ...],
    ],
) -> AgentEscalationSignal:
    signal_id, category, direction, threshold, evidence, policy_rules = spec
    return AgentEscalationSignal(
        signal_id=signal_id,
        category=category,
        threshold_direction=direction,
        advisory_threshold=threshold,
        evidence_sources=evidence,
        policy_rule_ids=policy_rules,
        signal_boundary=(
            "Escalation signal thresholds are advisory metadata only; they do "
            "not perform escalation, route providers, trigger HITL "
            "automatically, or change workflow control."
        ),
    )


AGENT_ESCALATION_SIGNALS = tuple(_signal(spec) for spec in _SIGNAL_SPECS)
AGENT_ESCALATION_SIGNAL_REGISTRY = AgentEscalationSignalRegistry(
    signals=AGENT_ESCALATION_SIGNALS,
    signal_ids=tuple(signal.signal_id for signal in AGENT_ESCALATION_SIGNALS),
    categories=tuple(signal.category for signal in AGENT_ESCALATION_SIGNALS),
    source_registries=(
        "escalation_policy_registry",
        "consensus_builder_registry",
        "agent_capability_alignment_registry",
    ),
)
