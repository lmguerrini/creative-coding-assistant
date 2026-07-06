"""Shared contracts for planning runtime decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from creative_coding_assistant.orchestration._metadata_utils import (
    PlanningMetadata,
    PlanningMetadataItem,
)


@dataclass(frozen=True)
class PlanningRuntimeArtifacts:
    """Derived planning artifacts stored on workflow and prompt-input state."""

    creative_strategy: Any
    creative_intent: Any
    creative_hierarchy: Any
    creative_techniques: Any
    creative_plan: Any
    creative_constraints: Any
    creative_constraint_priorities: Any
    runtime_capabilities: Any
    creative_tradeoffs: Any
    creative_quality_prediction: Any
    symbolic_narrative: Any
    creative_composition: Any
    procedural_structure: Any
    generative_structure: Any
    semantic_motif: Any
    emotional_consistency: Any
    cross_modality: Any
    audio_visual_scene: Any
    artifact_plan: Any
    artifact_dependency_graph: Any
    runtime_compatibility: Any
    artifact_capability_matrix: Any
    multi_artifact_strategy: Any
    artifact_critic: Any
    artifact_refiner: Any
    artifact_intelligence_synthesis: Any
    artifact_merge_planner: Any
    artifact_export_intelligence: Any
    artifact_engine_contracts: Any
    evaluation_engine_contracts: Any
    creative_critic: Any
    self_evaluation: Any
    creative_improvement_planner: Any
    reflection_loop: Any
    creative_confidence: Any
    creative_score: Any
    consistency_validation: Any
    evaluation_report: Any


PLANNING_RUNTIME_UPDATE_FIELDS = (
    "creative_strategy",
    "creative_intent",
    "creative_hierarchy",
    "creative_techniques",
    "creative_plan",
    "creative_constraints",
    "creative_constraint_priorities",
    "runtime_capabilities",
    "creative_tradeoffs",
    "creative_quality_prediction",
    "symbolic_narrative",
    "creative_composition",
    "procedural_structure",
    "generative_structure",
    "semantic_motif",
    "emotional_consistency",
    "cross_modality",
    "audio_visual_scene",
    "artifact_plan",
    "artifact_dependency_graph",
    "runtime_compatibility",
    "artifact_capability_matrix",
    "multi_artifact_strategy",
    "artifact_critic",
    "artifact_refiner",
    "artifact_intelligence_synthesis",
    "artifact_merge_planner",
    "artifact_export_intelligence",
    "artifact_engine_contracts",
    "evaluation_engine_contracts",
    "creative_critic",
    "self_evaluation",
    "creative_improvement_planner",
    "reflection_loop",
    "creative_confidence",
    "creative_score",
    "consistency_validation",
    "evaluation_report",
)

PLANNING_EVENT_PAYLOAD_FIELDS = (
    "creative_intent",
    "creative_hierarchy",
    "creative_strategy",
    "creative_techniques",
    "creative_plan",
    "creative_constraints",
    "creative_constraint_priorities",
    "runtime_capabilities",
    "creative_tradeoffs",
    "creative_quality_prediction",
    "symbolic_narrative",
    "creative_composition",
    "procedural_structure",
    "generative_structure",
    "semantic_motif",
    "emotional_consistency",
    "cross_modality",
    "audio_visual_scene",
    "artifact_plan",
    "artifact_dependency_graph",
    "runtime_compatibility",
    "artifact_capability_matrix",
    "multi_artifact_strategy",
    "artifact_critic",
    "artifact_refiner",
    "artifact_intelligence_synthesis",
    "artifact_merge_planner",
    "artifact_export_intelligence",
    "artifact_engine_contracts",
    "evaluation_engine_contracts",
    "creative_critic",
    "self_evaluation",
    "creative_improvement_planner",
    "reflection_loop",
    "creative_confidence",
    "creative_score",
    "consistency_validation",
    "evaluation_report",
)


def _evaluation_planning_metadata(
    *items: PlanningMetadataItem,
) -> PlanningMetadata:
    return items
