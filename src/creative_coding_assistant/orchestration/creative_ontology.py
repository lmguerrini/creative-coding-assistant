"""V6.2 advisory creative ontology metadata."""

from __future__ import annotations

from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.domains import (
    DomainCategory,
    get_domain_categories,
    get_domain_category_label,
    get_domains_for_category,
)
from creative_coding_assistant.orchestration.creative_dna import (
    CreativeDNAPlan,
    build_creative_dna,
    creative_dna_signature_by_id,
)
from creative_coding_assistant.orchestration.creative_lineage import (
    CreativeLineagePlan,
    build_creative_lineage,
    creative_lineage_record_by_id,
)
from creative_coding_assistant.orchestration.long_term_creative_memory import (
    LongTermCreativeMemoryPlan,
    build_long_term_creative_memory,
    long_term_creative_memory_record_by_id,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

CreativeOntologyConceptKind = Literal[
    "creative_intent_ontology",
    "style_system_ontology",
    "interaction_modality_ontology",
    "artifact_lineage_ontology",
    "domain_capability_ontology",
]
CreativeOntologyStatus = Literal["candidate", "review_required", "guarded"]
CreativeOntologyConfidence = Literal["low", "medium", "high", "guarded"]
CreativeOntologyPosture = Literal["candidate", "review_required", "guarded"]
CreativeOntologyAxis = Literal[
    "intent",
    "style",
    "interaction",
    "lineage",
    "domain_capability",
]

CREATIVE_ONTOLOGY_CONCEPT_SERIALIZATION_VERSION = "creative_ontology_concept.v1"
CREATIVE_ONTOLOGY_PLAN_SERIALIZATION_VERSION = "creative_ontology_plan.v1"
DOMAIN_METADATA_REGISTRY_REF = "creative_coding_assistant.domains.SUPPORTED_DOMAINS"
CREATIVE_ONTOLOGY_AUTHORITY_BOUNDARY = (
    "V6.2 Creative Ontology models creative ontology posture as inspectable "
    "advisory metadata only; it does not write ontology storage, create "
    "ontology nodes, create ontology edges, update ontology records, delete "
    "ontology records, infer ontology relationships, mutate taxonomies, "
    "materialize semantic graphs, mutate the domain registry, apply creative "
    "lineage, apply Creative DNA, execute memory retrieval, write memory "
    "storage, consolidate memory, mutate preferences, apply personalization, "
    "change provider or model routing, execute providers, invoke agents, "
    "control workflows, mutate workflow graphs, trigger retries or refinements, "
    "mutate prompts, modify generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "ontology_storage_write",
    "ontology_node_creation",
    "ontology_edge_creation",
    "ontology_record_update",
    "ontology_record_deletion",
    "ontology_relationship_inference",
    "taxonomy_mutation",
    "semantic_graph_materialization",
    "domain_registry_mutation",
    "creative_lineage_application",
    "creative_dna_application",
    "memory_retrieval_execution",
    "memory_storage_write",
    "automatic_memory_consolidation",
    "automatic_preference_mutation",
    "automatic_personalization_application",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "agent_invocation",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class CreativeOntologyConcept(BaseModel):
    """One advisory creative ontology concept record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    creative_ontology_id: str = Field(min_length=1, max_length=180)
    concept_kind: CreativeOntologyConceptKind
    status: CreativeOntologyStatus
    confidence: CreativeOntologyConfidence
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    ontology_axis: CreativeOntologyAxis
    source_creative_lineage_record_id: str = Field(min_length=1, max_length=180)
    source_creative_dna_signature_id: str = Field(min_length=1, max_length=180)
    source_long_term_memory_record_id: str = Field(min_length=1, max_length=180)
    source_domain_category: DomainCategory
    source_domain_category_label: str = Field(min_length=1, max_length=120)
    source_domain_values: tuple[CreativeCodingDomain, ...] = Field(
        min_length=1,
        max_length=12,
    )
    ontology_statement: str = Field(min_length=1, max_length=360)
    concept_coverage_score: int = Field(ge=0, le=100)
    taxonomy_alignment_score: int = Field(ge=0, le=100)
    lineage_alignment_score: int = Field(ge=0, le=100)
    governance_risk_score: int = Field(ge=0, le=100)
    governance_weight: int = Field(ge=0, le=240)
    creative_ontology_score: int = Field(ge=0, le=1_000)
    hitl_required_before_ontology_persistence: bool
    context_tags: tuple[str, ...] = Field(min_length=1, max_length=8)
    explainability_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=34,
    )
    creative_ontology_implemented: Literal[True] = True
    creative_ontology_metadata_implemented: Literal[True] = True
    creative_lineage_source_used: Literal[True] = True
    creative_dna_source_used: Literal[True] = True
    long_term_memory_source_used: Literal[True] = True
    domain_metadata_source_used: Literal[True] = True
    ontology_storage_write_implemented: Literal[False] = False
    ontology_node_creation_implemented: Literal[False] = False
    ontology_edge_creation_implemented: Literal[False] = False
    ontology_record_update_implemented: Literal[False] = False
    ontology_record_deletion_implemented: Literal[False] = False
    ontology_relationship_inference_implemented: Literal[False] = False
    taxonomy_mutation_implemented: Literal[False] = False
    semantic_graph_materialization_implemented: Literal[False] = False
    domain_registry_mutation_implemented: Literal[False] = False
    creative_lineage_application_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    memory_consolidation_implemented: Literal[False] = False
    preference_mutation_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["creative_ontology_concept.v1"] = (
        CREATIVE_ONTOLOGY_CONCEPT_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _concept_matches_contract(self) -> Self:
        if self.creative_ontology_id != f"creative_ontology::{self.concept_kind}":
            raise ValueError("creative_ontology_id must match concept_kind")
        if self.source_domain_category_label != get_domain_category_label(
            self.source_domain_category
        ):
            raise ValueError("source_domain_category_label must match category")
        if self.source_domain_values != get_domains_for_category(
            self.source_domain_category
        ):
            raise ValueError("source_domain_values must match category")
        if self.creative_ontology_score != _creative_ontology_score(
            concept_coverage_score=self.concept_coverage_score,
            taxonomy_alignment_score=self.taxonomy_alignment_score,
            lineage_alignment_score=self.lineage_alignment_score,
            governance_risk_score=self.governance_risk_score,
            governance_weight=self.governance_weight,
        ):
            raise ValueError("creative_ontology_score must combine source scores")
        if self.status != _creative_ontology_status(self.creative_ontology_score):
            raise ValueError("status must match creative_ontology_score")
        if self.confidence != _creative_ontology_confidence(
            self.creative_ontology_score
        ):
            raise ValueError("confidence must match creative_ontology_score")
        if not self.hitl_required_before_ontology_persistence:
            raise ValueError("creative ontology persistence requires HITL posture")
        return self


class CreativeOntologyPlan(BaseModel):
    """Bounded V6.2 advisory creative ontology plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_ontology"] = "creative_ontology"
    serialization_version: Literal["creative_ontology_plan.v1"] = (
        CREATIVE_ONTOLOGY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_ONTOLOGY_AUTHORITY_BOUNDARY,
        max_length=2100,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_creative_lineage_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_creative_dna_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_long_term_memory_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_domain_metadata_registry_ref: str = Field(min_length=1, max_length=160)
    source_creative_lineage_record_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_creative_dna_signature_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_long_term_memory_record_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_domain_category_ids: tuple[DomainCategory, ...] = Field(
        min_length=12,
        max_length=12,
    )
    source_domain_count: int = Field(ge=1, le=100)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    concepts: tuple[CreativeOntologyConcept, ...] = Field(min_length=5, max_length=5)
    concept_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    candidate_concept_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    review_required_concept_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    guarded_concept_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    high_confidence_concept_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    hitl_required_concept_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    persisted_ontology_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    inferred_ontology_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    materialized_semantic_graph_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    mutated_domain_registry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    concept_count: int = Field(ge=5, le=5)
    candidate_concept_count: int = Field(ge=0, le=5)
    review_required_concept_count: int = Field(ge=0, le=5)
    guarded_concept_count: int = Field(ge=0, le=5)
    high_confidence_concept_count: int = Field(ge=0, le=5)
    hitl_required_concept_count: int = Field(ge=0, le=5)
    highest_creative_ontology_score: int = Field(ge=0, le=1_000)
    overall_creative_ontology_score: int = Field(ge=0, le=1_000)
    overall_creative_ontology_posture: CreativeOntologyPosture
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=34,
    )
    creative_ontology_implemented: Literal[True] = True
    creative_ontology_metadata_implemented: Literal[True] = True
    creative_lineage_source_used: Literal[True] = True
    creative_dna_source_used: Literal[True] = True
    long_term_memory_source_used: Literal[True] = True
    domain_metadata_source_used: Literal[True] = True
    ontology_storage_write_implemented: Literal[False] = False
    ontology_node_creation_implemented: Literal[False] = False
    ontology_edge_creation_implemented: Literal[False] = False
    ontology_record_update_implemented: Literal[False] = False
    ontology_record_deletion_implemented: Literal[False] = False
    ontology_relationship_inference_implemented: Literal[False] = False
    taxonomy_mutation_implemented: Literal[False] = False
    semantic_graph_materialization_implemented: Literal[False] = False
    domain_registry_mutation_implemented: Literal[False] = False
    creative_lineage_application_implemented: Literal[False] = False
    creative_dna_application_implemented: Literal[False] = False
    memory_retrieval_execution_implemented: Literal[False] = False
    memory_storage_write_implemented: Literal[False] = False
    memory_consolidation_implemented: Literal[False] = False
    preference_mutation_implemented: Literal[False] = False
    personalization_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_concepts(self) -> Self:
        derived_concept_ids = tuple(
            concept.creative_ontology_id for concept in self.concepts
        )
        if len(set(derived_concept_ids)) != len(derived_concept_ids):
            raise ValueError("concept_ids must be unique")
        if self.concept_ids != derived_concept_ids:
            raise ValueError("concept_ids must match concepts")
        if self.candidate_concept_ids != _concept_ids_for_status(
            self.concepts,
            "candidate",
        ):
            raise ValueError("candidate_concept_ids must match concepts")
        if self.review_required_concept_ids != _concept_ids_for_status(
            self.concepts,
            "review_required",
        ):
            raise ValueError("review_required_concept_ids must match concepts")
        if self.guarded_concept_ids != _concept_ids_for_status(
            self.concepts,
            "guarded",
        ):
            raise ValueError("guarded_concept_ids must match concepts")
        if self.high_confidence_concept_ids != _concept_ids_for_confidence(
            self.concepts,
            "high",
            "guarded",
        ):
            raise ValueError("high_confidence_concept_ids must match concepts")
        if self.hitl_required_concept_ids != tuple(
            concept.creative_ontology_id
            for concept in self.concepts
            if concept.hitl_required_before_ontology_persistence
        ):
            raise ValueError("hitl_required_concept_ids must match concepts")
        if self.persisted_ontology_ids:
            raise ValueError("persisted_ontology_ids must remain empty")
        if self.inferred_ontology_ids:
            raise ValueError("inferred_ontology_ids must remain empty")
        if self.materialized_semantic_graph_ids:
            raise ValueError("materialized_semantic_graph_ids must remain empty")
        if self.mutated_domain_registry_ids:
            raise ValueError("mutated_domain_registry_ids must remain empty")
        if self.source_domain_category_ids != get_domain_categories():
            raise ValueError("source_domain_category_ids must match domain registry")
        if self.source_domain_count != _domain_count(self.source_domain_category_ids):
            raise ValueError("source_domain_count must match domain registry")
        if self.concept_count != len(self.concepts):
            raise ValueError("concept_count must match concepts")
        if self.candidate_concept_count != len(self.candidate_concept_ids):
            raise ValueError("candidate_concept_count must match concepts")
        if self.review_required_concept_count != len(self.review_required_concept_ids):
            raise ValueError("review_required_concept_count must match concepts")
        if self.guarded_concept_count != len(self.guarded_concept_ids):
            raise ValueError("guarded_concept_count must match concepts")
        if self.high_confidence_concept_count != len(self.high_confidence_concept_ids):
            raise ValueError("high_confidence_concept_count must match concepts")
        if self.hitl_required_concept_count != len(self.hitl_required_concept_ids):
            raise ValueError("hitl_required_concept_count must match concepts")
        if self.highest_creative_ontology_score != max(
            concept.creative_ontology_score for concept in self.concepts
        ):
            raise ValueError("highest_creative_ontology_score must match concepts")
        if self.overall_creative_ontology_score != _overall_creative_ontology_score(
            self.concepts
        ):
            raise ValueError("overall_creative_ontology_score must match concepts")
        if self.overall_creative_ontology_posture != _overall_creative_ontology_posture(
            self.concepts
        ):
            raise ValueError("overall_creative_ontology_posture must match concepts")
        for concept in self.concepts:
            if concept.route_name != self.route_name:
                raise ValueError("concept route_name must match plan")
            if (
                concept.source_creative_lineage_record_id
                not in self.source_creative_lineage_record_ids
            ):
                raise ValueError("source creative lineage record must be declared")
            if (
                concept.source_creative_dna_signature_id
                not in self.source_creative_dna_signature_ids
            ):
                raise ValueError("source Creative DNA signature must be declared")
            if (
                concept.source_long_term_memory_record_id
                not in self.source_long_term_memory_record_ids
            ):
                raise ValueError("source long-term memory record must be declared")
            if concept.source_domain_category not in self.source_domain_category_ids:
                raise ValueError("source domain category must be declared")
        return self


def build_creative_ontology(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    creative_lineage: CreativeLineagePlan | None = None,
    creative_dna: CreativeDNAPlan | None = None,
    long_term_memory: LongTermCreativeMemoryPlan | None = None,
) -> CreativeOntologyPlan:
    """Build creative ontology metadata without inference or graph materialization."""

    route_name = _resolve_route(route)
    normalized_task_type = _resolve_task_type(task_type)
    execution_modes = routing_execution_mode_registry()
    normalized_mode = _resolve_execution_mode(
        execution_mode_id or execution_modes.execution_mode_ids[0],
        execution_modes.execution_mode_ids,
    )
    memory_plan = long_term_memory or build_long_term_creative_memory(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
    )
    dna_plan = creative_dna or build_creative_dna(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        long_term_memory=memory_plan,
    )
    lineage_plan = creative_lineage or build_creative_lineage(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        creative_dna=dna_plan,
        long_term_memory=memory_plan,
    )
    domain_categories = get_domain_categories()
    concepts = _concepts(
        route_name=route_name,
        task_type=normalized_task_type,
        execution_mode_id=normalized_mode,
        creative_lineage=lineage_plan,
        creative_dna=dna_plan,
        long_term_memory=memory_plan,
    )
    return CreativeOntologyPlan(
        route_name=route_name,
        task_type=normalized_task_type,
        source_creative_lineage_serialization_version=(
            lineage_plan.serialization_version
        ),
        source_creative_dna_serialization_version=dna_plan.serialization_version,
        source_long_term_memory_serialization_version=memory_plan.serialization_version,
        source_domain_metadata_registry_ref=DOMAIN_METADATA_REGISTRY_REF,
        source_creative_lineage_record_ids=lineage_plan.record_ids,
        source_creative_dna_signature_ids=dna_plan.signature_ids,
        source_long_term_memory_record_ids=memory_plan.record_ids,
        source_domain_category_ids=domain_categories,
        source_domain_count=_domain_count(domain_categories),
        execution_mode_ids=execution_modes.execution_mode_ids,
        concepts=concepts,
        concept_ids=tuple(concept.creative_ontology_id for concept in concepts),
        candidate_concept_ids=_concept_ids_for_status(concepts, "candidate"),
        review_required_concept_ids=_concept_ids_for_status(
            concepts,
            "review_required",
        ),
        guarded_concept_ids=_concept_ids_for_status(concepts, "guarded"),
        high_confidence_concept_ids=_concept_ids_for_confidence(
            concepts,
            "high",
            "guarded",
        ),
        hitl_required_concept_ids=tuple(
            concept.creative_ontology_id
            for concept in concepts
            if concept.hitl_required_before_ontology_persistence
        ),
        persisted_ontology_ids=(),
        inferred_ontology_ids=(),
        materialized_semantic_graph_ids=(),
        mutated_domain_registry_ids=(),
        concept_count=len(concepts),
        candidate_concept_count=len(_concept_ids_for_status(concepts, "candidate")),
        review_required_concept_count=len(
            _concept_ids_for_status(concepts, "review_required")
        ),
        guarded_concept_count=len(_concept_ids_for_status(concepts, "guarded")),
        high_confidence_concept_count=len(
            _concept_ids_for_confidence(concepts, "high", "guarded")
        ),
        hitl_required_concept_count=sum(
            1
            for concept in concepts
            if concept.hitl_required_before_ontology_persistence
        ),
        highest_creative_ontology_score=max(
            concept.creative_ontology_score for concept in concepts
        ),
        overall_creative_ontology_score=_overall_creative_ontology_score(concepts),
        overall_creative_ontology_posture=_overall_creative_ontology_posture(concepts),
        advisory_actions=_plan_actions(concepts),
    )


def creative_ontology_concept_by_id(
    concept_id: str,
    plan: CreativeOntologyPlan | None = None,
) -> CreativeOntologyConcept | None:
    """Return one creative ontology concept without inferring relationships."""

    source_plan = plan or build_creative_ontology()
    for concept in source_plan.concepts:
        if concept.creative_ontology_id == concept_id:
            return concept
    return None


def creative_ontology_concepts_for_status(
    status: CreativeOntologyStatus,
    plan: CreativeOntologyPlan | None = None,
) -> tuple[CreativeOntologyConcept, ...]:
    """Return creative ontology concepts by advisory status."""

    source_plan = plan or build_creative_ontology()
    return tuple(
        concept for concept in source_plan.concepts if concept.status == status
    )


def creative_ontology_concepts_for_confidence(
    confidence: CreativeOntologyConfidence,
    plan: CreativeOntologyPlan | None = None,
) -> tuple[CreativeOntologyConcept, ...]:
    """Return creative ontology concepts by confidence band."""

    source_plan = plan or build_creative_ontology()
    return tuple(
        concept for concept in source_plan.concepts if concept.confidence == confidence
    )


def _concepts(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    creative_lineage: CreativeLineagePlan,
    creative_dna: CreativeDNAPlan,
    long_term_memory: LongTermCreativeMemoryPlan,
) -> tuple[CreativeOntologyConcept, ...]:
    return (
        _concept(
            kind="creative_intent_ontology",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            ontology_axis="intent",
            source_lineage_record_id="creative_lineage::source_transition_lineage",
            source_dna_signature_id="creative_dna::intent_dna",
            source_memory_record_id="long_term_creative_memory::creative_intent_memory",
            source_domain_category=DomainCategory.WEB_CREATIVE_CODING,
            creative_lineage=creative_lineage,
            creative_dna=creative_dna,
            long_term_memory=long_term_memory,
            concept_coverage_score=86,
            taxonomy_alignment_score=84,
            lineage_alignment_score=82,
            governance_risk_score=46,
            governance_weight=150,
        ),
        _concept(
            kind="style_system_ontology",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            ontology_axis="style",
            source_lineage_record_id="creative_lineage::style_evolution_lineage",
            source_dna_signature_id="creative_dna::style_dna",
            source_memory_record_id="long_term_creative_memory::style_pattern_memory",
            source_domain_category=DomainCategory.SHADERS_GPU,
            creative_lineage=creative_lineage,
            creative_dna=creative_dna,
            long_term_memory=long_term_memory,
            concept_coverage_score=82,
            taxonomy_alignment_score=78,
            lineage_alignment_score=80,
            governance_risk_score=44,
            governance_weight=140,
        ),
        _concept(
            kind="interaction_modality_ontology",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            ontology_axis="interaction",
            source_lineage_record_id="creative_lineage::timeline_stage_lineage",
            source_dna_signature_id="creative_dna::interaction_dna",
            source_memory_record_id=(
                "long_term_creative_memory::preference_signal_memory"
            ),
            source_domain_category=DomainCategory.AUDIO_LIVE_CODING,
            creative_lineage=creative_lineage,
            creative_dna=creative_dna,
            long_term_memory=long_term_memory,
            concept_coverage_score=74,
            taxonomy_alignment_score=76,
            lineage_alignment_score=72,
            governance_risk_score=42,
            governance_weight=120,
        ),
        _concept(
            kind="artifact_lineage_ontology",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            ontology_axis="lineage",
            source_lineage_record_id="creative_lineage::artifact_dependency_lineage",
            source_dna_signature_id="creative_dna::lineage_dna",
            source_memory_record_id="long_term_creative_memory::artifact_lineage_memory",
            source_domain_category=DomainCategory.DCC_PROCEDURAL,
            creative_lineage=creative_lineage,
            creative_dna=creative_dna,
            long_term_memory=long_term_memory,
            concept_coverage_score=64,
            taxonomy_alignment_score=60,
            lineage_alignment_score=66,
            governance_risk_score=32,
            governance_weight=110,
        ),
        _concept(
            kind="domain_capability_ontology",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            ontology_axis="domain_capability",
            source_lineage_record_id="creative_lineage::gap_review_lineage",
            source_dna_signature_id="creative_dna::constraint_dna",
            source_memory_record_id="long_term_creative_memory::project_context_memory",
            source_domain_category=DomainCategory.CREATIVE_AI,
            creative_lineage=creative_lineage,
            creative_dna=creative_dna,
            long_term_memory=long_term_memory,
            concept_coverage_score=50,
            taxonomy_alignment_score=52,
            lineage_alignment_score=54,
            governance_risk_score=18,
            governance_weight=85,
        ),
    )


def _concept(
    *,
    kind: CreativeOntologyConceptKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    ontology_axis: CreativeOntologyAxis,
    source_lineage_record_id: str,
    source_dna_signature_id: str,
    source_memory_record_id: str,
    source_domain_category: DomainCategory,
    creative_lineage: CreativeLineagePlan,
    creative_dna: CreativeDNAPlan,
    long_term_memory: LongTermCreativeMemoryPlan,
    concept_coverage_score: int,
    taxonomy_alignment_score: int,
    lineage_alignment_score: int,
    governance_risk_score: int,
    governance_weight: int,
) -> CreativeOntologyConcept:
    source_lineage_record = creative_lineage_record_by_id(
        source_lineage_record_id,
        creative_lineage,
    )
    source_dna_signature = creative_dna_signature_by_id(
        source_dna_signature_id,
        creative_dna,
    )
    source_memory_record = long_term_creative_memory_record_by_id(
        source_memory_record_id,
        long_term_memory,
    )
    if source_lineage_record is None:
        raise ValueError("source creative lineage record must exist")
    if source_dna_signature is None:
        raise ValueError("source Creative DNA signature must exist")
    if source_memory_record is None:
        raise ValueError("source long-term memory record must exist")
    score = _creative_ontology_score(
        concept_coverage_score=concept_coverage_score,
        taxonomy_alignment_score=taxonomy_alignment_score,
        lineage_alignment_score=lineage_alignment_score,
        governance_risk_score=governance_risk_score,
        governance_weight=governance_weight,
    )
    status = _creative_ontology_status(score)
    confidence = _creative_ontology_confidence(score)
    domain_values = get_domains_for_category(source_domain_category)
    category_label = get_domain_category_label(source_domain_category)
    return CreativeOntologyConcept(
        creative_ontology_id=f"creative_ontology::{kind}",
        concept_kind=kind,
        status=status,
        confidence=confidence,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        ontology_axis=ontology_axis,
        source_creative_lineage_record_id=(source_lineage_record.creative_lineage_id),
        source_creative_dna_signature_id=source_dna_signature.creative_dna_id,
        source_long_term_memory_record_id=source_memory_record.record_id,
        source_domain_category=source_domain_category,
        source_domain_category_label=category_label,
        source_domain_values=domain_values,
        ontology_statement=_ontology_statement(kind, category_label),
        concept_coverage_score=concept_coverage_score,
        taxonomy_alignment_score=taxonomy_alignment_score,
        lineage_alignment_score=lineage_alignment_score,
        governance_risk_score=governance_risk_score,
        governance_weight=governance_weight,
        creative_ontology_score=score,
        hitl_required_before_ontology_persistence=True,
        context_tags=_context_tags(kind, ontology_axis),
        explainability_notes=_explainability_notes(
            kind,
            source_lineage_record.creative_lineage_id,
            source_dna_signature.creative_dna_id,
            source_memory_record.record_id,
            source_domain_category,
        ),
        advisory_actions=_concept_actions(kind),
        evidence=(
            f"source_creative_lineage:{source_lineage_record.creative_lineage_id}",
            f"source_creative_dna:{source_dna_signature.creative_dna_id}",
            f"source_long_term_memory:{source_memory_record.record_id}",
            f"source_domain_category:{source_domain_category.value}",
            f"source_domain_count:{len(domain_values)}",
            f"ontology_axis:{ontology_axis}",
            f"concept_coverage_score:{concept_coverage_score}",
            f"taxonomy_alignment_score:{taxonomy_alignment_score}",
            f"lineage_alignment_score:{lineage_alignment_score}",
            f"governance_risk_score:{governance_risk_score}",
            "hitl_required_before_ontology_persistence:true",
        ),
    )


def _creative_ontology_score(
    *,
    concept_coverage_score: int,
    taxonomy_alignment_score: int,
    lineage_alignment_score: int,
    governance_risk_score: int,
    governance_weight: int,
) -> int:
    return min(
        1_000,
        max(
            0,
            concept_coverage_score * 3
            + taxonomy_alignment_score * 3
            + lineage_alignment_score * 2
            + governance_risk_score * 2
            + governance_weight,
        ),
    )


def _creative_ontology_status(score: int) -> CreativeOntologyStatus:
    if score >= 860:
        return "guarded"
    if score >= 650:
        return "review_required"
    return "candidate"


def _creative_ontology_confidence(score: int) -> CreativeOntologyConfidence:
    if score >= 860:
        return "guarded"
    if score >= 760:
        return "high"
    if score >= 520:
        return "medium"
    return "low"


def _overall_creative_ontology_score(
    concepts: tuple[CreativeOntologyConcept, ...],
) -> int:
    base = sum(concept.creative_ontology_score for concept in concepts) // len(concepts)
    guarded_count = len(_concept_ids_for_status(concepts, "guarded"))
    review_count = len(_concept_ids_for_status(concepts, "review_required"))
    return min(1_000, base + guarded_count * 20 + review_count * 10)


def _overall_creative_ontology_posture(
    concepts: tuple[CreativeOntologyConcept, ...],
) -> CreativeOntologyPosture:
    if any(concept.status == "guarded" for concept in concepts):
        return "guarded"
    if any(concept.status == "review_required" for concept in concepts):
        return "review_required"
    return "candidate"


def _concept_ids_for_status(
    concepts: tuple[CreativeOntologyConcept, ...],
    status: CreativeOntologyStatus,
) -> tuple[str, ...]:
    return tuple(
        concept.creative_ontology_id for concept in concepts if concept.status == status
    )


def _concept_ids_for_confidence(
    concepts: tuple[CreativeOntologyConcept, ...],
    *confidences: CreativeOntologyConfidence,
) -> tuple[str, ...]:
    return tuple(
        concept.creative_ontology_id
        for concept in concepts
        if concept.confidence in confidences
    )


def _domain_count(categories: tuple[DomainCategory, ...]) -> int:
    return sum(len(get_domains_for_category(category)) for category in categories)


def _plan_actions(concepts: tuple[CreativeOntologyConcept, ...]) -> tuple[str, ...]:
    guarded_concept_count = len(_concept_ids_for_status(concepts, "guarded"))
    return (
        "inspect_creative_ontology_concepts",
        "require_hitl_before_creative_ontology_persistence",
        "keep_creative_ontology_non_inferential",
        f"review_guarded_creative_ontology_count:{guarded_concept_count}",
    )


def _ontology_statement(
    kind: CreativeOntologyConceptKind,
    category_label: str,
) -> str:
    statements = {
        "creative_intent_ontology": (
            "Models advisory creative intent concepts for domain-scoped memory."
        ),
        "style_system_ontology": (
            "Models advisory style-system concepts for shader and visual language."
        ),
        "interaction_modality_ontology": (
            "Models advisory interaction modality concepts for live creative systems."
        ),
        "artifact_lineage_ontology": (
            "Models advisory artifact lineage concepts for procedural workflows."
        ),
        "domain_capability_ontology": (
            "Models advisory domain capability concepts for creative AI boundaries."
        ),
    }
    return f"{statements[kind]} Source category: {category_label}."


def _context_tags(
    kind: CreativeOntologyConceptKind,
    ontology_axis: CreativeOntologyAxis,
) -> tuple[str, ...]:
    return (
        "creative_memory",
        "creative_ontology",
        ontology_axis,
        kind.removesuffix("_ontology"),
    )


def _explainability_notes(
    kind: CreativeOntologyConceptKind,
    source_lineage_record_id: str,
    source_dna_signature_id: str,
    source_memory_record_id: str,
    source_domain_category: DomainCategory,
) -> tuple[str, ...]:
    return (
        f"creative_ontology_kind:{kind}",
        f"source_creative_lineage:{source_lineage_record_id}",
        f"source_creative_dna:{source_dna_signature_id}",
        f"source_record:{source_memory_record_id}",
        f"source_domain_category:{source_domain_category.value}",
        "score_inputs:concept_coverage,taxonomy_alignment,lineage_alignment,governance_risk,governance",
        "persistence_boundary:HITL_required_before_creative_ontology_persistence",
    )


def _concept_actions(kind: CreativeOntologyConceptKind) -> tuple[str, ...]:
    return (
        f"review_{kind}",
        "inspect_sources_before_creative_ontology_persistence",
        "preserve_no_ontology_inference_boundary",
    )


def _resolve_route(route: RouteName | str) -> RouteName:
    return route if isinstance(route, RouteName) else RouteName(str(route).strip())


def _resolve_task_type(task_type: TaskRoutingType | str) -> TaskRoutingType:
    normalized = str(task_type).strip()
    if normalized not in get_args(TaskRoutingType):
        raise ValueError("task_type must be a known routing task type")
    return cast(TaskRoutingType, normalized)


def _resolve_execution_mode(
    execution_mode_id: ExecutionModeId | str,
    allowed_execution_mode_ids: tuple[ExecutionModeId, ...],
) -> ExecutionModeId:
    normalized = str(execution_mode_id).strip()
    if normalized not in allowed_execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")
    return cast(ExecutionModeId, normalized)
