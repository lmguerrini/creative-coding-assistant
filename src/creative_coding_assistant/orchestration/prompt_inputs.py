"""Prompt-input contracts and transformation boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, Self

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.memory import ProjectMemoryKind
from creative_coding_assistant.orchestration.artifact_capability_matrix import (
    ArtifactCapabilityMatrix,
)
from creative_coding_assistant.orchestration.artifact_critic import (
    ArtifactCriticProfile,
)
from creative_coding_assistant.orchestration.artifact_dependency_graph import (
    ArtifactDependencyGraph,
)
from creative_coding_assistant.orchestration.artifact_engine_contracts import (
    ArtifactIntelligenceEngineContractRegistry,
)
from creative_coding_assistant.orchestration.artifact_export_intelligence import (
    ArtifactExportIntelligenceProfile,
)
from creative_coding_assistant.orchestration.artifact_intelligence_synthesis import (
    ArtifactIntelligenceSynthesisProfile,
)
from creative_coding_assistant.orchestration.artifact_merge_planner import (
    ArtifactMergePlannerProfile,
)
from creative_coding_assistant.orchestration.artifact_planner import ArtifactPlan
from creative_coding_assistant.orchestration.artifact_refiner import (
    ArtifactRefinerProfile,
)
from creative_coding_assistant.orchestration.audio_visual_scene import (
    AudioVisualSceneProfile,
)
from creative_coding_assistant.orchestration.clarification import (
    ClarificationRequest,
    derive_hitl_clarification,
)
from creative_coding_assistant.orchestration.context import AssembledContextResponse
from creative_coding_assistant.orchestration.creative_composition import (
    CreativeCompositionPlan,
)
from creative_coding_assistant.orchestration.creative_confidence_engine import (
    CreativeConfidenceProfile,
)
from creative_coding_assistant.orchestration.creative_constraint_priorities import (
    CreativeConstraintPrioritization,
)
from creative_coding_assistant.orchestration.creative_constraints import (
    CreativeConstraintSolution,
)
from creative_coding_assistant.orchestration.creative_critic_engine import (
    CreativeCriticProfile,
)
from creative_coding_assistant.orchestration.creative_director import (
    CreativeAssistantDirectorBrief,
)
from creative_coding_assistant.orchestration.creative_hierarchy import (
    CreativeHierarchyPlan,
)
from creative_coding_assistant.orchestration.creative_improvement_planner import (
    CreativeImprovementPlannerProfile,
)
from creative_coding_assistant.orchestration.creative_intent import (
    CreativeIntentDecomposition,
)
from creative_coding_assistant.orchestration.creative_planning import (
    CreativeExecutionPlan,
)
from creative_coding_assistant.orchestration.creative_quality_prediction import (
    CreativeQualityPrediction,
)
from creative_coding_assistant.orchestration.creative_reasoning import (
    CreativeReasoningResult,
)
from creative_coding_assistant.orchestration.creative_score_engine import (
    CreativeScoreProfile,
)
from creative_coding_assistant.orchestration.creative_strategy import (
    CreativeStrategyProfile,
)
from creative_coding_assistant.orchestration.creative_technique import (
    CreativeTechniqueProfile,
)
from creative_coding_assistant.orchestration.creative_tradeoffs import (
    CreativeTradeoffProfile,
)
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeTranslation,
    derive_creative_translation,
)
from creative_coding_assistant.orchestration.cross_modality import (
    CrossModalityCompositionProfile,
)
from creative_coding_assistant.orchestration.emotional_consistency import (
    EmotionalConsistencyProfile,
)
from creative_coding_assistant.orchestration.generative_structure import (
    GenerativeStructureBlueprint,
)
from creative_coding_assistant.orchestration.memory import MemoryContextResponse
from creative_coding_assistant.orchestration.multi_artifact_strategy import (
    MultiArtifactStrategy,
)
from creative_coding_assistant.orchestration.procedural_structure import (
    ProceduralStructurePlan,
)
from creative_coding_assistant.orchestration.prompt_memory import (
    PromptConversationTurnInput,
    PromptSessionMemorySummaryInput,
    build_prompt_recent_turns,
    build_session_memory_summaries,
    looks_like_follow_up_query,
)
from creative_coding_assistant.orchestration.retrieval import RetrievalContextResponse
from creative_coding_assistant.orchestration.reflection_loop_engine import (
    ReflectionLoopProfile,
)
from creative_coding_assistant.orchestration.routing import (
    DomainSelectionShape,
    RouteDecision,
    RouteName,
)
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.runtime_compatibility import (
    RuntimeCompatibilityProfile,
)
from creative_coding_assistant.orchestration.semantic_motif import SemanticMotifSystem
from creative_coding_assistant.orchestration.self_evaluation_engine import (
    SelfEvaluationProfile,
)
from creative_coding_assistant.orchestration.symbolic_narrative import (
    SymbolicNarrativePlan,
)
from creative_coding_assistant.rag.retrieval.domain_intent import (
    detect_explicit_query_domains,
)
from creative_coding_assistant.rag.sources import OfficialSourceType


class PromptImageReferenceInput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)


class PromptArtifactRefinementInput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    artifact_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    language: str = Field(min_length=1)
    content: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    domain: CreativeCodingDomain | None = None
    runtime: str | None = None
    renderer_id: str | None = None
    preview_eligible: bool | None = None
    quality_score: float | None = Field(default=None, ge=0, le=1)
    quality_rank: int | None = Field(default=None, ge=1)
    quality_before: float | None = Field(default=None, ge=0, le=1)
    pass_number: int | None = Field(default=None, ge=1)
    max_passes: int | None = Field(default=None, ge=1, le=3)
    refinement_objective: str | None = None
    refinement_passes: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    critique_rationale: str | None = None
    refinement_guidance: str | None = None
    creative_translation: CreativeTranslation | None = None
    creative_plan: CreativeExecutionPlan | None = None


class PromptUserInput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    query: str = Field(min_length=1)
    mode: AssistantMode
    domain: CreativeCodingDomain | None = None
    domains: tuple[CreativeCodingDomain, ...] = Field(default_factory=tuple)
    ui_selected_domains: tuple[CreativeCodingDomain, ...] = Field(
        default_factory=tuple
    )
    detected_domains: tuple[CreativeCodingDomain, ...] = Field(default_factory=tuple)
    effective_domains: tuple[CreativeCodingDomain, ...] = Field(default_factory=tuple)
    domain_selection: DomainSelectionShape = DomainSelectionShape.NONE
    is_follow_up: bool = False
    image_references: tuple[PromptImageReferenceInput, ...] = Field(
        default_factory=tuple
    )
    artifact_refinement: PromptArtifactRefinementInput | None = None
    clarification_response: str | None = None

    @field_validator(
        "domains",
        "ui_selected_domains",
        "detected_domains",
        "effective_domains",
        mode="before",
    )
    @classmethod
    def normalize_domains(
        cls,
        value: Sequence[CreativeCodingDomain | str] | CreativeCodingDomain | str | None,
    ) -> tuple[CreativeCodingDomain, ...]:
        if value is None:
            return ()
        if isinstance(value, CreativeCodingDomain):
            return (value,)
        if isinstance(value, str):
            return (CreativeCodingDomain(value.strip()),)

        normalized: list[CreativeCodingDomain] = []
        for item in value:
            domain = (
                item
                if isinstance(item, CreativeCodingDomain)
                else CreativeCodingDomain(str(item).strip())
            )
            if domain not in normalized:
                normalized.append(domain)
        return tuple(normalized)

    @model_validator(mode="before")
    @classmethod
    def populate_legacy_domain_fields(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        domain = normalized.get("domain")
        domains = normalized.get("domains")
        ui_selected_domains = normalized.get("ui_selected_domains")
        effective_domains = normalized.get("effective_domains")

        if domains and not ui_selected_domains:
            normalized["ui_selected_domains"] = domains

        if domains and not effective_domains:
            normalized["effective_domains"] = domains

        if effective_domains and not domains:
            normalized["domains"] = effective_domains

        if domain is not None and not normalized.get("domains"):
            normalized["domains"] = (domain,)
        if domain is not None and not normalized.get("ui_selected_domains"):
            normalized["ui_selected_domains"] = (domain,)
        if domain is not None and not normalized.get("effective_domains"):
            normalized["effective_domains"] = (domain,)

        return normalized

    @model_validator(mode="after")
    def validate_domain_alignment(self) -> PromptUserInput:
        if not self.domains and self.effective_domains:
            object.__setattr__(self, "domains", self.effective_domains)

        if not self.effective_domains and self.domains:
            object.__setattr__(self, "effective_domains", self.domains)

        if not self.ui_selected_domains and self.domains:
            object.__setattr__(self, "ui_selected_domains", self.domains)

        if self.domain is None and len(self.effective_domains) == 1:
            object.__setattr__(self, "domain", self.effective_domains[0])

        if self.domain is not None and self.domain not in self.effective_domains:
            raise ValueError(
                "Prompt user input domain must be included in domains "
                "when both are provided."
            )

        object.__setattr__(
            self,
            "domain_selection",
            _selection_shape_for_domains(self.effective_domains),
        )
        return self


class PromptRunningSummaryInput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    content: str = Field(min_length=1)
    covered_turn_count: int = Field(ge=1)


class PromptProjectMemoryInput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    memory_kind: ProjectMemoryKind
    content: str = Field(min_length=1)
    source: str = Field(min_length=1)


class PromptMemoryInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    recent_turns: tuple[PromptConversationTurnInput, ...] = Field(default_factory=tuple)
    running_summary: PromptRunningSummaryInput | None = None
    session_summaries: tuple[PromptSessionMemorySummaryInput, ...] = Field(
        default_factory=tuple
    )
    project_memories: tuple[PromptProjectMemoryInput, ...] = Field(
        default_factory=tuple
    )


class PromptKnowledgeChunkInput(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_id: str = Field(min_length=1)
    domain: CreativeCodingDomain
    source_type: OfficialSourceType
    publisher: str = Field(min_length=1)
    registry_title: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)


class PromptRetrievalInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunks: tuple[PromptKnowledgeChunkInput, ...] = Field(default_factory=tuple)


class PromptInputRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: RouteName
    route_decision: RouteDecision | None = None
    assistant_request: AssistantRequest
    assembled_context: AssembledContextResponse | None = None

    @model_validator(mode="after")
    def validate_route_alignment(self) -> Self:
        if self.route_decision is not None and self.route_decision.route != self.route:
            raise ValueError(
                "Prompt-input route decision must match the prompt-input route."
            )
        if (
            self.assembled_context is not None
            and self.assembled_context.request.route != self.route
        ):
            raise ValueError(
                "Assembled context route must match the prompt-input route."
            )
        return self


class PromptInputResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: PromptInputRequest
    user_input: PromptUserInput
    creative_translation: CreativeTranslation | None = None
    creative_intent: CreativeIntentDecomposition | None = None
    creative_hierarchy: CreativeHierarchyPlan | None = None
    creative_strategy: CreativeStrategyProfile | None = None
    creative_techniques: CreativeTechniqueProfile | None = None
    creative_plan: CreativeExecutionPlan | None = None
    creative_constraints: CreativeConstraintSolution | None = None
    creative_constraint_priorities: CreativeConstraintPrioritization | None = None
    runtime_capabilities: RuntimeCapabilityProfile | None = None
    creative_tradeoffs: CreativeTradeoffProfile | None = None
    creative_quality_prediction: CreativeQualityPrediction | None = None
    symbolic_narrative: SymbolicNarrativePlan | None = None
    creative_composition: CreativeCompositionPlan | None = None
    procedural_structure: ProceduralStructurePlan | None = None
    generative_structure: GenerativeStructureBlueprint | None = None
    semantic_motif: SemanticMotifSystem | None = None
    emotional_consistency: EmotionalConsistencyProfile | None = None
    cross_modality: CrossModalityCompositionProfile | None = None
    audio_visual_scene: AudioVisualSceneProfile | None = None
    artifact_plan: ArtifactPlan | None = None
    artifact_dependency_graph: ArtifactDependencyGraph | None = None
    runtime_compatibility: RuntimeCompatibilityProfile | None = None
    artifact_capability_matrix: ArtifactCapabilityMatrix | None = None
    multi_artifact_strategy: MultiArtifactStrategy | None = None
    artifact_critic: ArtifactCriticProfile | None = None
    artifact_refiner: ArtifactRefinerProfile | None = None
    artifact_intelligence_synthesis: (
        ArtifactIntelligenceSynthesisProfile | None
    ) = None
    artifact_merge_planner: ArtifactMergePlannerProfile | None = None
    artifact_export_intelligence: ArtifactExportIntelligenceProfile | None = None
    artifact_engine_contracts: ArtifactIntelligenceEngineContractRegistry | None = None
    creative_critic: CreativeCriticProfile | None = None
    self_evaluation: SelfEvaluationProfile | None = None
    creative_improvement_planner: CreativeImprovementPlannerProfile | None = None
    reflection_loop: ReflectionLoopProfile | None = None
    creative_confidence: CreativeConfidenceProfile | None = None
    creative_score: CreativeScoreProfile | None = None
    creative_director: CreativeAssistantDirectorBrief | None = None
    creative_reasoning: CreativeReasoningResult | None = None
    clarification: ClarificationRequest | None = None
    memory_input: PromptMemoryInput | None = None
    retrieval_input: PromptRetrievalInput | None = None


class PromptInputBuilder(Protocol):
    def build(
        self,
        request: PromptInputRequest,
    ) -> PromptInputResponse:
        """Return structured prompt-ready inputs without rendering prompt text."""


class StructuredPromptInputBuilder:
    """Transform assembled context into prompt-ready structured inputs."""

    def build(
        self,
        request: PromptInputRequest,
    ) -> PromptInputResponse:
        assembled_context = request.assembled_context
        memory_input = _build_memory_input(
            query=request.assistant_request.query,
            memory_context=(
                assembled_context.memory_context
                if assembled_context is not None
                else None
            ),
        )
        retrieval_input = _build_retrieval_input(
            retrieval_context=(
                assembled_context.retrieval_context
                if assembled_context is not None
                else None
            )
        )

        user_input = _build_user_input(
            request.assistant_request,
            route_decision=request.route_decision,
        )
        creative_translation = derive_creative_translation(
            user_input.query,
            domains=user_input.effective_domains,
            selected_runtime=(
                user_input.artifact_refinement.runtime
                if user_input.artifact_refinement is not None
                else None
            ),
            has_image_references=bool(user_input.image_references),
            image_references=user_input.image_references,
            base_translation=(
                user_input.artifact_refinement.creative_translation
                if user_input.artifact_refinement is not None
                else None
            ),
            artifact_content=(
                user_input.artifact_refinement.content
                if user_input.artifact_refinement is not None
                else None
            ),
            refinement_instruction=(
                user_input.artifact_refinement.instruction
                if user_input.artifact_refinement is not None
                else None
            ),
        )
        clarification = (
            derive_hitl_clarification(
                query=user_input.query,
                route_decision=request.route_decision,
                creative_translation=creative_translation,
                clarification_response=user_input.clarification_response,
                artifact_refinement=request.assistant_request.artifact_refinement,
            )
            if request.route_decision is not None
            else None
        )
        prompt_input = PromptInputResponse(
            request=request,
            user_input=user_input,
            creative_translation=creative_translation,
            clarification=clarification,
            memory_input=memory_input,
            retrieval_input=retrieval_input,
        )
        logger.info(
            "Built prompt inputs with memory={} and retrieval={}",
            memory_input is not None,
            retrieval_input is not None,
        )
        return prompt_input


def build_prompt_input_request(
    *,
    assistant_request: AssistantRequest,
    route_decision: RouteDecision,
    assembled_context: AssembledContextResponse | None,
) -> PromptInputRequest:
    return PromptInputRequest(
        route=route_decision.route,
        route_decision=route_decision,
        assistant_request=assistant_request,
        assembled_context=assembled_context,
    )


def _build_user_input(
    assistant_request: AssistantRequest,
    *,
    route_decision: RouteDecision | None,
) -> PromptUserInput:
    ui_selected_domains = assistant_request.domains
    detected_domains = detect_explicit_query_domains(assistant_request.query)
    effective_domains = (
        detected_domains
        if detected_domains
        else route_decision.domains
        if route_decision is not None and route_decision.domains
        else ui_selected_domains
    )
    return PromptUserInput(
        query=assistant_request.query,
        mode=assistant_request.mode,
        domain=effective_domains[0] if len(effective_domains) == 1 else None,
        domains=effective_domains,
        ui_selected_domains=ui_selected_domains,
        detected_domains=detected_domains,
        effective_domains=effective_domains,
        is_follow_up=looks_like_follow_up_query(assistant_request.query),
        image_references=tuple(
            PromptImageReferenceInput(
                id=attachment.id,
                name=attachment.name,
                mime_type=attachment.mime_type,
                size_bytes=attachment.size_bytes,
            )
            for attachment in assistant_request.attachments
        ),
        artifact_refinement=_build_artifact_refinement_input(assistant_request),
        clarification_response=assistant_request.clarification_response,
    )


def _build_artifact_refinement_input(
    assistant_request: AssistantRequest,
) -> PromptArtifactRefinementInput | None:
    refinement = assistant_request.artifact_refinement
    if refinement is None:
        return None

    return PromptArtifactRefinementInput(
        artifact_id=refinement.artifact_id,
        title=refinement.title,
        language=refinement.language,
        content=refinement.content,
        instruction=refinement.instruction,
        domain=refinement.domain,
        runtime=refinement.runtime,
        renderer_id=refinement.renderer_id,
        preview_eligible=refinement.preview_eligible,
        quality_score=refinement.quality_score,
        quality_rank=refinement.quality_rank,
        quality_before=refinement.quality_before,
        pass_number=refinement.pass_number,
        max_passes=refinement.max_passes,
        refinement_objective=refinement.refinement_objective,
        refinement_passes=refinement.refinement_passes,
        critique_rationale=refinement.critique_rationale,
        refinement_guidance=refinement.refinement_guidance,
        creative_translation=_parse_creative_translation(
            refinement.creative_translation
        ),
        creative_plan=_parse_creative_plan(refinement.creative_plan),
    )


def _parse_creative_translation(
    value: dict[str, object] | None,
) -> CreativeTranslation | None:
    if value is None:
        return None
    try:
        return CreativeTranslation.model_validate(value)
    except ValueError:
        logger.warning("Ignored invalid optional creative translation metadata.")
        return None


def _parse_creative_plan(
    value: dict[str, object] | None,
) -> CreativeExecutionPlan | None:
    if value is None:
        return None
    try:
        return CreativeExecutionPlan.model_validate(value)
    except ValueError:
        logger.warning("Ignored invalid optional creative plan metadata.")
        return None


def _build_memory_input(
    *,
    query: str,
    memory_context: MemoryContextResponse | None,
) -> PromptMemoryInput | None:
    if memory_context is None:
        return None

    return PromptMemoryInput(
        recent_turns=build_prompt_recent_turns(
            query=query,
            recent_turns=memory_context.recent_turns,
        ),
        running_summary=_build_running_summary_input(memory_context),
        session_summaries=build_session_memory_summaries(
            memory_context.recent_turns
        ),
        project_memories=_build_project_memory_inputs(memory_context),
    )


def _build_running_summary_input(
    memory_context: MemoryContextResponse,
) -> PromptRunningSummaryInput | None:
    if memory_context.running_summary is None:
        return None
    return PromptRunningSummaryInput(
        content=memory_context.running_summary.content,
        covered_turn_count=memory_context.running_summary.covered_turn_count,
    )


def _build_project_memory_inputs(
    memory_context: MemoryContextResponse,
) -> tuple[PromptProjectMemoryInput, ...]:
    return tuple(
        PromptProjectMemoryInput(
            memory_kind=memory.memory_kind,
            content=memory.content,
            source=memory.source,
        )
        for memory in memory_context.project_memories
    )


def _build_retrieval_input(
    *,
    retrieval_context: RetrievalContextResponse | None,
) -> PromptRetrievalInput | None:
    if retrieval_context is None:
        return None

    return PromptRetrievalInput(
        chunks=tuple(
            PromptKnowledgeChunkInput(
                source_id=chunk.source_id,
                domain=chunk.domain,
                source_type=chunk.source_type,
                publisher=chunk.publisher,
                registry_title=chunk.registry_title,
                document_title=chunk.document_title,
                source_url=chunk.source_url,
                excerpt=chunk.excerpt,
                score=chunk.score,
            )
            for chunk in retrieval_context.chunks
        )
    )


def _selection_shape_for_domains(
    domains: tuple[CreativeCodingDomain, ...],
) -> DomainSelectionShape:
    if not domains:
        return DomainSelectionShape.NONE
    if len(domains) == 1:
        return DomainSelectionShape.SINGLE
    return DomainSelectionShape.MULTI
