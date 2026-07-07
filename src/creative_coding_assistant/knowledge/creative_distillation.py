"""V8.1 creative knowledge distillation and KB reality helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.eval import (
    RetrievalDemoPack,
    build_capstone_retrieval_demo_pack,
    capstone_retrieval_demo_source_ids,
)
from creative_coding_assistant.rag import OfficialSource, approved_official_sources, get_official_source

V8_1_CAPABILITY_ID = "v8_1_creative_knowledge_distillation"
V8_1_DISTILLATION_SCOPE = (
    "Distill creative-production knowledge from registered official sources, "
    "demo retrieval scenarios, repository contracts, and local indexed KB reality."
)
V8_1_AUTHORITY_BOUNDARY = (
    "V8.1 creative knowledge distillation builds typed local reports, "
    "provenance, confidence scores, taxonomy nodes, repository relationships, "
    "and KB reality gaps without fetching external sources, writing Chroma, "
    "mutating source registries, changing retrieval configuration, routing "
    "providers/models, controlling workflows, or claiming future HoloMind or "
    "HOLOiVERSE behavior."
)


class CreativeKnowledgeRecordKind(StrEnum):
    TECHNIQUE = "technique"
    WORKFLOW = "workflow"
    PATTERN = "pattern"
    TAXONOMY = "taxonomy"
    BEST_PRACTICE = "best_practice"
    KB_REALITY = "kb_reality"


class KnowledgeProvenanceKind(StrEnum):
    OFFICIAL_SOURCE = "official_source"
    RETRIEVAL_DEMO = "retrieval_demo"
    REPOSITORY_SURFACE = "repository_surface"
    DOCUMENTATION_SURFACE = "documentation_surface"
    INDEXED_KB = "indexed_kb"


class KnowledgeConfidenceBand(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    GUARDED = "guarded"


class KnowledgeDomainCoverageStatus(StrEnum):
    STRONG = "strong"
    PARTIAL = "partial"
    REGISTERED_ONLY = "registered_only"
    MISSING = "missing"


class IndexedKnowledgeBaseInventory(BaseModel):
    """Read-only inventory of local Chroma state relevant to V8.1."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    sqlite_path: str = Field(min_length=1)
    chroma_exists: bool
    collection_counts: dict[str, int] = Field(default_factory=dict)
    source_chunk_counts: dict[str, int] = Field(default_factory=dict)
    domain_chunk_counts: dict[str, int] = Field(default_factory=dict)
    fetched_at_min: str | None = None
    fetched_at_max: str | None = None

    @property
    def indexed_source_ids(self) -> tuple[str, ...]:
        return tuple(sorted(source_id for source_id, count in self.source_chunk_counts.items() if count > 0))

    @property
    def indexed_domain_ids(self) -> tuple[str, ...]:
        return tuple(sorted(domain_id for domain_id, count in self.domain_chunk_counts.items() if count > 0))


class KnowledgeDomainCoverage(BaseModel):
    """Registry-vs-index coverage for one creative domain."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    domain: CreativeCodingDomain
    registry_source_count: int = Field(ge=0)
    indexed_source_count: int = Field(ge=0)
    indexed_chunk_count: int = Field(ge=0)
    demo_required_source_count: int = Field(ge=0)
    demo_indexed_source_count: int = Field(ge=0)
    status: KnowledgeDomainCoverageStatus
    missing_demo_source_ids: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _status_matches_counts(self) -> Self:
        expected = _domain_coverage_status(
            registry_source_count=self.registry_source_count,
            indexed_source_count=self.indexed_source_count,
            demo_required_source_count=self.demo_required_source_count,
            demo_indexed_source_count=self.demo_indexed_source_count,
        )
        if self.status != expected:
            raise ValueError("status must match registry, indexed, and demo coverage")
        if self.demo_indexed_source_count > self.demo_required_source_count:
            raise ValueError("demo_indexed_source_count cannot exceed demo_required_source_count")
        return self


class KnowledgeBaseRealitySnapshot(BaseModel):
    """Honest report separating approved registry coverage from indexed KB reality."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    registry_source_count: int = Field(ge=0)
    registry_domain_count: int = Field(ge=0)
    indexed_source_count: int = Field(ge=0)
    indexed_domain_count: int = Field(ge=0)
    indexed_chunk_count: int = Field(ge=0)
    demo_required_source_count: int = Field(ge=0)
    demo_indexed_source_count: int = Field(ge=0)
    indexed_source_ids: tuple[str, ...] = Field(default_factory=tuple)
    unindexed_source_ids: tuple[str, ...] = Field(default_factory=tuple)
    demo_required_source_ids: tuple[str, ...] = Field(default_factory=tuple)
    unindexed_demo_source_ids: tuple[str, ...] = Field(default_factory=tuple)
    domain_coverage: tuple[KnowledgeDomainCoverage, ...] = Field(default_factory=tuple)
    strengths: tuple[str, ...] = Field(default_factory=tuple)
    weaknesses: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _snapshot_matches_counts(self) -> Self:
        if self.indexed_source_count != len(self.indexed_source_ids):
            raise ValueError("indexed_source_count must match indexed_source_ids")
        if self.demo_required_source_count != len(self.demo_required_source_ids):
            raise ValueError("demo_required_source_count must match demo source ids")
        if self.demo_indexed_source_count != (
            self.demo_required_source_count - len(self.unindexed_demo_source_ids)
        ):
            raise ValueError("demo_indexed_source_count must match unindexed demo sources")
        return self


class CreativeKnowledgeConfidence(BaseModel):
    """Confidence score for a distilled record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    score: float = Field(ge=0, le=1)
    band: KnowledgeConfidenceBand
    indexed_source_ratio: float = Field(ge=0, le=1)
    provenance_count: int = Field(ge=1)
    domain_count: int = Field(ge=1)
    caveats: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _band_matches_score(self) -> Self:
        if self.band != _confidence_band(self.score):
            raise ValueError("band must match score")
        return self


class CreativeKnowledgeProvenance(BaseModel):
    """Traceable evidence behind a distilled creative knowledge record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    provenance_id: str = Field(min_length=1)
    kind: KnowledgeProvenanceKind
    title: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    domain: CreativeCodingDomain | None = None
    indexed: bool = False
    indexed_chunk_count: int = Field(default=0, ge=0)
    note: str = Field(min_length=1, max_length=420)

    @model_validator(mode="after")
    def _indexed_count_matches_flag(self) -> Self:
        if self.indexed and self.indexed_chunk_count < 1:
            raise ValueError("indexed provenance requires at least one indexed chunk")
        if not self.indexed and self.indexed_chunk_count != 0:
            raise ValueError("unindexed provenance cannot declare indexed chunks")
        return self


class CreativeKnowledgeRecord(BaseModel):
    """One distilled V8.1 creative knowledge artifact."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(min_length=1)
    kind: CreativeKnowledgeRecordKind
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=520)
    domains: tuple[CreativeCodingDomain, ...] = Field(min_length=1)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    technique_tags: tuple[str, ...] = Field(default_factory=tuple)
    workflow_steps: tuple[str, ...] = Field(default_factory=tuple)
    pattern_tags: tuple[str, ...] = Field(default_factory=tuple)
    taxonomy_path: tuple[str, ...] = Field(min_length=1)
    provenance: tuple[CreativeKnowledgeProvenance, ...] = Field(min_length=1)
    confidence: CreativeKnowledgeConfidence

    @field_validator("domains", mode="before")
    @classmethod
    def _normalize_domains(
        cls,
        value: Sequence[CreativeCodingDomain | str] | CreativeCodingDomain | str,
    ) -> tuple[CreativeCodingDomain, ...]:
        if isinstance(value, CreativeCodingDomain):
            return (value,)
        if isinstance(value, str):
            return (CreativeCodingDomain(value),)
        normalized: list[CreativeCodingDomain] = []
        for item in value:
            domain = item if isinstance(item, CreativeCodingDomain) else CreativeCodingDomain(str(item))
            if domain not in normalized:
                normalized.append(domain)
        return tuple(normalized)

    @model_validator(mode="after")
    def _record_matches_sources(self) -> Self:
        provenance_refs = {item.reference for item in self.provenance}
        missing_source_ids = tuple(source_id for source_id in self.source_ids if source_id not in provenance_refs)
        if missing_source_ids:
            raise ValueError("source_ids must be represented in provenance references")
        if self.kind is CreativeKnowledgeRecordKind.WORKFLOW and not self.workflow_steps:
            raise ValueError("workflow records require workflow_steps")
        if self.kind is CreativeKnowledgeRecordKind.TECHNIQUE and not self.technique_tags:
            raise ValueError("technique records require technique_tags")
        if self.kind is CreativeKnowledgeRecordKind.PATTERN and not self.pattern_tags:
            raise ValueError("pattern records require pattern_tags")
        return self


class CreativeKnowledgeRelationship(BaseModel):
    """Relationship between distilled creative knowledge records."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    relationship_id: str = Field(min_length=1)
    from_record_id: str = Field(min_length=1)
    to_record_id: str = Field(min_length=1)
    relationship_type: Literal["shared_source", "shared_domain", "workflow_supports_pattern"]
    evidence_source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _relationship_not_self_referential(self) -> Self:
        if self.from_record_id == self.to_record_id:
            raise ValueError("relationship cannot connect a record to itself")
        return self


class CreativeKnowledgeTaxonomyNode(BaseModel):
    """Taxonomy node produced from distilled record paths."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    node_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    parent_node_id: str | None = None
    record_ids: tuple[str, ...] = Field(default_factory=tuple)


class CreativeKnowledgeRepositoryNode(BaseModel):
    """Repository or documentation surface that contributes to V8.1 knowledge."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    node_id: str = Field(min_length=1)
    surface_kind: Literal["source", "test", "documentation", "architecture"]
    path: str = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=420)
    roadmap_items: tuple[str, ...] = Field(min_length=1)


class CreativeKnowledgeRepositoryEdge(BaseModel):
    """Read-only relationship between repository knowledge surfaces."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    edge_id: str = Field(min_length=1)
    from_node_id: str = Field(min_length=1)
    to_node_id: str = Field(min_length=1)
    relationship: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def _edge_matches_nodes(self) -> Self:
        if self.from_node_id == self.to_node_id:
            raise ValueError("repository edge cannot point to itself")
        return self


class CreativeKnowledgeRepositoryGraph(BaseModel):
    """Repository/documentation relationship graph for V8.1 audit surfaces."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    graph_id: Literal["v8_1_repository_knowledge_graph"] = "v8_1_repository_knowledge_graph"
    nodes: tuple[CreativeKnowledgeRepositoryNode, ...] = Field(min_length=1)
    edges: tuple[CreativeKnowledgeRepositoryEdge, ...] = Field(default_factory=tuple)
    mutation_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _edges_reference_nodes(self) -> Self:
        node_ids = {node.node_id for node in self.nodes}
        for edge in self.edges:
            if edge.from_node_id not in node_ids or edge.to_node_id not in node_ids:
                raise ValueError("repository edges must reference graph nodes")
        return self


class KnowledgeHardeningAction(BaseModel):
    """Capability-scoped RAG/KB hardening recommendation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    action_id: str = Field(min_length=1)
    priority: Literal["high", "medium", "low"]
    summary: str = Field(min_length=1, max_length=420)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    execution_boundary: str = Field(min_length=1, max_length=420)


class CreativeKnowledgeDistillationReport(BaseModel):
    """Top-level V8.1 creative knowledge distillation report."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    capability_id: Literal["v8_1_creative_knowledge_distillation"] = V8_1_CAPABILITY_ID
    distillation_scope: str = Field(default=V8_1_DISTILLATION_SCOPE, min_length=1)
    authority_boundary: str = Field(default=V8_1_AUTHORITY_BOUNDARY, min_length=1)
    kb_reality: KnowledgeBaseRealitySnapshot
    records: tuple[CreativeKnowledgeRecord, ...] = Field(min_length=1)
    relationships: tuple[CreativeKnowledgeRelationship, ...] = Field(default_factory=tuple)
    taxonomy_nodes: tuple[CreativeKnowledgeTaxonomyNode, ...] = Field(default_factory=tuple)
    repository_graph: CreativeKnowledgeRepositoryGraph
    hardening_actions: tuple[KnowledgeHardeningAction, ...] = Field(default_factory=tuple)
    implemented_roadmap_items: tuple[str, ...] = Field(min_length=1)
    deferred_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    external_fetch_implemented: Literal[False] = False
    chroma_write_implemented: Literal[False] = False
    source_registry_mutation_implemented: Literal[False] = False
    retrieval_configuration_mutation_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    holomind_implemented: Literal[False] = False
    holoiverse_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _report_matches_records(self) -> Self:
        record_ids = {record.record_id for record in self.records}
        for relationship in self.relationships:
            if relationship.from_record_id not in record_ids or relationship.to_record_id not in record_ids:
                raise ValueError("relationships must reference distilled records")
        taxonomy_record_ids = {record_id for node in self.taxonomy_nodes for record_id in node.record_ids}
        if not taxonomy_record_ids.issubset(record_ids):
            raise ValueError("taxonomy nodes must reference distilled records")
        return self


@dataclass(frozen=True)
class _ScenarioBlueprint:
    kind: CreativeKnowledgeRecordKind
    title: str
    summary: str
    technique_tags: tuple[str, ...]
    workflow_steps: tuple[str, ...]
    pattern_tags: tuple[str, ...]
    taxonomy_path: tuple[str, ...]


_SCENARIO_BLUEPRINTS: dict[str, _ScenarioBlueprint] = {
    "runtime_selection_hydra_vs_p5": _ScenarioBlueprint(
        kind=CreativeKnowledgeRecordKind.WORKFLOW,
        title="Live visual runtime triage",
        summary=(
            "Distills a practical workflow for choosing between Hydra feedback chains "
            "and p5.js sketch control when fast browser iteration is the goal."
        ),
        technique_tags=("feedback_systems", "runtime_triage", "live_coding"),
        workflow_steps=("identify feedback needs", "compare live-coding latency", "select controllable sketch path"),
        pattern_tags=("feedback_loop", "runtime_selection"),
        taxonomy_path=("creative production", "runtime choice", "live visuals"),
    ),
    "audio_reactive_browser_mapping": _ScenarioBlueprint(
        kind=CreativeKnowledgeRecordKind.TECHNIQUE,
        title="Audio-reactive browser mapping",
        summary=(
            "Distills bass, mids, amplitude, and FFT mapping as an evidence-backed "
            "creative technique for browser audiovisual systems."
        ),
        technique_tags=("audio_reactive_mappings", "fft_mapping", "amplitude_envelope"),
        workflow_steps=("separate amplitude and frequency bands", "map bands to motion", "guard browser audio startup"),
        pattern_tags=("signal_to_motion", "browser_audio_analysis"),
        taxonomy_path=("creative production", "audio reactive", "signal mapping"),
    ),
    "shader_post_fx_pipeline": _ScenarioBlueprint(
        kind=CreativeKnowledgeRecordKind.PATTERN,
        title="Shader and post-processing pipeline",
        summary=(
            "Distills a pattern for glow-heavy browser visuals that combines shader "
            "references, post-processing passes, and render-pipeline tradeoffs."
        ),
        technique_tags=("shader_composition", "post_processing", "glow_pipeline"),
        workflow_steps=("choose shader or pass boundary", "compose bloom/glow stages", "check render target cost"),
        pattern_tags=("shader_pipeline", "post_fx", "kaleidoscopic_composition"),
        taxonomy_path=("creative production", "shader systems", "post effects"),
    ),
    "audiovisual_composition_browser_set": _ScenarioBlueprint(
        kind=CreativeKnowledgeRecordKind.WORKFLOW,
        title="Audiovisual timing workflow",
        summary=(
            "Distills a workflow for coordinating loop timing, sample playback, "
            "visual accents, and browser runtime constraints."
        ),
        technique_tags=("transport_timing", "visual_accent_mapping", "loop_coordination"),
        workflow_steps=("establish transport clock", "bind sample events to visual accents", "budget render updates"),
        pattern_tags=("timebase_alignment", "audio_visual_sync"),
        taxonomy_path=("creative production", "audiovisual composition", "timing"),
    ),
    "creative_debugging_silent_audio": _ScenarioBlueprint(
        kind=CreativeKnowledgeRecordKind.BEST_PRACTICE,
        title="Silent browser audio debugging",
        summary=(
            "Distills a best-practice checklist for browser audio startup, analyser "
            "wiring, p5.sound input, and flat-signal diagnosis."
        ),
        technique_tags=("audio_debugging", "analyser_node", "p5_sound"),
        workflow_steps=("confirm user gesture unlock", "inspect analyser node wiring", "verify input source and gain"),
        pattern_tags=("browser_audio_startup", "flat_analyser_signal"),
        taxonomy_path=("creative production", "debugging", "browser audio"),
    ),
    "creative_debugging_three_effects": _ScenarioBlueprint(
        kind=CreativeKnowledgeRecordKind.BEST_PRACTICE,
        title="Three.js effects debugging",
        summary=(
            "Distills render-target, shadow, and post-processing checks for slow or "
            "visually incorrect Three.js effect pipelines."
        ),
        technique_tags=("render_target_debugging", "shadow_tradeoffs", "post_processing"),
        workflow_steps=("inspect render target sizes", "profile post-processing passes", "review shadow settings"),
        pattern_tags=("render_pipeline_debugging", "effect_budgeting"),
        taxonomy_path=("creative production", "debugging", "three.js effects"),
    ),
    "symbol_to_art_operational_translation": _ScenarioBlueprint(
        kind=CreativeKnowledgeRecordKind.PATTERN,
        title="Symbolic motif to operational visual system",
        summary=(
            "Distills a pattern for translating symbolic geometry into concrete "
            "browser motion, rhythm, runtime, and mapping decisions."
        ),
        technique_tags=("recursive_geometry", "motif_translation", "rhythmic_motion"),
        workflow_steps=(
            "reduce symbol to geometry primitives",
            "select motion grammar",
            "ground in runtime constraints",
        ),
        pattern_tags=("mandala_motif", "operational_translation", "morphogenesis_seed"),
        taxonomy_path=("creative production", "morphogenesis", "symbolic geometry"),
    ),
}


def inventory_local_chroma_kb(
    sqlite_path: Path | str = Path("data/chroma/chroma.sqlite3"),
) -> IndexedKnowledgeBaseInventory:
    """Return a read-only inventory of the local Chroma SQLite metadata."""

    path = Path(sqlite_path)
    if not path.exists():
        return IndexedKnowledgeBaseInventory(sqlite_path=str(path), chroma_exists=False)

    with sqlite3.connect(path) as connection:
        collection_counts = _query_collection_counts(connection)
        source_chunk_counts = _query_kb_metadata_counts(connection, "source_id")
        domain_chunk_counts = _query_kb_metadata_counts(connection, "domain")
        fetched_at_min, fetched_at_max = _query_kb_fetched_range(connection)

    return IndexedKnowledgeBaseInventory(
        sqlite_path=str(path),
        chroma_exists=True,
        collection_counts=collection_counts,
        source_chunk_counts=source_chunk_counts,
        domain_chunk_counts=domain_chunk_counts,
        fetched_at_min=fetched_at_min,
        fetched_at_max=fetched_at_max,
    )


def build_kb_reality_snapshot(
    *,
    indexed_chunk_counts_by_source: Mapping[str, int] | None = None,
    registry_sources: Sequence[OfficialSource] | None = None,
    demo_source_ids: Sequence[str] | None = None,
) -> KnowledgeBaseRealitySnapshot:
    """Build a registry-vs-indexed KB snapshot without mutating Chroma."""

    sources = tuple(registry_sources or approved_official_sources())
    source_by_id = {source.source_id: source for source in sources}
    indexed_counts = {
        source_id: int(count)
        for source_id, count in (indexed_chunk_counts_by_source or {}).items()
        if count > 0
    }
    indexed_source_ids = tuple(source_id for source_id in source_by_id if indexed_counts.get(source_id, 0) > 0)
    demo_ids = tuple(demo_source_ids or capstone_retrieval_demo_source_ids())
    unindexed_demo_ids = tuple(source_id for source_id in demo_ids if indexed_counts.get(source_id, 0) == 0)
    unindexed_source_ids = tuple(source_id for source_id in source_by_id if indexed_counts.get(source_id, 0) == 0)
    domain_coverage = _build_domain_coverage(
        sources=sources,
        indexed_counts=indexed_counts,
        demo_source_ids=demo_ids,
    )
    indexed_domain_count = len(
        {
            source_by_id[source_id].domain
            for source_id in indexed_source_ids
            if source_id in source_by_id
        }
    )

    return KnowledgeBaseRealitySnapshot(
        registry_source_count=len(sources),
        registry_domain_count=len({source.domain for source in sources}),
        indexed_source_count=len(indexed_source_ids),
        indexed_domain_count=indexed_domain_count,
        indexed_chunk_count=sum(indexed_counts.values()),
        demo_required_source_count=len(demo_ids),
        demo_indexed_source_count=len(demo_ids) - len(unindexed_demo_ids),
        indexed_source_ids=indexed_source_ids,
        unindexed_source_ids=unindexed_source_ids,
        demo_required_source_ids=demo_ids,
        unindexed_demo_source_ids=unindexed_demo_ids,
        domain_coverage=domain_coverage,
        strengths=_kb_strengths(domain_coverage),
        weaknesses=_kb_weaknesses(domain_coverage, unindexed_demo_ids),
    )


def build_repository_knowledge_graph() -> CreativeKnowledgeRepositoryGraph:
    """Return the V8.1 repository/documentation relationship graph."""

    nodes = (
        CreativeKnowledgeRepositoryNode(
            node_id="repo::source_registry",
            surface_kind="source",
            path="src/creative_coding_assistant/rag/sources.py",
            summary="Approved official source registry and domain/source governance.",
            roadmap_items=("Repository Distillation", "Documentation Distillation", "Knowledge Provenance"),
        ),
        CreativeKnowledgeRepositoryNode(
            node_id="repo::retrieval_models",
            surface_kind="source",
            path="src/creative_coding_assistant/rag/retrieval/models.py",
            summary="Typed official KB retrieval request, filter, result, and response contracts.",
            roadmap_items=("RAG Knowledge Hardening", "KB Confidence Scoring"),
        ),
        CreativeKnowledgeRepositoryNode(
            node_id="repo::retrieval_runtime",
            surface_kind="source",
            path="src/creative_coding_assistant/orchestration/runtime/retrieval.py",
            summary="Runtime adapter that converts official KB search results into orchestration context chunks.",
            roadmap_items=("Workflow Extraction", "RAG Knowledge Hardening"),
        ),
        CreativeKnowledgeRepositoryNode(
            node_id="repo::demo_pack",
            surface_kind="source",
            path="src/creative_coding_assistant/eval/retrieval_demo_pack.py",
            summary="Capstone retrieval scenarios used as demo-domain creative knowledge seeds.",
            roadmap_items=("Demo KB Expansion", "Demo Domain Sync", "Knowledge Pattern Extraction"),
        ),
        CreativeKnowledgeRepositoryNode(
            node_id="doc::eval_pipeline",
            surface_kind="documentation",
            path="docs/eval_pipeline.md",
            summary="Retrieval evaluation pipeline and latest-sample evaluation boundary.",
            roadmap_items=("Knowledge Quality Validation", "RAG Knowledge Hardening"),
        ),
        CreativeKnowledgeRepositoryNode(
            node_id="architecture::engine_matrix",
            surface_kind="architecture",
            path="architecture/engine_matrix.md",
            summary="Version and engine boundaries for retrieval, memory, knowledge, and future OS claims.",
            roadmap_items=("Repository Relationship Graph", "Creative Knowledge Graph Enrichment"),
        ),
    )
    edges = (
        CreativeKnowledgeRepositoryEdge(
            edge_id="repo::source_registry->repo::demo_pack",
            from_node_id="repo::source_registry",
            to_node_id="repo::demo_pack",
            relationship="Demo scenarios must reference approved source IDs from the official registry.",
        ),
        CreativeKnowledgeRepositoryEdge(
            edge_id="repo::source_registry->repo::retrieval_models",
            from_node_id="repo::source_registry",
            to_node_id="repo::retrieval_models",
            relationship="Retrieval result contracts preserve source identity, domain, publisher, and source type.",
        ),
        CreativeKnowledgeRepositoryEdge(
            edge_id="repo::retrieval_models->repo::retrieval_runtime",
            from_node_id="repo::retrieval_models",
            to_node_id="repo::retrieval_runtime",
            relationship="Runtime retrieval context is adapted from official KB search result contracts.",
        ),
        CreativeKnowledgeRepositoryEdge(
            edge_id="repo::demo_pack->doc::eval_pipeline",
            from_node_id="repo::demo_pack",
            to_node_id="doc::eval_pipeline",
            relationship="Demo scenarios define focused retrieval samples that can be validated by latest-sample eval.",
        ),
        CreativeKnowledgeRepositoryEdge(
            edge_id="architecture::engine_matrix->repo::source_registry",
            from_node_id="architecture::engine_matrix",
            to_node_id="repo::source_registry",
            relationship="Architecture claims must distinguish registry coverage from indexed KB coverage.",
        ),
    )
    return CreativeKnowledgeRepositoryGraph(nodes=nodes, edges=edges)


def build_v8_1_creative_knowledge_distillation(
    *,
    indexed_chunk_counts_by_source: Mapping[str, int] | None = None,
    demo_pack: RetrievalDemoPack | None = None,
) -> CreativeKnowledgeDistillationReport:
    """Build the V8.1 creative knowledge distillation report."""

    pack = demo_pack or build_capstone_retrieval_demo_pack()
    kb_reality = build_kb_reality_snapshot(
        indexed_chunk_counts_by_source=indexed_chunk_counts_by_source,
        demo_source_ids=capstone_retrieval_demo_source_ids(),
    )
    indexed_counts = {
        source_id: int(count)
        for source_id, count in (indexed_chunk_counts_by_source or {}).items()
        if count > 0
    }
    demo_provenance = _demo_pack_provenance(pack)
    records = tuple(
        _build_scenario_record(
            scenario_id=scenario.demo_id,
            domains=scenario.domains,
            source_ids=scenario.expected_source_ids,
            indexed_counts=indexed_counts,
            demo_provenance=demo_provenance,
        )
        for scenario in pack.scenarios
    )
    relationships = _build_record_relationships(records)
    taxonomy_nodes = _build_taxonomy_nodes(records)

    return CreativeKnowledgeDistillationReport(
        kb_reality=kb_reality,
        records=records,
        relationships=relationships,
        taxonomy_nodes=taxonomy_nodes,
        repository_graph=build_repository_knowledge_graph(),
        hardening_actions=_build_hardening_actions(kb_reality),
        implemented_roadmap_items=(
            "Repository Distillation",
            "Documentation Distillation",
            "Knowledge Pattern Extraction",
            "Workflow Extraction",
            "Creative Technique Extraction",
            "Taxonomy Builder",
            "Creative Knowledge Graph Enrichment",
            "Best Practice Extraction",
            "Cross-Domain Relationship Discovery",
            "Morphogenesis Pattern Distillation",
            "KB Reality Audit",
            "RAG Knowledge Hardening",
            "Knowledge Quality Validation",
            "Repository Relationship Graph",
            "Creative Workflow Graph",
            "Knowledge Provenance",
            "KB Confidence Scoring",
        ),
        deferred_roadmap_items=(
            "PDF/Paper Distillation",
            "Demo KB Expansion",
            "Demo Domain Sync",
        ),
    )


def _query_collection_counts(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute(
        """
        select c.name, count(e.id)
        from collections c
        left join segments s on s.collection = c.id
        left join embeddings e on e.segment_id = s.id
        group by c.name
        order by c.name
        """
    ).fetchall()
    return {str(name): int(count) for name, count in rows}


def _query_kb_metadata_counts(connection: sqlite3.Connection, key: str) -> dict[str, int]:
    rows = connection.execute(
        """
        select m.string_value, count(distinct e.id)
        from embedding_metadata m
        join embeddings e on e.id = m.id
        join segments s on s.id = e.segment_id
        join collections c on c.id = s.collection
        where c.name = 'kb_official_docs' and m.key = ?
        group by m.string_value
        order by m.string_value
        """,
        (key,),
    ).fetchall()
    return {str(value): int(count) for value, count in rows if value is not None}


def _query_kb_fetched_range(connection: sqlite3.Connection) -> tuple[str | None, str | None]:
    row = connection.execute(
        """
        select min(m.string_value), max(m.string_value)
        from embedding_metadata m
        join embeddings e on e.id = m.id
        join segments s on s.id = e.segment_id
        join collections c on c.id = s.collection
        where c.name = 'kb_official_docs' and m.key = 'fetched_at'
        """
    ).fetchone()
    if row is None:
        return None, None
    return row[0], row[1]


def _build_domain_coverage(
    *,
    sources: Sequence[OfficialSource],
    indexed_counts: Mapping[str, int],
    demo_source_ids: Sequence[str],
) -> tuple[KnowledgeDomainCoverage, ...]:
    source_by_id = {source.source_id: source for source in sources}
    demo_ids = tuple(demo_source_ids)
    coverage: list[KnowledgeDomainCoverage] = []
    for domain in sorted({source.domain for source in sources}, key=lambda item: item.value):
        domain_sources = tuple(source for source in sources if source.domain == domain)
        indexed_sources = tuple(source for source in domain_sources if indexed_counts.get(source.source_id, 0) > 0)
        required_demo_sources = tuple(
            source_by_id[source_id]
            for source_id in demo_ids
            if source_id in source_by_id and source_by_id[source_id].domain == domain
        )
        missing_demo_source_ids = tuple(
            source.source_id for source in required_demo_sources if indexed_counts.get(source.source_id, 0) == 0
        )
        demo_indexed_source_count = len(required_demo_sources) - len(missing_demo_source_ids)
        coverage.append(
            KnowledgeDomainCoverage(
                domain=domain,
                registry_source_count=len(domain_sources),
                indexed_source_count=len(indexed_sources),
                indexed_chunk_count=sum(indexed_counts.get(source.source_id, 0) for source in domain_sources),
                demo_required_source_count=len(required_demo_sources),
                demo_indexed_source_count=demo_indexed_source_count,
                status=_domain_coverage_status(
                    registry_source_count=len(domain_sources),
                    indexed_source_count=len(indexed_sources),
                    demo_required_source_count=len(required_demo_sources),
                    demo_indexed_source_count=demo_indexed_source_count,
                ),
                missing_demo_source_ids=missing_demo_source_ids,
            )
        )
    return tuple(coverage)


def _domain_coverage_status(
    *,
    registry_source_count: int,
    indexed_source_count: int,
    demo_required_source_count: int,
    demo_indexed_source_count: int,
) -> KnowledgeDomainCoverageStatus:
    if registry_source_count == 0:
        return KnowledgeDomainCoverageStatus.MISSING
    if indexed_source_count == 0:
        return KnowledgeDomainCoverageStatus.REGISTERED_ONLY
    if demo_required_source_count and demo_indexed_source_count == demo_required_source_count:
        return KnowledgeDomainCoverageStatus.STRONG
    return KnowledgeDomainCoverageStatus.PARTIAL


def _kb_strengths(domain_coverage: Sequence[KnowledgeDomainCoverage]) -> tuple[str, ...]:
    strong_domains = tuple(
        item.domain.value
        for item in domain_coverage
        if item.status is KnowledgeDomainCoverageStatus.STRONG
    )
    partial_domains = tuple(
        item.domain.value
        for item in domain_coverage
        if item.status is KnowledgeDomainCoverageStatus.PARTIAL
    )
    strengths: list[str] = []
    if strong_domains:
        strengths.append(f"Demo-required sources are indexed for: {', '.join(strong_domains)}.")
    if partial_domains:
        strengths.append(f"Indexed retrieval coverage exists for: {', '.join(partial_domains[:8])}.")
    if not strengths:
        strengths.append("Approved registry is available even when local indexed coverage is empty.")
    return tuple(strengths)


def _kb_weaknesses(
    domain_coverage: Sequence[KnowledgeDomainCoverage],
    unindexed_demo_source_ids: Sequence[str],
) -> tuple[str, ...]:
    registered_only = tuple(
        item.domain.value for item in domain_coverage if item.status is KnowledgeDomainCoverageStatus.REGISTERED_ONLY
    )
    weaknesses: list[str] = []
    if unindexed_demo_source_ids:
        weaknesses.append(
            "Demo-required sources not indexed locally: " + ", ".join(unindexed_demo_source_ids) + "."
        )
    if registered_only:
        weaknesses.append(
            "Registry-only domains cannot be claimed as indexed KB coverage: "
            f"{', '.join(registered_only[:12])}."
        )
    return tuple(weaknesses)


def _demo_pack_provenance(pack: RetrievalDemoPack) -> CreativeKnowledgeProvenance:
    return CreativeKnowledgeProvenance(
        provenance_id=f"demo_pack::{pack.pack_id}",
        kind=KnowledgeProvenanceKind.RETRIEVAL_DEMO,
        title=pack.title,
        reference=pack.pack_id,
        indexed=False,
        note="Capstone retrieval scenario pack provides bounded creative-production distillation seeds.",
    )


def _build_scenario_record(
    *,
    scenario_id: str,
    domains: tuple[CreativeCodingDomain, ...],
    source_ids: tuple[str, ...],
    indexed_counts: Mapping[str, int],
    demo_provenance: CreativeKnowledgeProvenance,
) -> CreativeKnowledgeRecord:
    blueprint = _SCENARIO_BLUEPRINTS[scenario_id]
    provenance = (demo_provenance,) + tuple(_source_provenance(source_id, indexed_counts) for source_id in source_ids)
    confidence = _record_confidence(
        source_ids=source_ids,
        domains=domains,
        provenance=provenance,
        indexed_counts=indexed_counts,
        signal_count=len(blueprint.technique_tags) + len(blueprint.workflow_steps) + len(blueprint.pattern_tags),
    )
    return CreativeKnowledgeRecord(
        record_id=f"creative_knowledge::{scenario_id}",
        kind=blueprint.kind,
        title=blueprint.title,
        summary=blueprint.summary,
        domains=domains,
        source_ids=source_ids,
        technique_tags=blueprint.technique_tags,
        workflow_steps=blueprint.workflow_steps,
        pattern_tags=blueprint.pattern_tags,
        taxonomy_path=blueprint.taxonomy_path,
        provenance=provenance,
        confidence=confidence,
    )


def _source_provenance(
    source_id: str,
    indexed_counts: Mapping[str, int],
) -> CreativeKnowledgeProvenance:
    source = get_official_source(source_id)
    indexed_count = indexed_counts.get(source_id, 0)
    indexed = indexed_count > 0
    return CreativeKnowledgeProvenance(
        provenance_id=f"official_source::{source.source_id}",
        kind=KnowledgeProvenanceKind.OFFICIAL_SOURCE if not indexed else KnowledgeProvenanceKind.INDEXED_KB,
        title=source.title,
        reference=source.source_id,
        domain=source.domain,
        indexed=indexed,
        indexed_chunk_count=indexed_count,
        note=(
            "Official source is indexed in local Chroma."
            if indexed
            else "Official source is registered but not present in the supplied indexed KB inventory."
        ),
    )


def _record_confidence(
    *,
    source_ids: Sequence[str],
    domains: Sequence[CreativeCodingDomain],
    provenance: Sequence[CreativeKnowledgeProvenance],
    indexed_counts: Mapping[str, int],
    signal_count: int,
) -> CreativeKnowledgeConfidence:
    indexed_source_count = sum(1 for source_id in source_ids if indexed_counts.get(source_id, 0) > 0)
    indexed_ratio = indexed_source_count / len(source_ids) if source_ids else 0
    provenance_score = min(len(provenance) / 5, 1)
    domain_score = min(len(domains) / 3, 1)
    signal_score = min(signal_count / 9, 1)
    score = round(
        min(1.0, indexed_ratio * 0.45 + provenance_score * 0.25 + domain_score * 0.15 + signal_score * 0.15),
        3,
    )
    caveats: list[str] = []
    unindexed = tuple(source_id for source_id in source_ids if indexed_counts.get(source_id, 0) == 0)
    if unindexed:
        caveats.append("Unindexed source references: " + ", ".join(unindexed) + ".")
    return CreativeKnowledgeConfidence(
        score=score,
        band=_confidence_band(score),
        indexed_source_ratio=round(indexed_ratio, 3),
        provenance_count=len(provenance),
        domain_count=len(tuple(domains)),
        caveats=tuple(caveats),
    )


def _confidence_band(score: float) -> KnowledgeConfidenceBand:
    if score >= 0.75:
        return KnowledgeConfidenceBand.HIGH
    if score >= 0.55:
        return KnowledgeConfidenceBand.MEDIUM
    if score >= 0.35:
        return KnowledgeConfidenceBand.LOW
    return KnowledgeConfidenceBand.GUARDED


def _build_record_relationships(
    records: Sequence[CreativeKnowledgeRecord],
) -> tuple[CreativeKnowledgeRelationship, ...]:
    relationships: list[CreativeKnowledgeRelationship] = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            shared_sources = tuple(source_id for source_id in left.source_ids if source_id in right.source_ids)
            shared_domains = tuple(domain.value for domain in left.domains if domain in right.domains)
            if shared_sources:
                relationship_type: Literal[
                    "shared_source",
                    "shared_domain",
                    "workflow_supports_pattern",
                ] = "shared_source"
                evidence_source_ids = shared_sources
            elif shared_domains:
                relationship_type = "shared_domain"
                evidence_source_ids = ()
            elif (
                left.kind is CreativeKnowledgeRecordKind.WORKFLOW
                and right.kind is CreativeKnowledgeRecordKind.PATTERN
            ):
                relationship_type = "workflow_supports_pattern"
                evidence_source_ids = ()
            else:
                continue
            relationships.append(
                CreativeKnowledgeRelationship(
                    relationship_id=f"{left.record_id}->{right.record_id}",
                    from_record_id=left.record_id,
                    to_record_id=right.record_id,
                    relationship_type=relationship_type,
                    evidence_source_ids=evidence_source_ids,
                    confidence=min(left.confidence.score, right.confidence.score),
                )
            )
    return tuple(relationships)


def _build_taxonomy_nodes(records: Sequence[CreativeKnowledgeRecord]) -> tuple[CreativeKnowledgeTaxonomyNode, ...]:
    node_records: dict[str, list[str]] = {}
    node_labels: dict[str, str] = {}
    node_parents: dict[str, str | None] = {}
    for record in records:
        parent_id: str | None = None
        path: list[str] = []
        for label in record.taxonomy_path:
            path.append(_slug(label))
            node_id = "taxonomy::" + "::".join(path)
            node_labels[node_id] = label
            node_parents[node_id] = parent_id
            node_records.setdefault(node_id, []).append(record.record_id)
            parent_id = node_id
    return tuple(
        CreativeKnowledgeTaxonomyNode(
            node_id=node_id,
            label=node_labels[node_id],
            parent_node_id=node_parents[node_id],
            record_ids=tuple(_dedupe(node_records[node_id])),
        )
        for node_id in sorted(node_labels)
    )


def _build_hardening_actions(snapshot: KnowledgeBaseRealitySnapshot) -> tuple[KnowledgeHardeningAction, ...]:
    actions: list[KnowledgeHardeningAction] = []
    if snapshot.unindexed_demo_source_ids:
        actions.append(
            KnowledgeHardeningAction(
                action_id="v8_1_hardening::index_demo_required_sources",
                priority="high",
                summary="Run a focused, approved-source sync for demo-required sources missing from local Chroma.",
                source_ids=snapshot.unindexed_demo_source_ids,
                execution_boundary=(
                    "Requires explicit sync/reindex action; this report does not "
                    "fetch or write KB data."
                ),
            )
        )
    if snapshot.unindexed_source_ids:
        actions.append(
            KnowledgeHardeningAction(
                action_id="v8_1_hardening::separate_registry_from_index_claims",
                priority="high",
                summary=(
                    "Keep public and demo claims explicit that registry coverage is "
                    "broader than indexed KB coverage."
                ),
                source_ids=(),
                execution_boundary="Documentation and audit language only; no source registry mutation.",
            )
        )
    actions.append(
        KnowledgeHardeningAction(
            action_id="v8_1_hardening::refresh_eval_after_sync",
            priority="medium",
            summary=(
                "After any focused demo-source sync, rerun latest-sample retrieval "
                "evaluation instead of stale fixtures."
            ),
            source_ids=snapshot.demo_required_source_ids,
            execution_boundary="Evaluation should be run separately with provider/cost boundaries respected.",
        )
    )
    return tuple(actions)


def _slug(value: str) -> str:
    return "_".join(value.strip().lower().replace("/", " ").replace("-", " ").split())


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)
