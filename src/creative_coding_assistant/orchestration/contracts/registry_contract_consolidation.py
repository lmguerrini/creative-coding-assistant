"""V7.3 passive registry and contract consolidation contracts."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

REGISTRY_CONTRACT_CONSOLIDATION_SERIALIZATION_VERSION = (
    "registry_contract_consolidation.v1"
)
SOURCE_REGISTRY_INVENTORY_SERIALIZATION_VERSION = "source_registry_inventory.v1"
REGISTRY_COVERAGE_REPORT_SERIALIZATION_VERSION = "registry_coverage_report.v1"
PUBLIC_EXPORT_AUDIT_SERIALIZATION_VERSION = "public_export_audit.v1"
REGISTRY_INTEGRITY_VERIFICATION_SERIALIZATION_VERSION = (
    "registry_integrity_verification.v1"
)
CONTRACT_COMPATIBILITY_SERIALIZATION_VERSION = "contract_compatibility.v1"
SCHEMA_EVOLUTION_MANAGER_SERIALIZATION_VERSION = "schema_evolution_manager.v1"
REGISTRY_DEPENDENCY_GRAPH_SERIALIZATION_VERSION = "registry_dependency_graph.v1"
REGISTRY_DIFF_ENGINE_SERIALIZATION_VERSION = "registry_diff_engine.v1"
ARCHITECTURE_SIMPLIFICATION_REVIEW_SERIALIZATION_VERSION = (
    "architecture_simplification_review.v1"
)

REGISTRY_CONTRACT_CONSOLIDATION_ROADMAP_ITEMS = (
    "Registry Family Split",
    "Shared Registry Builders",
    "Shared Passive Boundary Base Models",
    "Source Registry Inventory Generator",
    "Registry Coverage Reports",
    "Contract Schema Normalization",
    "Import Surface Stabilization",
    "Public Export Audit",
    "Pydantic Review",
    "Jinja2 Review",
    "Style Review",
    "Code Style & Comment Quality Audit",
    "Logging Architecture Review",
    "Registry Package Consolidation",
    "Contract Simplification",
    "Metadata-to-Code Ratio Review",
    "Registry Integrity Verification",
    "Contract Compatibility Checker",
    "Schema Evolution Manager",
    "Contract Version Migration",
    "Registry Explainability",
    "Registry Dependency Graph",
    "Registry Diff Engine",
    "Architecture Simplification Review",
)
REGISTRY_CONTRACT_BLOCKED_RUNTIME_BEHAVIORS = (
    "provider_model_routing_change",
    "provider_execution",
    "workflow_execution",
    "workflow_graph_mutation",
    "prompt_rendering_change",
    "jinja_template_mutation",
    "logging_configuration_mutation",
    "persistent_storage_write",
    "generated_output_mutation",
    "schema_version_rewrite",
    "runtime_evolution_application",
    "hitl_decision_application",
)

RegistryFamilyId = Literal[
    "workflow_runtime",
    "failure_taxonomy",
    "agent_contracts",
    "artifact_contracts",
    "model_routing",
    "hybrid_studio",
    "cognitive_os",
    "knowledge_research",
    "production_governance",
]
ContractKind = Literal[
    "runtime_graph",
    "failure_contract",
    "agent_contract",
    "engine_contract",
    "routing_contract",
    "studio_contract",
    "surface_plan",
    "governance_audit",
]
ReviewStatus = Literal["pass", "advisory"]
DiffStatus = Literal["no_change", "changed"]


class PassiveBoundaryModel(BaseModel):
    """Shared base for passive V7.3 metadata-only contract records."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    advisory_only: Literal[True] = True
    provider_model_routing_change_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    prompt_rendering_change_implemented: Literal[False] = False
    jinja_template_mutation_implemented: Literal[False] = False
    logging_configuration_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False


class PassiveRegistryModel(PassiveBoundaryModel):
    """Shared shape for passive registry summaries built in V7.3."""

    role: str = Field(min_length=1, max_length=120)
    serialization_version: str = Field(min_length=1, max_length=120)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=32)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def _passive_registry_boundary_matches_v7(self) -> Self:
        if (
            self.blocked_runtime_behaviors
            != REGISTRY_CONTRACT_BLOCKED_RUNTIME_BEHAVIORS
        ):
            raise ValueError("blocked_runtime_behaviors must match V7.3 boundary")
        return self


class SharedRegistryBuilder(PassiveBoundaryModel):
    """Reusable builder contract for source registry inventory construction."""

    builder_id: str = Field(min_length=1, max_length=120)
    builder_role: Literal[
        "family_index",
        "inventory_generator",
        "coverage_report",
        "schema_normalizer",
        "integrity_verifier",
    ]
    output_contract: str = Field(min_length=1, max_length=120)
    stable_inputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    validation_guards: tuple[str, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _builder_id_matches_role(self) -> Self:
        if self.builder_id != f"registry_builder::{self.builder_role}":
            raise ValueError("builder_id must match builder_role")
        return self


class SharedPassiveBoundaryContract(PassiveBoundaryModel):
    """Shared passive boundary profile for registry and contract models."""

    boundary_id: str = Field(min_length=1, max_length=120)
    boundary_role: Literal[
        "base_model",
        "registry_model",
        "schema_record",
        "audit_record",
    ]
    enforced_flags: tuple[str, ...] = Field(min_length=1, max_length=16)
    compatible_model_families: tuple[RegistryFamilyId, ...] = Field(
        min_length=1,
        max_length=12,
    )

    @model_validator(mode="after")
    def _boundary_id_matches_role(self) -> Self:
        if self.boundary_id != f"passive_boundary::{self.boundary_role}":
            raise ValueError("boundary_id must match boundary_role")
        return self


class RegistrySourceRecord(PassiveBoundaryModel):
    """One existing source registry or contract surface in the consolidation map."""

    source_registry_id: str = Field(min_length=1, max_length=120)
    family_id: RegistryFamilyId
    module_path: str = Field(min_length=1, max_length=180)
    builder_name: str = Field(min_length=1, max_length=120)
    contract_kind: ContractKind
    serialization_version: str = Field(min_length=1, max_length=120)
    source_capability: str = Field(min_length=1, max_length=80)
    public_export_names: tuple[str, ...] = Field(min_length=1, max_length=8)
    pydantic_model_names: tuple[str, ...] = Field(min_length=1, max_length=8)
    dependency_registry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    normalized_schema_keys: tuple[str, ...] = Field(min_length=3, max_length=12)
    stable_import_surface: Literal[True] = True

    @model_validator(mode="after")
    def _source_id_matches_builder(self) -> Self:
        if not self.source_registry_id.startswith(f"{self.family_id}::"):
            raise ValueError("source_registry_id must be prefixed by family_id")
        if self.builder_name not in self.public_export_names:
            raise ValueError("builder_name must be a public export")
        return self


class RegistryFamily(PassiveBoundaryModel):
    """Family split record for a coherent group of registry contracts."""

    family_id: RegistryFamilyId
    family_name: str = Field(min_length=1, max_length=120)
    source_registry_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    primary_contract_kind: ContractKind
    consolidation_target: str = Field(min_length=1, max_length=220)
    split_rationale: str = Field(min_length=1, max_length=360)


class SourceRegistryInventory(PassiveRegistryModel):
    """Generated inventory of source registries used by V7.3 consolidation."""

    role: Literal["source_registry_inventory"] = "source_registry_inventory"
    serialization_version: Literal["source_registry_inventory.v1"] = (
        SOURCE_REGISTRY_INVENTORY_SERIALIZATION_VERSION
    )
    covered_roadmap_items: tuple[str, ...] = (
        "Registry Family Split",
        "Shared Registry Builders",
        "Shared Passive Boundary Base Models",
        "Source Registry Inventory Generator",
        "Registry Package Consolidation",
    )
    source_registries: tuple[RegistrySourceRecord, ...] = Field(
        min_length=1,
        max_length=40,
    )
    registry_families: tuple[RegistryFamily, ...] = Field(min_length=1, max_length=12)
    source_registry_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    family_ids: tuple[RegistryFamilyId, ...] = Field(min_length=1, max_length=12)
    source_registry_count: int = Field(ge=1)
    family_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _inventory_is_consistent(self) -> Self:
        source_ids = tuple(
            source.source_registry_id for source in self.source_registries
        )
        family_ids = tuple(family.family_id for family in self.registry_families)
        if self.source_registry_ids != source_ids:
            raise ValueError("source_registry_ids must match source_registries")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source_registry_ids must be unique")
        if self.family_ids != family_ids:
            raise ValueError("family_ids must match registry_families")
        if len(set(family_ids)) != len(family_ids):
            raise ValueError("family_ids must be unique")
        if self.source_registry_count != len(source_ids):
            raise ValueError("source_registry_count must match source_registries")
        if self.family_count != len(family_ids):
            raise ValueError("family_count must match registry_families")

        source_id_set = set(source_ids)
        family_id_set = set(family_ids)
        for source in self.source_registries:
            if source.family_id not in family_id_set:
                raise ValueError("source family_id must be known")
            unknown_dependencies = set(source.dependency_registry_ids).difference(
                source_id_set,
            )
            if unknown_dependencies:
                raise ValueError("dependency_registry_ids must be known")
        for family in self.registry_families:
            unknown_sources = set(family.source_registry_ids).difference(source_id_set)
            if unknown_sources:
                raise ValueError("family source_registry_ids must be known")
        return self


class ContractSchemaRecord(PassiveBoundaryModel):
    """Normalized schema summary for one source registry contract."""

    schema_id: str = Field(min_length=1, max_length=140)
    source_registry_id: str = Field(min_length=1, max_length=120)
    schema_version: str = Field(min_length=1, max_length=120)
    contract_kind: ContractKind
    normalized_required_fields: tuple[str, ...] = Field(min_length=3, max_length=14)
    compatibility_keys: tuple[str, ...] = Field(min_length=3, max_length=12)
    pydantic_model_names: tuple[str, ...] = Field(min_length=1, max_length=8)
    normalized_schema_hash: str = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def _schema_id_matches_source(self) -> Self:
        if self.schema_id != f"schema::{self.source_registry_id}":
            raise ValueError("schema_id must match source_registry_id")
        return self


class PublicExportAudit(PassiveRegistryModel):
    """Audit of stable import/export coverage for source registries."""

    role: Literal["public_export_audit"] = "public_export_audit"
    serialization_version: Literal["public_export_audit.v1"] = (
        PUBLIC_EXPORT_AUDIT_SERIALIZATION_VERSION
    )
    covered_roadmap_items: tuple[str, ...] = (
        "Import Surface Stabilization",
        "Public Export Audit",
    )
    checked_export_count: int = Field(ge=1)
    exported_names: tuple[str, ...] = Field(min_length=1, max_length=200)
    missing_export_names: tuple[str, ...] = Field(default_factory=tuple, max_length=40)
    duplicate_export_names: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=40,
    )
    public_exports_stable: bool

    @model_validator(mode="after")
    def _audit_status_matches_findings(self) -> Self:
        if self.public_exports_stable != (
            not self.missing_export_names and not self.duplicate_export_names
        ):
            raise ValueError("public_exports_stable must match export findings")
        return self


class RegistryCoverageReport(PassiveRegistryModel):
    """Coverage report tying V7.3 roadmap items to consolidation artifacts."""

    role: Literal["registry_coverage_report"] = "registry_coverage_report"
    serialization_version: Literal["registry_coverage_report.v1"] = (
        REGISTRY_COVERAGE_REPORT_SERIALIZATION_VERSION
    )
    covered_roadmap_items: tuple[str, ...] = Field(min_length=24, max_length=24)
    checked_source_registry_count: int = Field(ge=1)
    checked_family_count: int = Field(ge=1)
    checked_schema_count: int = Field(ge=1)
    missing_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    uncovered_source_registry_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_export_names: tuple[str, ...] = Field(default_factory=tuple)
    public_exports_stable: bool
    coverage_passed: bool

    @model_validator(mode="after")
    def _coverage_matches_v7_3(self) -> Self:
        if self.covered_roadmap_items != REGISTRY_CONTRACT_CONSOLIDATION_ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V7.3 roadmap")
        if self.coverage_passed != (
            not self.missing_roadmap_items
            and not self.uncovered_source_registry_ids
            and not self.missing_export_names
            and self.public_exports_stable
        ):
            raise ValueError("coverage_passed must match coverage findings")
        return self


class RegistryReviewFinding(PassiveBoundaryModel):
    """Single review finding for style, dependencies, logging, or schema posture."""

    finding_id: str = Field(min_length=1, max_length=140)
    roadmap_item: str = Field(min_length=1, max_length=120)
    status: ReviewStatus
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    recommendation: str = Field(min_length=1, max_length=360)

    @model_validator(mode="after")
    def _finding_id_matches_item(self) -> Self:
        expected = f"review::{_slug(self.roadmap_item)}"
        if self.finding_id != expected:
            raise ValueError("finding_id must match roadmap_item")
        return self


class RegistryIntegrityVerification(PassiveRegistryModel):
    """Integrity verification for registry ids, families, schemas, and exports."""

    role: Literal["registry_integrity_verification"] = "registry_integrity_verification"
    serialization_version: Literal["registry_integrity_verification.v1"] = (
        REGISTRY_INTEGRITY_VERIFICATION_SERIALIZATION_VERSION
    )
    covered_roadmap_items: tuple[str, ...] = ("Registry Integrity Verification",)
    checked_source_registry_count: int = Field(ge=1)
    checked_schema_count: int = Field(ge=1)
    duplicate_source_registry_ids: tuple[str, ...] = Field(default_factory=tuple)
    duplicate_schema_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_dependency_registry_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_schema_registry_ids: tuple[str, ...] = Field(default_factory=tuple)
    integrity_passed: bool

    @model_validator(mode="after")
    def _integrity_status_matches_findings(self) -> Self:
        if self.integrity_passed != (
            not self.duplicate_source_registry_ids
            and not self.duplicate_schema_ids
            and not self.missing_dependency_registry_ids
            and not self.missing_schema_registry_ids
        ):
            raise ValueError("integrity_passed must match integrity findings")
        return self


class ContractCompatibilityReport(PassiveRegistryModel):
    """Backward-compatibility report for normalized contract schemas."""

    role: Literal["contract_compatibility"] = "contract_compatibility"
    serialization_version: Literal["contract_compatibility.v1"] = (
        CONTRACT_COMPATIBILITY_SERIALIZATION_VERSION
    )
    covered_roadmap_items: tuple[str, ...] = ("Contract Compatibility Checker",)
    checked_schema_count: int = Field(ge=1)
    compatible_schema_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=40)
    incompatible_schema_ids: tuple[str, ...] = Field(default_factory=tuple)
    backward_compatibility_preserved: bool

    @model_validator(mode="after")
    def _compatibility_status_matches_findings(self) -> Self:
        if self.backward_compatibility_preserved != (not self.incompatible_schema_ids):
            raise ValueError(
                "backward_compatibility_preserved must match incompatible schemas",
            )
        if self.checked_schema_count != (
            len(self.compatible_schema_ids) + len(self.incompatible_schema_ids)
        ):
            raise ValueError("checked_schema_count must match schema ids")
        return self


class ContractVersionMigration(PassiveBoundaryModel):
    """Passive version migration descriptor for a normalized schema family."""

    migration_id: str = Field(min_length=1, max_length=140)
    source_schema_id: str = Field(min_length=1, max_length=140)
    from_version: str = Field(min_length=1, max_length=80)
    to_version: str = Field(min_length=1, max_length=80)
    compatibility_strategy: str = Field(min_length=1, max_length=220)
    breaking_change: Literal[False] = False
    migration_execution_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _migration_id_matches_schema(self) -> Self:
        if self.migration_id != f"migration::{self.source_schema_id}":
            raise ValueError("migration_id must match source_schema_id")
        return self


class SchemaEvolutionPlan(PassiveRegistryModel):
    """Schema evolution manager for current and next passive contract versions."""

    role: Literal["schema_evolution_manager"] = "schema_evolution_manager"
    serialization_version: Literal["schema_evolution_manager.v1"] = (
        SCHEMA_EVOLUTION_MANAGER_SERIALIZATION_VERSION
    )
    covered_roadmap_items: tuple[str, ...] = (
        "Schema Evolution Manager",
        "Contract Version Migration",
    )
    current_registry_version: Literal["registry_contract_consolidation.v1"] = (
        REGISTRY_CONTRACT_CONSOLIDATION_SERIALIZATION_VERSION
    )
    next_compatible_version: Literal["registry_contract_consolidation.v2"] = (
        "registry_contract_consolidation.v2"
    )
    migration_count: int = Field(ge=1)
    migration_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    compatibility_checker_version: Literal["contract_compatibility.v1"] = (
        CONTRACT_COMPATIBILITY_SERIALIZATION_VERSION
    )
    automatic_migration_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _migration_count_matches_ids(self) -> Self:
        if self.migration_count != len(self.migration_ids):
            raise ValueError("migration_count must match migration_ids")
        return self


class RegistryExplanation(PassiveBoundaryModel):
    """Explainability entry for one registry family."""

    explanation_id: str = Field(min_length=1, max_length=140)
    family_id: RegistryFamilyId
    source_registry_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    stable_export_names: tuple[str, ...] = Field(min_length=1, max_length=24)
    explanation: str = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def _explanation_id_matches_family(self) -> Self:
        if self.explanation_id != f"registry_explanation::{self.family_id}":
            raise ValueError("explanation_id must match family_id")
        return self


class RegistryDependencyNode(PassiveBoundaryModel):
    """Dependency graph node for one source registry."""

    node_id: str = Field(min_length=1, max_length=120)
    family_id: RegistryFamilyId
    contract_kind: ContractKind
    public_export_names: tuple[str, ...] = Field(min_length=1, max_length=8)


class RegistryDependencyEdge(PassiveBoundaryModel):
    """Dependency graph edge from one source registry to another."""

    edge_id: str = Field(min_length=1, max_length=180)
    source_registry_id: str = Field(min_length=1, max_length=120)
    target_registry_id: str = Field(min_length=1, max_length=120)
    dependency_reason: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def _edge_id_matches_source_target(self) -> Self:
        expected = (
            f"registry_dependency::{self.source_registry_id}->{self.target_registry_id}"
        )
        if self.edge_id != expected:
            raise ValueError("edge_id must match source and target")
        return self


class RegistryDependencyGraph(PassiveRegistryModel):
    """Passive dependency graph for registry consolidation planning."""

    role: Literal["registry_dependency_graph"] = "registry_dependency_graph"
    serialization_version: Literal["registry_dependency_graph.v1"] = (
        REGISTRY_DEPENDENCY_GRAPH_SERIALIZATION_VERSION
    )
    covered_roadmap_items: tuple[str, ...] = ("Registry Dependency Graph",)
    nodes: tuple[RegistryDependencyNode, ...] = Field(min_length=1, max_length=40)
    edges: tuple[RegistryDependencyEdge, ...] = Field(
        default_factory=tuple,
        max_length=80,
    )
    node_count: int = Field(ge=1)
    edge_count: int = Field(ge=0)
    dependency_cycles_detected: Literal[False] = False

    @model_validator(mode="after")
    def _graph_counts_match(self) -> Self:
        if self.node_count != len(self.nodes):
            raise ValueError("node_count must match nodes")
        if self.edge_count != len(self.edges):
            raise ValueError("edge_count must match edges")
        node_ids = {node.node_id for node in self.nodes}
        for edge in self.edges:
            if edge.source_registry_id not in node_ids:
                raise ValueError("edge source_registry_id must be known")
            if edge.target_registry_id not in node_ids:
                raise ValueError("edge target_registry_id must be known")
        return self


class RegistryDiffReport(PassiveRegistryModel):
    """Diff report between two registry inventories."""

    role: Literal["registry_diff_engine"] = "registry_diff_engine"
    serialization_version: Literal["registry_diff_engine.v1"] = (
        REGISTRY_DIFF_ENGINE_SERIALIZATION_VERSION
    )
    covered_roadmap_items: tuple[str, ...] = ("Registry Diff Engine",)
    diff_status: DiffStatus
    added_registry_ids: tuple[str, ...] = Field(default_factory=tuple)
    removed_registry_ids: tuple[str, ...] = Field(default_factory=tuple)
    changed_schema_ids: tuple[str, ...] = Field(default_factory=tuple)
    behavior_change_detected: Literal[False] = False

    @model_validator(mode="after")
    def _diff_status_matches_findings(self) -> Self:
        has_change = bool(
            self.added_registry_ids
            or self.removed_registry_ids
            or self.changed_schema_ids
        )
        if self.diff_status != ("changed" if has_change else "no_change"):
            raise ValueError("diff_status must match diff findings")
        return self


class ArchitectureSimplificationReview(PassiveRegistryModel):
    """Architecture simplification review for the V7.3 consolidation layer."""

    role: Literal["architecture_simplification_review"] = (
        "architecture_simplification_review"
    )
    serialization_version: Literal["architecture_simplification_review.v1"] = (
        ARCHITECTURE_SIMPLIFICATION_REVIEW_SERIALIZATION_VERSION
    )
    covered_roadmap_items: tuple[str, ...] = ("Architecture Simplification Review",)
    reviewed_families: tuple[RegistryFamilyId, ...] = Field(min_length=1, max_length=12)
    simplification_decisions: tuple[str, ...] = Field(min_length=1, max_length=12)
    deferred_refactor_surfaces: tuple[str, ...] = Field(default_factory=tuple)
    long_term_system_simpler: bool

    @model_validator(mode="after")
    def _simplification_status_matches_review(self) -> Self:
        if self.long_term_system_simpler != (not self.deferred_refactor_surfaces):
            raise ValueError("long_term_system_simpler must match deferred refactors")
        return self


class RegistryContractConsolidationPlan(PassiveRegistryModel):
    """Aggregate V7.3 registry and contract consolidation plan."""

    role: Literal["registry_contract_consolidation"] = "registry_contract_consolidation"
    serialization_version: Literal["registry_contract_consolidation.v1"] = (
        REGISTRY_CONTRACT_CONSOLIDATION_SERIALIZATION_VERSION
    )
    covered_roadmap_items: tuple[str, ...] = Field(min_length=24, max_length=24)
    shared_builders: tuple[SharedRegistryBuilder, ...] = Field(
        min_length=5,
        max_length=5,
    )
    shared_passive_boundaries: tuple[SharedPassiveBoundaryContract, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_inventory: SourceRegistryInventory
    coverage_report: RegistryCoverageReport
    schema_records: tuple[ContractSchemaRecord, ...] = Field(
        min_length=1,
        max_length=40,
    )
    public_export_audit: PublicExportAudit
    review_findings: tuple[RegistryReviewFinding, ...] = Field(
        min_length=8,
        max_length=8,
    )
    integrity_report: RegistryIntegrityVerification
    compatibility_report: ContractCompatibilityReport
    schema_evolution_plan: SchemaEvolutionPlan
    version_migrations: tuple[ContractVersionMigration, ...] = Field(
        min_length=1,
        max_length=40,
    )
    explanations: tuple[RegistryExplanation, ...] = Field(min_length=1, max_length=12)
    dependency_graph: RegistryDependencyGraph
    diff_report: RegistryDiffReport
    architecture_simplification_review: ArchitectureSimplificationReview
    source_registry_count: int = Field(ge=1)
    family_count: int = Field(ge=1)
    schema_count: int = Field(ge=1)
    roadmap_item_count: int = Field(ge=24, le=24)

    @model_validator(mode="after")
    def _plan_is_consistent(self) -> Self:
        if self.covered_roadmap_items != REGISTRY_CONTRACT_CONSOLIDATION_ROADMAP_ITEMS:
            raise ValueError("covered_roadmap_items must match V7.3 roadmap")
        if self.source_registry_count != self.source_inventory.source_registry_count:
            raise ValueError("source_registry_count must match source inventory")
        if self.family_count != self.source_inventory.family_count:
            raise ValueError("family_count must match source inventory")
        if self.schema_count != len(self.schema_records):
            raise ValueError("schema_count must match schema_records")
        if self.roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("roadmap_item_count must match covered roadmap")
        if self.coverage_report.covered_roadmap_items != self.covered_roadmap_items:
            raise ValueError("coverage report must match plan roadmap")
        if self.schema_evolution_plan.migration_ids != tuple(
            migration.migration_id for migration in self.version_migrations
        ):
            raise ValueError("schema evolution plan must reference migrations")
        if any(
            finding.roadmap_item not in self.covered_roadmap_items
            for finding in self.review_findings
        ):
            raise ValueError("review findings must reference V7.3 roadmap items")
        if not all(_is_passive_model(item) for item in self._passive_children()):
            raise ValueError("all V7.3 consolidation records must remain passive")
        return self

    def _passive_children(self) -> tuple[PassiveBoundaryModel, ...]:
        return (
            *self.shared_builders,
            *self.shared_passive_boundaries,
            *self.source_inventory.source_registries,
            *self.source_inventory.registry_families,
            *self.schema_records,
            self.public_export_audit,
            *self.review_findings,
            self.integrity_report,
            self.compatibility_report,
            self.schema_evolution_plan,
            *self.version_migrations,
            *self.explanations,
            self.dependency_graph,
            self.diff_report,
            self.architecture_simplification_review,
        )


def build_registry_contract_consolidation_plan() -> RegistryContractConsolidationPlan:
    """Build the passive V7.3 registry consolidation plan."""

    source_records = _source_registry_records()
    family_records = _registry_families(source_records)
    inventory = generate_source_registry_inventory(source_records, family_records)
    schema_records = normalize_contract_schemas(inventory)
    public_export_audit = audit_registry_public_exports(inventory)
    coverage_report = build_registry_coverage_report(
        inventory=inventory,
        schema_records=schema_records,
        public_export_audit=public_export_audit,
    )
    integrity_report = verify_registry_integrity(inventory, schema_records)
    compatibility_report = check_contract_compatibility(schema_records)
    migrations = build_contract_version_migrations(schema_records)
    schema_evolution_plan = build_schema_evolution_plan(migrations)
    dependency_graph = build_registry_dependency_graph(inventory)
    return RegistryContractConsolidationPlan(
        covered_roadmap_items=REGISTRY_CONTRACT_CONSOLIDATION_ROADMAP_ITEMS,
        shared_builders=_shared_registry_builders(),
        shared_passive_boundaries=_shared_passive_boundaries(),
        source_inventory=inventory,
        coverage_report=coverage_report,
        schema_records=schema_records,
        public_export_audit=public_export_audit,
        review_findings=_review_findings(),
        integrity_report=integrity_report,
        compatibility_report=compatibility_report,
        schema_evolution_plan=schema_evolution_plan,
        version_migrations=migrations,
        explanations=explain_registry_families(inventory),
        dependency_graph=dependency_graph,
        diff_report=diff_registry_inventories(inventory, inventory),
        architecture_simplification_review=review_architecture_simplification(
            inventory,
        ),
        blocked_runtime_behaviors=REGISTRY_CONTRACT_BLOCKED_RUNTIME_BEHAVIORS,
        source_registry_count=inventory.source_registry_count,
        family_count=inventory.family_count,
        schema_count=len(schema_records),
        roadmap_item_count=len(REGISTRY_CONTRACT_CONSOLIDATION_ROADMAP_ITEMS),
    )


def generate_source_registry_inventory(
    source_records: tuple[RegistrySourceRecord, ...] | None = None,
    family_records: tuple[RegistryFamily, ...] | None = None,
) -> SourceRegistryInventory:
    """Generate the source registry inventory without importing providers."""

    sources = source_records or _source_registry_records()
    families = family_records or _registry_families(sources)
    return SourceRegistryInventory(
        source_registries=sources,
        registry_families=families,
        source_registry_ids=tuple(source.source_registry_id for source in sources),
        family_ids=tuple(family.family_id for family in families),
        source_registry_count=len(sources),
        family_count=len(families),
        blocked_runtime_behaviors=REGISTRY_CONTRACT_BLOCKED_RUNTIME_BEHAVIORS,
    )


def build_registry_coverage_report(
    *,
    inventory: SourceRegistryInventory | None = None,
    schema_records: tuple[ContractSchemaRecord, ...] | None = None,
    public_export_audit: PublicExportAudit | None = None,
) -> RegistryCoverageReport:
    """Report roadmap and source coverage for the V7.3 consolidation."""

    source_inventory = inventory or generate_source_registry_inventory()
    schemas = schema_records or normalize_contract_schemas(source_inventory)
    export_audit = public_export_audit or audit_registry_public_exports(
        source_inventory,
    )
    schema_source_ids = {schema.source_registry_id for schema in schemas}
    uncovered_sources = tuple(
        source_id
        for source_id in source_inventory.source_registry_ids
        if source_id not in schema_source_ids
    )
    missing_roadmap = tuple(
        item
        for item in REGISTRY_CONTRACT_CONSOLIDATION_ROADMAP_ITEMS
        if item not in _artifact_covered_roadmap_items()
    )
    return RegistryCoverageReport(
        covered_roadmap_items=REGISTRY_CONTRACT_CONSOLIDATION_ROADMAP_ITEMS,
        checked_source_registry_count=source_inventory.source_registry_count,
        checked_family_count=source_inventory.family_count,
        checked_schema_count=len(schemas),
        missing_roadmap_items=missing_roadmap,
        uncovered_source_registry_ids=uncovered_sources,
        missing_export_names=export_audit.missing_export_names,
        public_exports_stable=export_audit.public_exports_stable,
        coverage_passed=(
            not missing_roadmap
            and not uncovered_sources
            and not export_audit.missing_export_names
            and export_audit.public_exports_stable
        ),
        blocked_runtime_behaviors=REGISTRY_CONTRACT_BLOCKED_RUNTIME_BEHAVIORS,
    )


def normalize_contract_schemas(
    inventory: SourceRegistryInventory | None = None,
) -> tuple[ContractSchemaRecord, ...]:
    """Normalize source registry contracts into stable schema records."""

    source_inventory = inventory or generate_source_registry_inventory()
    return tuple(
        ContractSchemaRecord(
            schema_id=f"schema::{source.source_registry_id}",
            source_registry_id=source.source_registry_id,
            schema_version=source.serialization_version,
            contract_kind=source.contract_kind,
            normalized_required_fields=source.normalized_schema_keys,
            compatibility_keys=(
                "role",
                "serialization_version",
                "authority_boundary",
                "advisory_only",
            ),
            pydantic_model_names=source.pydantic_model_names,
            normalized_schema_hash=_stable_schema_hash(source),
        )
        for source in source_inventory.source_registries
    )


def audit_registry_public_exports(
    inventory: SourceRegistryInventory | None = None,
) -> PublicExportAudit:
    """Audit source registry exports against the package public surface."""

    source_inventory = inventory or generate_source_registry_inventory()
    raw_exports = tuple(
        export
        for source in source_inventory.source_registries
        for export in source.public_export_names
    )
    required_exports = tuple(
        dict.fromkeys(raw_exports),
    )
    package_exports = _public_export_names()
    missing_exports = tuple(
        export for export in required_exports if export not in package_exports
    )
    duplicate_exports = tuple(
        name for name, count in Counter(raw_exports).items() if count > 1
    )
    return PublicExportAudit(
        checked_export_count=len(required_exports),
        exported_names=required_exports,
        missing_export_names=missing_exports,
        duplicate_export_names=duplicate_exports,
        public_exports_stable=not missing_exports and not duplicate_exports,
        blocked_runtime_behaviors=REGISTRY_CONTRACT_BLOCKED_RUNTIME_BEHAVIORS,
    )


def verify_registry_integrity(
    inventory: SourceRegistryInventory | None = None,
    schema_records: tuple[ContractSchemaRecord, ...] | None = None,
) -> RegistryIntegrityVerification:
    """Verify registry id, dependency, and schema integrity."""

    source_inventory = inventory or generate_source_registry_inventory()
    schemas = schema_records or normalize_contract_schemas(source_inventory)
    source_ids = tuple(source_inventory.source_registry_ids)
    schema_ids = tuple(schema.schema_id for schema in schemas)
    schema_source_ids = {schema.source_registry_id for schema in schemas}
    missing_dependencies = tuple(
        dependency_id
        for source in source_inventory.source_registries
        for dependency_id in source.dependency_registry_ids
        if dependency_id not in source_ids
    )
    missing_schemas = tuple(
        source_id for source_id in source_ids if source_id not in schema_source_ids
    )
    duplicate_sources = _duplicates(source_ids)
    duplicate_schemas = _duplicates(schema_ids)
    return RegistryIntegrityVerification(
        checked_source_registry_count=len(source_ids),
        checked_schema_count=len(schemas),
        duplicate_source_registry_ids=duplicate_sources,
        duplicate_schema_ids=duplicate_schemas,
        missing_dependency_registry_ids=missing_dependencies,
        missing_schema_registry_ids=missing_schemas,
        integrity_passed=not (
            duplicate_sources
            or duplicate_schemas
            or missing_dependencies
            or missing_schemas
        ),
        blocked_runtime_behaviors=REGISTRY_CONTRACT_BLOCKED_RUNTIME_BEHAVIORS,
    )


def check_contract_compatibility(
    schema_records: tuple[ContractSchemaRecord, ...] | None = None,
) -> ContractCompatibilityReport:
    """Check normalized schemas for backward-compatible required keys."""

    schemas = schema_records or normalize_contract_schemas()
    required_keys = {"role", "serialization_version", "authority_boundary"}
    incompatible_schema_ids = tuple(
        schema.schema_id
        for schema in schemas
        if not required_keys.issubset(schema.normalized_required_fields)
    )
    compatible_schema_ids = tuple(
        schema.schema_id
        for schema in schemas
        if schema.schema_id not in incompatible_schema_ids
    )
    return ContractCompatibilityReport(
        checked_schema_count=len(schemas),
        compatible_schema_ids=compatible_schema_ids,
        incompatible_schema_ids=incompatible_schema_ids,
        backward_compatibility_preserved=not incompatible_schema_ids,
        blocked_runtime_behaviors=REGISTRY_CONTRACT_BLOCKED_RUNTIME_BEHAVIORS,
    )


def build_contract_version_migrations(
    schema_records: tuple[ContractSchemaRecord, ...] | None = None,
) -> tuple[ContractVersionMigration, ...]:
    """Build passive migration descriptors without rewriting schemas."""

    schemas = schema_records or normalize_contract_schemas()
    return tuple(
        ContractVersionMigration(
            migration_id=f"migration::{schema.schema_id}",
            source_schema_id=schema.schema_id,
            from_version=schema.schema_version,
            to_version=_next_schema_version(schema.schema_version),
            compatibility_strategy=(
                "Additive schema evolution only: retain role, serialization "
                "version, authority boundary, and advisory-only flags."
            ),
        )
        for schema in schemas
    )


def build_schema_evolution_plan(
    migrations: tuple[ContractVersionMigration, ...] | None = None,
) -> SchemaEvolutionPlan:
    """Build the passive schema evolution manager plan."""

    version_migrations = migrations or build_contract_version_migrations()
    return SchemaEvolutionPlan(
        migration_count=len(version_migrations),
        migration_ids=tuple(migration.migration_id for migration in version_migrations),
        blocked_runtime_behaviors=REGISTRY_CONTRACT_BLOCKED_RUNTIME_BEHAVIORS,
    )


def explain_registry_families(
    inventory: SourceRegistryInventory | None = None,
) -> tuple[RegistryExplanation, ...]:
    """Return explanation records for each source registry family."""

    source_inventory = inventory or generate_source_registry_inventory()
    sources_by_id = {
        source.source_registry_id: source
        for source in source_inventory.source_registries
    }
    explanations: list[RegistryExplanation] = []
    for family in source_inventory.registry_families:
        family_sources = tuple(
            sources_by_id[source_id] for source_id in family.source_registry_ids
        )
        explanations.append(
            RegistryExplanation(
                explanation_id=f"registry_explanation::{family.family_id}",
                family_id=family.family_id,
                source_registry_ids=family.source_registry_ids,
                stable_export_names=tuple(
                    dict.fromkeys(
                        export
                        for source in family_sources
                        for export in source.public_export_names
                    ),
                ),
                explanation=(
                    f"{family.family_name} consolidates "
                    f"{len(family.source_registry_ids)} passive source "
                    "registries behind stable schema and export metadata."
                ),
            )
        )
    return tuple(explanations)


def build_registry_dependency_graph(
    inventory: SourceRegistryInventory | None = None,
) -> RegistryDependencyGraph:
    """Build a passive dependency graph for source registries."""

    source_inventory = inventory or generate_source_registry_inventory()
    nodes = tuple(
        RegistryDependencyNode(
            node_id=source.source_registry_id,
            family_id=source.family_id,
            contract_kind=source.contract_kind,
            public_export_names=source.public_export_names,
        )
        for source in source_inventory.source_registries
    )
    edges = tuple(
        RegistryDependencyEdge(
            edge_id=(
                f"registry_dependency::{source.source_registry_id}->{dependency_id}"
            ),
            source_registry_id=source.source_registry_id,
            target_registry_id=dependency_id,
            dependency_reason="Source registry references upstream passive metadata.",
        )
        for source in source_inventory.source_registries
        for dependency_id in source.dependency_registry_ids
    )
    return RegistryDependencyGraph(
        nodes=nodes,
        edges=edges,
        node_count=len(nodes),
        edge_count=len(edges),
        blocked_runtime_behaviors=REGISTRY_CONTRACT_BLOCKED_RUNTIME_BEHAVIORS,
    )


def diff_registry_inventories(
    before: SourceRegistryInventory,
    after: SourceRegistryInventory,
) -> RegistryDiffReport:
    """Diff two source registry inventories without changing either inventory."""

    before_ids = set(before.source_registry_ids)
    after_ids = set(after.source_registry_ids)
    before_schemas = {
        schema.schema_id: schema.normalized_schema_hash
        for schema in normalize_contract_schemas(before)
    }
    after_schemas = {
        schema.schema_id: schema.normalized_schema_hash
        for schema in normalize_contract_schemas(after)
    }
    changed_schema_ids = tuple(
        schema_id
        for schema_id, schema_hash in after_schemas.items()
        if before_schemas.get(schema_id) not in {None, schema_hash}
    )
    added = tuple(sorted(after_ids.difference(before_ids)))
    removed = tuple(sorted(before_ids.difference(after_ids)))
    changed = tuple(sorted(changed_schema_ids))
    return RegistryDiffReport(
        diff_status="changed" if added or removed or changed else "no_change",
        added_registry_ids=added,
        removed_registry_ids=removed,
        changed_schema_ids=changed,
        blocked_runtime_behaviors=REGISTRY_CONTRACT_BLOCKED_RUNTIME_BEHAVIORS,
    )


def review_architecture_simplification(
    inventory: SourceRegistryInventory | None = None,
) -> ArchitectureSimplificationReview:
    """Review whether the consolidation layer reduces long-term complexity."""

    source_inventory = inventory or generate_source_registry_inventory()
    return ArchitectureSimplificationReview(
        reviewed_families=source_inventory.family_ids,
        simplification_decisions=(
            "Use one passive base boundary for V7.3 consolidation records.",
            "Keep source registries in their current modules to avoid churn.",
            "Represent families, schemas, exports, integrity, compatibility, "
            "diffs, and dependencies as read-only metadata.",
            "Centralize future schema migration posture in one manager record.",
        ),
        deferred_refactor_surfaces=(),
        long_term_system_simpler=True,
        blocked_runtime_behaviors=REGISTRY_CONTRACT_BLOCKED_RUNTIME_BEHAVIORS,
    )


def registry_source_by_id(
    source_registry_id: str,
    inventory: SourceRegistryInventory | None = None,
) -> RegistrySourceRecord | None:
    """Return one source registry inventory record by stable id."""

    source_inventory = inventory or generate_source_registry_inventory()
    for source in source_inventory.source_registries:
        if source.source_registry_id == source_registry_id:
            return source
    return None


def registry_family_by_id(
    family_id: RegistryFamilyId,
    inventory: SourceRegistryInventory | None = None,
) -> RegistryFamily | None:
    """Return one registry family split record."""

    source_inventory = inventory or generate_source_registry_inventory()
    for family in source_inventory.registry_families:
        if family.family_id == family_id:
            return family
    return None


def registry_sources_for_family(
    family_id: RegistryFamilyId,
    inventory: SourceRegistryInventory | None = None,
) -> tuple[RegistrySourceRecord, ...]:
    """Return all source registry records for one family."""

    source_inventory = inventory or generate_source_registry_inventory()
    return tuple(
        source
        for source in source_inventory.source_registries
        if source.family_id == family_id
    )


def explain_registry_source(
    source_registry_id: str,
    plan: RegistryContractConsolidationPlan | None = None,
) -> str | None:
    """Return a stable explanation for one source registry."""

    source_plan = plan or build_registry_contract_consolidation_plan()
    source = registry_source_by_id(source_registry_id, source_plan.source_inventory)
    if source is None:
        return None
    family = registry_family_by_id(source.family_id, source_plan.source_inventory)
    family_name = source.family_id if family is None else family.family_name
    return (
        f"{source.source_registry_id} belongs to {family_name}, exposes "
        f"{', '.join(source.public_export_names)}, and remains advisory-only."
    )


def _shared_registry_builders() -> tuple[SharedRegistryBuilder, ...]:
    specs = (
        ("family_index", "SourceRegistryInventory"),
        ("inventory_generator", "SourceRegistryInventory"),
        ("coverage_report", "RegistryCoverageReport"),
        ("schema_normalizer", "ContractSchemaRecord"),
        ("integrity_verifier", "RegistryIntegrityVerification"),
    )
    return tuple(
        SharedRegistryBuilder(
            builder_id=f"registry_builder::{role}",
            builder_role=role,
            output_contract=output_contract,
            stable_inputs=("source_registry_records", "registry_families"),
            validation_guards=(
                "unique_ids",
                "known_dependencies",
                "stable_public_exports",
                "passive_boundary_flags",
            ),
        )
        for role, output_contract in specs
    )


def _shared_passive_boundaries() -> tuple[SharedPassiveBoundaryContract, ...]:
    families: tuple[RegistryFamilyId, ...] = (
        "workflow_runtime",
        "failure_taxonomy",
        "agent_contracts",
        "artifact_contracts",
        "model_routing",
        "hybrid_studio",
        "cognitive_os",
        "knowledge_research",
        "production_governance",
    )
    return tuple(
        SharedPassiveBoundaryContract(
            boundary_id=f"passive_boundary::{role}",
            boundary_role=role,
            enforced_flags=(
                "advisory_only",
                "provider_execution_implemented",
                "workflow_execution_implemented",
                "persistent_storage_write_implemented",
                "runtime_evolution_implemented",
            ),
            compatible_model_families=families,
        )
        for role in ("base_model", "registry_model", "schema_record", "audit_record")
    )


def _source_registry_records() -> tuple[RegistrySourceRecord, ...]:
    specs: tuple[dict[str, object], ...] = (
        _source_spec(
            source_registry_id="workflow_runtime::runtime_graph_consolidation",
            family_id="workflow_runtime",
            module_path="creative_coding_assistant.orchestration.runtime_graph_consolidation",
            builder_name="build_runtime_graph_consolidation_plan",
            contract_kind="runtime_graph",
            serialization_version="runtime_graph_consolidation.v1",
            source_capability="V7.1",
            public_export_names=(
                "RuntimeGraphConsolidationPlan",
                "build_runtime_graph_consolidation_plan",
                "validate_runtime_graph_contracts",
            ),
            pydantic_model_names=(
                "RuntimeGraphConsolidationPlan",
                "RuntimeGraphNodeContract",
                "RuntimeGraphSubgraphContract",
            ),
        ),
        _source_spec(
            source_registry_id="failure_taxonomy::typed_failure_taxonomy",
            family_id="failure_taxonomy",
            module_path="creative_coding_assistant.orchestration.typed_failure_taxonomy",
            builder_name="build_typed_failure_taxonomy_registry",
            contract_kind="failure_contract",
            serialization_version="typed_failure_taxonomy.v1",
            source_capability="V7.2",
            public_export_names=(
                "TypedFailureTaxonomyRegistry",
                "build_typed_failure_taxonomy_registry",
                "validate_typed_failure_taxonomy",
            ),
            pydantic_model_names=(
                "TypedFailureTaxonomyRegistry",
                "FailureTypeDefinition",
                "FailureEventContract",
            ),
            dependency_registry_ids=("workflow_runtime::runtime_graph_consolidation",),
        ),
        _source_spec(
            source_registry_id="agent_contracts::agent_contract_registry",
            family_id="agent_contracts",
            module_path="creative_coding_assistant.orchestration.agent_contracts",
            builder_name="agent_contract_registry",
            contract_kind="agent_contract",
            serialization_version="agent_contract_registry.v1",
            source_capability="V4.1",
            public_export_names=(
                "AgentContractRegistry",
                "agent_contract_registry",
                "build_agent_contract_registry",
            ),
            pydantic_model_names=("AgentContractRegistry", "AgentContract"),
        ),
        _source_spec(
            source_registry_id="agent_contracts::agent_memory_contract_registry",
            family_id="agent_contracts",
            module_path="creative_coding_assistant.orchestration.agent_memory_contracts",
            builder_name="agent_memory_contract_registry",
            contract_kind="agent_contract",
            serialization_version="agent_memory_contract_registry.v1",
            source_capability="V4.6",
            public_export_names=(
                "AgentMemoryContractRegistry",
                "agent_memory_contract_registry",
            ),
            pydantic_model_names=(
                "AgentMemoryContractRegistry",
                "AgentMemoryContract",
            ),
            dependency_registry_ids=("agent_contracts::agent_contract_registry",),
        ),
        _source_spec(
            source_registry_id="agent_contracts::agent_dependency_graph_registry",
            family_id="agent_contracts",
            module_path="creative_coding_assistant.orchestration.agent_dependency_graph",
            builder_name="agent_dependency_graph_registry",
            contract_kind="agent_contract",
            serialization_version="agent_dependency_graph_registry.v1",
            source_capability="V4.3",
            public_export_names=(
                "AgentDependencyGraphRegistry",
                "agent_dependency_graph_registry",
            ),
            pydantic_model_names=(
                "AgentDependencyGraphRegistry",
                "AgentDependencyNode",
                "AgentDependencyEdge",
            ),
            dependency_registry_ids=("agent_contracts::agent_contract_registry",),
        ),
        _source_spec(
            source_registry_id="artifact_contracts::artifact_engine_contract_registry",
            family_id="artifact_contracts",
            module_path="creative_coding_assistant.orchestration.artifact_engine_contracts",
            builder_name="artifact_intelligence_engine_contracts",
            contract_kind="engine_contract",
            serialization_version="artifact_engine_contract_registry.v1",
            source_capability="V3.3",
            public_export_names=(
                "ArtifactIntelligenceEngineContractRegistry",
                "artifact_intelligence_engine_contracts",
            ),
            pydantic_model_names=(
                "ArtifactIntelligenceEngineContractRegistry",
                "ArtifactIntelligenceEngineContract",
            ),
        ),
        _source_spec(
            source_registry_id="artifact_contracts::evaluation_engine_contract_registry",
            family_id="artifact_contracts",
            module_path="creative_coding_assistant.orchestration.evaluation_engine_contracts",
            builder_name="evaluation_engine_contract_by_id",
            contract_kind="engine_contract",
            serialization_version="evaluation_engine_contract_registry.v1",
            source_capability="V3.4",
            public_export_names=(
                "EvaluationEngineContractRegistry",
                "evaluation_engine_contract_by_id",
            ),
            pydantic_model_names=(
                "EvaluationEngineContractRegistry",
                "EvaluationEngineContract",
            ),
        ),
        _source_spec(
            source_registry_id="artifact_contracts::workstation_engine_contract_registry",
            family_id="artifact_contracts",
            module_path="creative_coding_assistant.orchestration.workstation_contracts",
            builder_name="workstation_engine_contracts",
            contract_kind="engine_contract",
            serialization_version="workstation_engine_contract_registry.v1",
            source_capability="V3.5",
            public_export_names=(
                "WorkstationEngineContractRegistry",
                "workstation_engine_contracts",
            ),
            pydantic_model_names=(
                "WorkstationEngineContractRegistry",
                "WorkstationEngineContract",
            ),
        ),
        _source_spec(
            source_registry_id="model_routing::model_routing_intelligence_registry",
            family_id="model_routing",
            module_path="creative_coding_assistant.orchestration.routing_intelligence",
            builder_name="model_routing_intelligence_registry",
            contract_kind="routing_contract",
            serialization_version="model_routing_intelligence_registry.v1",
            source_capability="V5.2",
            public_export_names=(
                "ModelRoutingIntelligenceRegistry",
                "model_routing_intelligence_registry",
            ),
            pydantic_model_names=("ModelRoutingIntelligenceRegistry",),
            dependency_registry_ids=(
                "hybrid_studio::model_profile_registry",
                "hybrid_studio::provider_selection_registry",
            ),
        ),
        _source_spec(
            source_registry_id="hybrid_studio::model_profile_registry",
            family_id="hybrid_studio",
            module_path="creative_coding_assistant.orchestration.hybrid_studio",
            builder_name="model_profile_registry",
            contract_kind="studio_contract",
            serialization_version="model_profile_registry.v1",
            source_capability="V4.4",
            public_export_names=("ModelProfileRegistry", "model_profile_registry"),
            pydantic_model_names=("ModelProfileRegistry", "ModelProfile"),
        ),
        _source_spec(
            source_registry_id="hybrid_studio::provider_selection_registry",
            family_id="hybrid_studio",
            module_path="creative_coding_assistant.orchestration.hybrid_studio",
            builder_name="provider_selection_registry",
            contract_kind="studio_contract",
            serialization_version="provider_selection_registry.v1",
            source_capability="V4.4",
            public_export_names=(
                "ProviderSelectionRegistry",
                "provider_selection_registry",
            ),
            pydantic_model_names=(
                "ProviderSelectionRegistry",
                "ProviderSelectionProfile",
            ),
            dependency_registry_ids=("hybrid_studio::model_profile_registry",),
        ),
        _source_spec(
            source_registry_id="cognitive_os::cognitive_os_core_surface",
            family_id="cognitive_os",
            module_path="creative_coding_assistant.orchestration.cognitive_os_core_surface",
            builder_name="build_cognitive_os_core_surface",
            contract_kind="surface_plan",
            serialization_version="cognitive_os_core_surface.v1",
            source_capability="V6.6",
            public_export_names=(
                "CognitiveOSCoreSurfacePlan",
                "build_cognitive_os_core_surface",
            ),
            pydantic_model_names=(
                "CognitiveOSCoreSurfacePlan",
                "CognitiveOSCoreSurfaceEntry",
            ),
            dependency_registry_ids=("agent_contracts::agent_contract_registry",),
        ),
        _source_spec(
            source_registry_id="knowledge_research::knowledge_evolution_core_surface",
            family_id="knowledge_research",
            module_path="creative_coding_assistant.orchestration.knowledge_evolution_core_surface",
            builder_name="build_knowledge_evolution_core_surface",
            contract_kind="surface_plan",
            serialization_version="knowledge_evolution_core_surface.v1",
            source_capability="V6.3",
            public_export_names=(
                "KnowledgeEvolutionCoreSurfacePlan",
                "build_knowledge_evolution_core_surface",
            ),
            pydantic_model_names=(
                "KnowledgeEvolutionCoreSurfacePlan",
                "KnowledgeEvolutionCoreSurfaceEntry",
            ),
        ),
        _source_spec(
            source_registry_id="knowledge_research::research_core_surface",
            family_id="knowledge_research",
            module_path="creative_coding_assistant.orchestration.research_core_surface",
            builder_name="build_research_core_surface",
            contract_kind="surface_plan",
            serialization_version="research_core_surface_plan.v1",
            source_capability="V6.4",
            public_export_names=(
                "ResearchCoreSurfacePlan",
                "build_research_core_surface",
            ),
            pydantic_model_names=(
                "ResearchCoreSurfacePlan",
                "ResearchCoreSurfaceEntry",
            ),
            dependency_registry_ids=(
                "knowledge_research::knowledge_evolution_core_surface",
            ),
        ),
        _source_spec(
            source_registry_id="production_governance::production_architecture_consistency",
            family_id="production_governance",
            module_path="creative_coding_assistant.orchestration.production_architecture_consistency",
            builder_name="production_architecture_consistency_registry",
            contract_kind="governance_audit",
            serialization_version="production_architecture_consistency_registry.v1",
            source_capability="V5.6",
            public_export_names=(
                "ProductionArchitectureConsistencyRegistry",
                "production_architecture_consistency_registry",
            ),
            pydantic_model_names=(
                "ProductionArchitectureConsistencyRegistry",
                "ProductionArchitectureConsistencyRecord",
            ),
        ),
    )
    return tuple(RegistrySourceRecord(**spec) for spec in specs)


def _registry_families(
    source_records: tuple[RegistrySourceRecord, ...],
) -> tuple[RegistryFamily, ...]:
    families: tuple[tuple[RegistryFamilyId, str, ContractKind, str, str], ...] = (
        (
            "workflow_runtime",
            "Workflow Runtime Registries",
            "runtime_graph",
            "Keep runtime graph contracts separate from failure and routing metadata.",
            "Workflow topology owns graph shape and static execution contracts.",
        ),
        (
            "failure_taxonomy",
            "Failure Taxonomy Registries",
            "failure_contract",
            "Keep typed failures isolated from runtime execution and recovery.",
            "Failure contracts are advisory lookup metadata only.",
        ),
        (
            "agent_contracts",
            "Agent Contract Registries",
            "agent_contract",
            "Group agent identity, memory, and dependency contracts together.",
            "Agent registries share passive boundary and cross-agent references.",
        ),
        (
            "artifact_contracts",
            "Artifact and Evaluation Contract Registries",
            "engine_contract",
            "Group engine/workstation contracts by shared schema shape.",
            "Artifact, evaluation, and workstation contracts share stable metadata.",
        ),
        (
            "model_routing",
            "Model Routing Registries",
            "routing_contract",
            "Keep advisory routing contracts separated from provider execution.",
            "Routing metadata must not mutate provider or model selection.",
        ),
        (
            "hybrid_studio",
            "Hybrid Studio Registries",
            "studio_contract",
            "Group local/cloud and provider-selection studio profiles together.",
            "Hybrid Studio profiles are source metadata for advisory routing.",
        ),
        (
            "cognitive_os",
            "Cognitive OS Registries",
            "surface_plan",
            "Keep OS surface contracts separate from runtime graph execution.",
            "Cognitive surfaces aggregate passive agent and workflow metadata.",
        ),
        (
            "knowledge_research",
            "Knowledge and Research Registries",
            "surface_plan",
            "Group knowledge and research surfaces as read-only source metadata.",
            "Research depends on knowledge posture without executing research.",
        ),
        (
            "production_governance",
            "Production Governance Registries",
            "governance_audit",
            "Keep release and architecture governance in audit registries.",
            "Governance metadata should not apply freezes or release operations.",
        ),
    )
    return tuple(
        RegistryFamily(
            family_id=family_id,
            family_name=family_name,
            source_registry_ids=tuple(
                source.source_registry_id
                for source in source_records
                if source.family_id == family_id
            ),
            primary_contract_kind=contract_kind,
            consolidation_target=target,
            split_rationale=rationale,
        )
        for family_id, family_name, contract_kind, target, rationale in families
    )


def _source_spec(
    *,
    source_registry_id: str,
    family_id: RegistryFamilyId,
    module_path: str,
    builder_name: str,
    contract_kind: ContractKind,
    serialization_version: str,
    source_capability: str,
    public_export_names: tuple[str, ...],
    pydantic_model_names: tuple[str, ...],
    dependency_registry_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "source_registry_id": source_registry_id,
        "family_id": family_id,
        "module_path": module_path,
        "builder_name": builder_name,
        "contract_kind": contract_kind,
        "serialization_version": serialization_version,
        "source_capability": source_capability,
        "public_export_names": public_export_names,
        "pydantic_model_names": pydantic_model_names,
        "dependency_registry_ids": dependency_registry_ids,
        "normalized_schema_keys": (
            "role",
            "serialization_version",
            "authority_boundary",
            "source_registry_ids",
            "blocked_runtime_behaviors",
            "advisory_only",
        ),
    }


def _review_findings() -> tuple[RegistryReviewFinding, ...]:
    findings = (
        (
            "Pydantic Review",
            "pass",
            (
                "V7.3 consolidation models use frozen Pydantic v2 contracts.",
                "Shared passive base flags make mutation boundaries explicit.",
            ),
            "Use PassiveBoundaryModel for new passive consolidation records.",
        ),
        (
            "Jinja2 Review",
            "pass",
            (
                "Existing prompt renderer remains the only Jinja2 integration.",
                "No template source or rendered prompt behavior is changed.",
            ),
            "Keep registry consolidation outside prompt rendering.",
        ),
        (
            "Style Review",
            "pass",
            (
                "New module follows explicit registry/report naming.",
                "No UI styles or generated artifact styles are changed.",
            ),
            "Prefer compact report records over per-task metadata modules.",
        ),
        (
            "Code Style & Comment Quality Audit",
            "pass",
            (
                "Comments are limited to model/function docstrings.",
                "Validators encode invariants rather than prose-only notes.",
            ),
            "Keep future comments tied to non-obvious validation rules.",
        ),
        (
            "Logging Architecture Review",
            "pass",
            (
                "No loguru, stdlib logging, sink, or level configuration changes.",
                "Registry reports remain returned data, not emitted telemetry.",
            ),
            "Do not log passive registry audits from builders.",
        ),
        (
            "Contract Simplification",
            "pass",
            (
                "Shared builder, boundary, schema, export, and integrity models "
                "remove duplicate report shapes.",
            ),
            "Keep source registries in place while normalizing their metadata.",
        ),
        (
            "Metadata-to-Code Ratio Review",
            "pass",
            (
                "One aggregate V7.3 module covers all roadmap items.",
                "Inventory records summarize existing surfaces without copying "
                "their full registries.",
            ),
            "Avoid duplicating source registry payloads in consolidation reports.",
        ),
        (
            "Registry Package Consolidation",
            "pass",
            (
                "A single orchestration module owns V7.3 consolidation metadata.",
                "Existing registry modules keep their stable ownership boundaries.",
            ),
            "Do not move mature registry modules during V7.3.",
        ),
    )
    return tuple(
        RegistryReviewFinding(
            finding_id=f"review::{_slug(roadmap_item)}",
            roadmap_item=roadmap_item,
            status=status,
            evidence=evidence,
            recommendation=recommendation,
        )
        for roadmap_item, status, evidence, recommendation in findings
    )


def _artifact_covered_roadmap_items() -> tuple[str, ...]:
    return (
        *SourceRegistryInventory.model_fields["covered_roadmap_items"].default,
        "Registry Coverage Reports",
        "Contract Schema Normalization",
        "Import Surface Stabilization",
        "Public Export Audit",
        *(finding.roadmap_item for finding in _review_findings()),
        "Registry Integrity Verification",
        "Contract Compatibility Checker",
        "Schema Evolution Manager",
        "Contract Version Migration",
        "Registry Explainability",
        "Registry Dependency Graph",
        "Registry Diff Engine",
        "Architecture Simplification Review",
    )


def _public_export_names() -> tuple[str, ...]:
    from creative_coding_assistant import orchestration

    return tuple(orchestration.__all__)


def _stable_schema_hash(source: RegistrySourceRecord) -> str:
    raw = "|".join(
        (
            source.source_registry_id,
            source.serialization_version,
            ",".join(source.normalized_schema_keys),
            ",".join(source.public_export_names),
        )
    )
    return f"registry_schema::{sha256(raw.encode()).hexdigest()[:16]}"


def _next_schema_version(version: str) -> str:
    if version.endswith(".v1"):
        return f"{version[:-2]}.v2"
    return f"{version}.next"


def _slug(value: str) -> str:
    return value.lower().replace("&", "and").replace("/", "_").replace(" ", "_")


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    counts = Counter(values)
    return tuple(value for value, count in counts.items() if count > 1)


def _is_passive_model(item: PassiveBoundaryModel) -> bool:
    return (
        item.advisory_only
        and not item.provider_model_routing_change_implemented
        and not item.provider_execution_implemented
        and not item.workflow_execution_implemented
        and not item.workflow_graph_mutation_implemented
        and not item.prompt_rendering_change_implemented
        and not item.jinja_template_mutation_implemented
        and not item.logging_configuration_mutation_implemented
        and not item.persistent_storage_write_implemented
        and not item.generated_output_mutation_implemented
        and not item.runtime_evolution_implemented
    )
