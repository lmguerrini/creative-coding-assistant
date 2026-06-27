import re
import unittest
from pathlib import Path

from creative_coding_assistant.orchestration import ASSISTANT_WORKFLOW_NODE_ORDER

REPO_ROOT = Path(__file__).resolve().parents[1]
CREATIVE_COGNITION_CAPABILITIES = (
    "Creative Intent Decomposer",
    "Creative Hierarchy Planner",
    "Creative Strategy Engine",
    "Creative Technique Selector",
    "Creative Constraint Solver",
    "Runtime Capability Reasoner",
    "Creative Trade-off Explorer",
    "Creative Constraint Prioritizer",
    "Creative Quality Predictor",
    "Symbolic Narrative Planner",
    "Creative Composition Planner",
)
V32_GENERATIVE_DESIGN_CAPABILITIES = (
    "Procedural Structure Planner",
    "Generative Structure Engine",
    "Semantic Motif Engine",
    "Emotional Consistency Engine",
    "Cross-Modality Composer",
    "Audio-Visual Scene System",
)
V33_ARTIFACT_INTELLIGENCE_CAPABILITIES = (
    "Artifact Planner",
    "Artifact Dependency Graph",
    "Runtime Compatibility Engine",
    "Artifact Capability Matrix",
    "Multi-Artifact Strategy",
    "Artifact Critic",
    "Artifact Refiner",
    "Artifact Intelligence Synthesis",
    "Artifact Merge Planner",
    "Artifact Export Intelligence",
    "Artifact Engine Contracts",
)
V35_WORKSTATION_SURFACES = (
    "Workstation State",
    "Session Intelligence",
    "Workflow Explorer",
    "Provenance Engine",
    "Creative Timeline",
    "V3 Inspector Panels",
    "Workstation Dashboard",
)
V43_HYBRID_WORKFLOW_REGISTRIES = (
    "V3 Backbone Mode Registry",
    "Conditional Multi-Agent Escalation Registry",
    "Specialist Agent Loop Registry",
    "Escalation Gate Registry",
    "Creative Escalation Policy Registry",
    "Reflection Escalation Registry",
    "Hybrid Agent Debate Loop Registry",
    "Hybrid Agent Voting Registry",
    "Agent Confidence Fusion Registry",
    "Decision Provenance Registry",
    "Escalation Trace Registry",
    "Creative Exploration Budget Registry",
    "Result Normalization Registry",
    "Return-to-Workflow Handoff Registry",
    "HITL Escalation Gate Registry",
    "Confidence Threshold Routing Registry",
    "Cost Threshold Routing Registry",
    "Latency Threshold Routing Registry",
    "Ambiguity Escalation Registry",
    "Risk Escalation Registry",
    "Quality Escalation Registry",
    "Adaptive Multi-Agent Escalation Registry",
)
PUBLIC_README_INTERNAL_MARKERS = (
    "Current Branch Status",
    "feature/",
    "current branch",
    "Macro-Capability Lifecycle",
    "Role Split",
    "Junie",
    "Codex",
    "ChatGPT",
    "Create Version Tag",
    "Merge & Push",
    "Engineering Validation",
    "close `v3.3.0`",
)


class WorkflowDocumentationAlignmentTests(unittest.TestCase):
    def test_readme_workflow_order_matches_backend_node_order(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        expected_order = " -> ".join(ASSISTANT_WORKFLOW_NODE_ORDER)

        self.assertIn(f"`{expected_order}`", readme)

    def test_readme_remains_product_oriented_public_documentation(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        normalized_readme = re.sub(r"\s+", " ", readme)

        self.assertIn("AI-native creative translation", readme)
        self.assertIn("creative coding platform", readme)
        self.assertIn("V3.5", readme)
        self.assertIn("Creative Workstation", readme)
        self.assertIn("V4.1", readme)
        self.assertIn("Multi-Agent Core", readme)
        self.assertIn("V4.2", readme)
        self.assertIn("Agent Orchestration", readme)
        self.assertIn("V4.3", readme)
        self.assertIn("Hybrid Agentic Workflow", readme)
        self.assertIn("Next.js workstation", readme)
        self.assertIn("Capability Scope", readme)
        self.assertIn("architecture/artifact_intelligence_graph.md", readme)
        self.assertIn("architecture/workstation_surface_graph.md", readme)
        self.assertIn("architecture/engine_matrix.md", readme)
        self.assertIn("Creative Evaluation", readme)
        self.assertIn("Workstation Dashboard", readme)
        self.assertIn("Creative Timeline", readme)
        self.assertIn("Provenance Engine", readme)
        self.assertIn("Agent Identity Registry", readme)
        self.assertIn("Agent Contract Registry", readme)
        self.assertIn("Agent Memory Contract Registry", readme)
        self.assertIn("Agent Metadata Registry", readme)
        self.assertIn("Agent Routing Registry", readme)
        self.assertIn("Blackboard Memory Registry", readme)
        self.assertIn("Shared Context View Registry", readme)
        self.assertIn("Agent Lifecycle Registry", readme)
        self.assertIn("Workflow Agent Handoff Registry", readme)
        self.assertIn("Orchestration Contract Integration Registry", readme)
        for registry_name in V43_HYBRID_WORKFLOW_REGISTRIES:
            self.assertIn(registry_name, readme)
        self.assertIn("Hybrid Workflow Integration source coverage", readme)
        self.assertIn("metadata-only", readme)
        self.assertIn("passive role and contract metadata", normalized_readme)
        self.assertIn("passive orchestration metadata", normalized_readme)
        self.assertIn("passive hybrid workflow metadata", normalized_readme)
        self.assertIn("orchestration readiness metadata", normalized_readme)
        self.assertIn("not active multi-agent orchestration", normalized_readme)
        self.assertIn(
            "not active agent execution or autonomous escalation",
            normalized_readme,
        )
        self.assertIn("do not execute orchestration", normalized_readme)
        self.assertIn("do not create agents", normalized_readme)
        self.assertIn("route tasks", normalized_readme)
        self.assertIn("modify artifacts", normalized_readme)
        self.assertIn("select runtimes", normalized_readme)
        self.assertIn("product roadmap context", normalized_readme)

        for internal_marker in PUBLIC_README_INTERNAL_MARKERS:
            self.assertNotIn(internal_marker, readme)

    def test_architecture_doc_node_order_matches_backend_node_order(self) -> None:
        architecture_doc = (
            REPO_ROOT / "architecture" / "workflow_graph.md"
        ).read_text(encoding="utf-8")
        marker = (
            "`ASSISTANT_WORKFLOW_NODE_ORDER` is the source of truth for node "
            "ordering:"
        )
        section = architecture_doc.split(marker, maxsplit=1)[1].split(
            "Current transition rules:",
            maxsplit=1,
        )[0]

        listed_nodes = tuple(
            re.findall(r"^\d+\.\s+`([^`]+)`$", section, flags=re.MULTILINE)
        )

        self.assertEqual(listed_nodes, ASSISTANT_WORKFLOW_NODE_ORDER)
        self.assertIn(
            "start --> intake --> routing --> memory --> retrieval --> "
            "context_assembly --> prompt_input --> planning --> "
            "director --> reasoning --> prompt_rendering --> generation",
            architecture_doc,
        )

    def test_mermaid_source_keeps_reasoning_between_director_and_prompt_rendering(
        self,
    ) -> None:
        mermaid = (
            REPO_ROOT / "architecture" / "workflow_graph.mmd"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "prompt_input --> planning --> director --> reasoning --> prompt_rendering",
            mermaid,
        )
        self.assertIn("V4.3 hybrid workflow metadata boundary", mermaid)
        self.assertIn("no runtime escalation", mermaid)

    def test_workflow_doc_distinguishes_runtime_pipeline_and_dependency_views(
        self,
    ) -> None:
        architecture_doc = (
            REPO_ROOT / "architecture" / "workflow_graph.md"
        ).read_text(encoding="utf-8")
        normalized_architecture_doc = re.sub(r"\s+", " ", architecture_doc)

        self.assertIn(
            "## Runtime Graph Vs Internal Capability Graph",
            architecture_doc,
        )
        self.assertIn("creative_intelligence_graph.md", architecture_doc)
        self.assertIn("creative_intelligence_graph.mmd", architecture_doc)
        self.assertIn("generative_design_graph.md", architecture_doc)
        self.assertIn("generative_design_graph.mmd", architecture_doc)
        self.assertIn("artifact_intelligence_graph.md", architecture_doc)
        self.assertIn("artifact_intelligence_graph.mmd", architecture_doc)
        self.assertIn("workstation_surface_graph.md", architecture_doc)
        self.assertIn("workstation_surface_graph.mmd", architecture_doc)
        self.assertIn(
            "metadata, design guidance, artifact intelligence, evaluation "
            "summaries, and contract summaries",
            normalized_architecture_doc,
        )
        self.assertIn("workstation surface contracts", architecture_doc)
        self.assertIn("not code generation execution", normalized_architecture_doc)
        self.assertIn("true multi-agent", architecture_doc)
        self.assertIn("multi-node runtime graph", architecture_doc)
        self.assertIn("## V4.1 Multi-Agent Core Contract Boundary", architecture_doc)
        self.assertIn("## V4.2 Agent Orchestration Metadata Boundary", architecture_doc)
        self.assertIn(
            "## V4.3 Hybrid Agentic Workflow Metadata Boundary",
            architecture_doc,
        )
        self.assertIn("Agent Contract Registry", architecture_doc)
        self.assertIn("Agent Memory Contract Registry", architecture_doc)
        self.assertIn("Agent Metadata Registry", architecture_doc)
        self.assertIn("Agent Routing Registry", architecture_doc)
        self.assertIn("Blackboard Memory Registry", architecture_doc)
        self.assertIn("Workflow Agent Handoff Registry", architecture_doc)
        self.assertIn("Orchestration Contract Integration Registry", architecture_doc)
        self.assertIn("Adaptive Escalation", architecture_doc)
        self.assertIn("Hybrid Workflow Integration source coverage", architecture_doc)
        self.assertIn("passive orchestration metadata", normalized_architecture_doc)
        self.assertIn("passive hybrid workflow metadata", normalized_architecture_doc)
        self.assertIn("do not execute orchestration", normalized_architecture_doc)
        self.assertIn("do not execute escalation", normalized_architecture_doc)
        self.assertIn("do not enter provider prompts", normalized_architecture_doc)
        self.assertIn("LangGraph node ordering", architecture_doc)

    def test_creative_intelligence_graph_docs_cover_current_pipeline(self) -> None:
        architecture_doc = (
            REPO_ROOT / "architecture" / "creative_intelligence_graph.md"
        ).read_text(encoding="utf-8")
        mermaid = (
            REPO_ROOT / "architecture" / "creative_intelligence_graph.mmd"
        ).read_text(encoding="utf-8")

        for capability in (
            *CREATIVE_COGNITION_CAPABILITIES,
            *V32_GENERATIVE_DESIGN_CAPABILITIES,
        ):
            self.assertIn(capability, architecture_doc)
            self.assertIn(capability, mermaid)

        self.assertIn("Creative Execution Plan", architecture_doc)
        self.assertIn("Creative Execution Plan", mermaid)
        self.assertIn("Metadata Store", architecture_doc)
        self.assertIn("Metadata Store", mermaid)
        self.assertIn("Creative Assistant Director runtime node", architecture_doc)
        self.assertIn("Creative Assistant Director runtime node", mermaid)
        self.assertIn("Creative Reasoning Engine runtime node", architecture_doc)
        self.assertIn("Creative Reasoning Engine runtime node", mermaid)
        self.assertIn("Creative Evaluation", architecture_doc)
        self.assertIn("Creative Evaluation", mermaid)
        self.assertIn("Workstation Hydration", architecture_doc)
        self.assertIn("Workstation Hydration", mermaid)
        self.assertIn("workstation_surface_graph.md", architecture_doc)
        self.assertIn("human-readable pipeline", architecture_doc)
        self.assertIn("serpentine readability view", architecture_doc)
        self.assertIn("dependency matrix remains the preferred way", architecture_doc)
        self.assertIn("not separate LangGraph nodes", architecture_doc)
        self.assertIn("future V4 multi-agent blueprint", architecture_doc)
        self.assertIn("flowchart TB", mermaid)
        self.assertIn("direction RL", mermaid)
        self.assertIn("director --> reasoning --> prompt_rendering", mermaid)

    def test_generative_design_graph_docs_cover_current_capabilities_and_matrix(
        self,
    ) -> None:
        architecture_doc = (
            REPO_ROOT / "architecture" / "generative_design_graph.md"
        ).read_text(encoding="utf-8")
        mermaid = (
            REPO_ROOT / "architecture" / "generative_design_graph.mmd"
        ).read_text(encoding="utf-8")

        for capability in V32_GENERATIVE_DESIGN_CAPABILITIES:
            self.assertIn(capability, architecture_doc)
            self.assertIn(capability, mermaid)

        self.assertIn("## Dependency Matrix", architecture_doc)
        self.assertIn("| Capability | Reads | Produces | Used by |", architecture_doc)
        self.assertIn("preferred way to show dense dependencies", architecture_doc)
        self.assertIn("metadata and design guidance", architecture_doc)
        self.assertIn("future V4 multi-agent blueprint", architecture_doc)
        self.assertIn("AssistantWorkflowState", architecture_doc)
        self.assertIn("PromptInputResponse", architecture_doc)
        self.assertIn("V3.5 workstation surfaces", architecture_doc)
        self.assertIn("V3.5 Workstation surfaces", mermaid)
        self.assertIn("metadata_store --> director", mermaid)
        self.assertIn("director --> reasoning --> prompt_rendering", mermaid)

    def test_engine_matrix_covers_cross_cutting_layers_and_versions(self) -> None:
        engine_matrix = (
            REPO_ROOT / "architecture" / "engine_matrix.md"
        ).read_text(encoding="utf-8")
        normalized_engine_matrix = re.sub(r"\s+", " ", engine_matrix)

        self.assertIn("Versions are chronological delivery increments.", engine_matrix)
        self.assertIn("cross-cutting architecture layers", engine_matrix)

        for layer in (
            "Core Engine",
            "Knowledge Engine",
            "Execution Engine",
            "Experience Layer",
        ):
            self.assertIn(layer, engine_matrix)

        for version_marker in (
            "V3.1",
            "V3.2",
            "V3.3",
            "V3.4",
            "Creative Evaluation",
            "V3.5",
            "Creative Workstation",
            "V3.6",
            "Stabilization & Refactor Pass",
            "V4.1",
            "Multi-Agent Core",
            "V4.2",
            "Agent Orchestration",
            "V4.3",
            "Hybrid Agentic Workflow",
            "V4",
            "Agentic Studio",
            "V5",
            "Execution Optimization & Production Intelligence",
            "V6",
            "HoloGenesis Core OS",
        ):
            self.assertIn(version_marker, engine_matrix)

        self.assertIn("workstation_surface_graph.md", engine_matrix)
        self.assertIn("workstation_engine_contract_registry.v1", engine_matrix)
        self.assertIn("V3.6 Audit And Future-Readiness Metadata", engine_matrix)
        self.assertIn("## V4.1 Multi-Agent Core Registries", engine_matrix)
        self.assertIn("## V4.2 Agent Orchestration Registries", engine_matrix)
        self.assertIn("## V4.3 Hybrid Agentic Workflow Registries", engine_matrix)

        for registry_marker in (
            "agent_capability_registry.v1",
            "escalation_policy_registry.v1",
            "hybrid_workflow_registry.v1",
            "engine_contract_consistency_registry.v1",
            "agent_contract_registry.v1",
            "agent_identity_registry.v1",
            "agent_memory_contract_registry.v1",
            "agent_role_registry.v1",
            "agent_boundary_registry.v1",
            "agent_metadata_registry.v1",
            "agent_routing_registry.v1",
            "blackboard_memory_registry.v1",
            "shared_context_view_registry.v1",
            "agent_dependency_graph.v1",
            "parallel_scheduling_registry.v1",
            "coordination_registry.v1",
            "agent_debate_registry.v1",
            "consensus_builder_registry.v1",
            "agent_capability_alignment_registry.v1",
            "agent_escalation_signal_registry.v1",
            "agent_lifecycle_registry.v1",
            "agent_state_sync_registry.v1",
            "workflow_agent_handoff_registry.v1",
            "orchestration_contract_integration.v1",
            "v3_backbone_mode_registry.v1",
            "conditional_multi_agent_escalation_registry.v1",
            "specialist_agent_loop_registry.v1",
            "escalation_gate_registry.v1",
            "creative_escalation_policy_registry.v1",
            "reflection_escalation_registry.v1",
            "hybrid_agent_debate_loop_registry.v1",
            "hybrid_agent_voting_registry.v1",
            "agent_confidence_fusion_registry.v1",
            "decision_provenance_registry.v1",
            "escalation_trace_registry.v1",
            "creative_exploration_budget_registry.v1",
            "result_normalization_registry.v1",
            "return_to_workflow_handoff_registry.v1",
            "hitl_escalation_gate_registry.v1",
            "confidence_threshold_routing_registry.v1",
            "cost_threshold_routing_registry.v1",
            "latency_threshold_routing_registry.v1",
            "ambiguity_escalation_registry.v1",
            "risk_escalation_registry.v1",
            "quality_escalation_registry.v1",
            "adaptive_multi_agent_escalation_registry.v1",
        ):
            self.assertIn(registry_marker, engine_matrix)

        for module_path in (
            "agent_capabilities.py",
            "escalation_policy.py",
            "hybrid_agentic_workflow.py",
            "engine_contract_consistency.py",
            "agent_contracts.py",
            "agent_identities.py",
            "agent_memory_contracts.py",
            "agent_roles.py",
            "agent_boundaries.py",
            "agent_metadata.py",
            "agent_routing.py",
            "blackboard_memory.py",
            "shared_context_views.py",
            "agent_dependency_graph.py",
            "agent_parallel_scheduling.py",
            "agent_coordination.py",
            "agent_debate.py",
            "agent_consensus.py",
            "agent_capability_alignment.py",
            "agent_escalation_signals.py",
            "agent_lifecycle.py",
            "agent_state_synchronization.py",
            "workflow_agent_handoff.py",
            "orchestration_contract_integration.py",
        ):
            self.assertIn(module_path, engine_matrix)

        self.assertIn("export-only metadata surfaces", engine_matrix)
        self.assertIn("not active multi-agent orchestration", normalized_engine_matrix)
        self.assertIn("passive hybrid workflow metadata", normalized_engine_matrix)
        self.assertIn("runtime selection", normalized_engine_matrix)
        self.assertIn("storage behavior", normalized_engine_matrix)
        self.assertIn("active runtime orchestration", normalized_engine_matrix)
        self.assertIn("change LangGraph node order", normalized_engine_matrix)
        self.assertIn(
            "not additional LangGraph runtime nodes",
            normalized_engine_matrix,
        )
        self.assertIn(
            "do not enter workflow payloads",
            normalized_engine_matrix,
        )
        self.assertIn(
            "do not alter provider/model routing",
            normalized_engine_matrix,
        )
        self.assertIn(
            "generated output, or the V3 node order",
            normalized_engine_matrix,
        )

    def test_workstation_surface_docs_cover_v35_surface_layer(self) -> None:
        architecture_doc = (
            REPO_ROOT / "architecture" / "workstation_surface_graph.md"
        ).read_text(encoding="utf-8")
        mermaid = (
            REPO_ROOT / "architecture" / "workstation_surface_graph.mmd"
        ).read_text(encoding="utf-8")
        normalized_architecture_doc = re.sub(r"\s+", " ", architecture_doc)

        for surface in V35_WORKSTATION_SURFACES:
            self.assertIn(surface, architecture_doc)
            self.assertIn(surface, mermaid)

        self.assertIn("metadata projection and inspection layer", architecture_doc)
        self.assertIn("workstation_engine_contract_registry.v1", architecture_doc)
        self.assertIn("Future V4/V5/V6 consumers", mermaid)
        self.assertIn("named hooks only", mermaid)
        self.assertIn(
            "does not add provider routing, runtime selection, execution "
            "optimization, artifact execution, artifact modification, "
            "autonomous retries, preview execution, or generated output changes",
            normalized_architecture_doc,
        )


if __name__ == "__main__":
    unittest.main()
