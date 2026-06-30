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
V44_HYBRID_STUDIO_REGISTRIES = (
    "Local Model Registry",
    "Cloud Model Registry",
    "Hybrid Execution Registry",
    "Auto Mode Registry",
    "Studio Mode Registry",
    "HITL Decision Registry",
    "Provider Selection Registry",
    "Execution Simulator Registry",
    "Model Profile Registry",
    "Cost Profile Registry",
    "Quality Profile Registry",
    "Local/Cloud Comparison Registry",
    "Agent Workspace Registry",
    "Agent Conversation View Registry",
    "Workspace Snapshot Registry",
    "Session Replay Registry",
    "Execution Replay Registry",
    "Hybrid Studio Integration Registry",
)
V45_MULTIMODAL_STUDIO_REGISTRIES = (
    "Live Preview Registry",
    "Multi Preview Registry",
    "Interactive Canvas Registry",
    "Visual Workspace Registry",
    "Runtime Collaboration Registry",
    "Artifact Collaboration Registry",
    "Artifact Provenance Registry",
    "Artifact Lineage Registry",
    "Cross-Agent Workspace Registry",
    "Shared Artifact Board Registry",
    "Workspace History Registry",
    "Branching Timeline Registry",
    "Creative Evolution Timeline Registry",
    "Real-Time Workflow Visualization Registry",
    "Multimodal Studio Integration Registry",
)
V46_AGENTIC_STUDIO_HARDENING_REGISTRIES = (
    "Agent Contract Audit Registry",
    "Escalation Policy Audit Registry",
    "Hybrid Workflow Audit Registry",
    "Agent Registry Audit Registry",
    "Blackboard Audit Registry",
    "Shared Context Audit Registry",
    "Agent Collaboration Audit Registry",
    "Creative Diversity Audit Registry",
    "Agent Explainability Audit Registry",
    "Agent Reliability Audit Registry",
    "Agent Determinism Audit Registry",
    "Agent Telemetry Foundation Registry",
    "Agent Cost Tracking Foundation Registry",
    "Agent Performance Tracking Foundation Registry",
    "Architecture Consistency Pass Registry",
    "Final V4 Hardening Registry",
    "LangGraph Error Path Audit",
)
V52_MODEL_ROUTING_SURFACES = (
    "Model Router",
    "Local vs Cloud Routing",
    "Hybrid Routing",
    "Quality/Cost Optimizer",
    "Cost Estimator",
    "Budget Policies",
    "HITL Budget Gate",
    "Runtime Recommendation Engine",
    "Execution Policy Engine",
    "Model Recommendation Engine",
    "Model Capability Matrix",
    "Provider Capability Matrix",
    "Quality Prediction Engine",
    "Cost Prediction Engine",
    "Creative Quality Predictor",
    "Creative Diversity Predictor",
    "Creative Consistency Predictor",
    "Routing Explainability",
)
V54_PRODUCTION_OBSERVABILITY_SURFACES = (
    "Token Dashboard",
    "Cost Dashboard",
    "Quality Dashboard",
    "Performance Dashboard",
    "Production Telemetry",
    "Workflow Diagnostics",
    "Agent Diagnostics",
    "Routing Diagnostics",
    "Escalation Diagnostics",
    "Failure Analysis",
    "Error Intelligence",
    "Workflow Health Monitoring",
    "System Health Monitoring",
    "Creative Analytics",
    "Confidence Analytics",
    "Creative Diversity Analytics",
    "Runtime Timeline",
    "Workflow Explainability Dashboard",
    "Production Observability Architecture Consistency",
    "Production Observability Failure Path Audit",
)
V55_ADAPTIVE_EXECUTION_SURFACES = (
    "Adaptive Hybrid Workflow Optimizer",
    "Adaptive Escalation Optimizer",
    "Agent Activation Optimizer",
    "Adaptive Cost/Quality Optimizer",
    "Adaptive Latency Optimizer",
    "Adaptive Execution Strategy Selection",
    "Dynamic Agent Allocation",
    "Dynamic Resource Allocation",
    "Workflow Self-Tuning Policies",
    "Execution Confidence Engine",
    "Workflow Risk Engine",
    "Creative Exploration Optimizer",
    "Emergence Optimizer",
    "Agent Diversity Optimizer",
    "Reflection Budget Optimizer",
    "Adaptive Policy Explainability",
    "Adaptive Execution Architecture Consistency",
    "Adaptive Execution Failure Path Audit",
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
        self.assertIn("V4.4", readme)
        self.assertIn("Hybrid Studio", readme)
        self.assertIn("V4.5", readme)
        self.assertIn("Multimodal Studio", readme)
        self.assertIn("V4.6", readme)
        self.assertIn("Agentic Studio Hardening", readme)
        self.assertIn("V5.1", readme)
        self.assertIn("Execution Optimization", readme)
        self.assertIn("V5.2", readme)
        self.assertIn("Intelligent Model Routing", readme)
        self.assertIn("V5.4", readme)
        self.assertIn("Production Observability", readme)
        self.assertIn("V5.5", readme)
        self.assertIn("Adaptive Execution Intelligence", readme)
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
        for registry_name in V44_HYBRID_STUDIO_REGISTRIES:
            self.assertIn(registry_name, readme)
        self.assertIn("Hybrid Studio Integration source coverage", readme)
        for registry_name in V45_MULTIMODAL_STUDIO_REGISTRIES:
            self.assertIn(registry_name, readme)
        self.assertIn("Multimodal Studio Integration source coverage", readme)
        for registry_name in V46_AGENTIC_STUDIO_HARDENING_REGISTRIES:
            self.assertIn(registry_name, readme)
        for surface in V52_MODEL_ROUTING_SURFACES:
            self.assertIn(surface, readme)
        for surface in V54_PRODUCTION_OBSERVABILITY_SURFACES:
            self.assertIn(surface, readme)
        for surface in V55_ADAPTIVE_EXECUTION_SURFACES:
            self.assertIn(surface, readme)
        self.assertIn("Execution Optimization Failure Audit", readme)
        self.assertIn("Model Routing Architecture Consistency", readme)
        self.assertIn("Model Routing Failure Path Audit", readme)
        self.assertIn("metadata-only", readme)
        self.assertIn("passive role and contract metadata", normalized_readme)
        self.assertIn("passive orchestration metadata", normalized_readme)
        self.assertIn("passive hybrid workflow metadata", normalized_readme)
        self.assertIn("passive hybrid studio metadata", normalized_readme)
        self.assertIn("passive multimodal studio metadata", normalized_readme)
        self.assertIn(
            "passive agentic studio hardening metadata",
            normalized_readme,
        )
        self.assertIn("advisory execution optimization metadata", normalized_readme)
        self.assertIn("advisory model-routing metadata", normalized_readme)
        self.assertIn("read-only observability metadata", normalized_readme)
        self.assertIn("advisory adaptive execution metadata", normalized_readme)
        self.assertIn("adaptive behavior application", normalized_readme)
        self.assertIn("orchestration readiness metadata", normalized_readme)
        self.assertIn("not active Studio runtime", normalized_readme)
        self.assertIn("not rendering execution", normalized_readme)
        self.assertIn("not active multi-agent orchestration", normalized_readme)
        self.assertIn(
            "not active agent execution or autonomous escalation",
            normalized_readme,
        )
        self.assertIn("does not activate Studio runtime", normalized_readme)
        self.assertIn("does not execute rendering", normalized_readme)
        self.assertIn("do not execute orchestration", normalized_readme)
        self.assertIn("do not create agents", normalized_readme)
        self.assertIn("route tasks", normalized_readme)
        self.assertIn("modify artifacts", normalized_readme)
        self.assertIn("select runtimes", normalized_readme)
        self.assertIn("product roadmap context", normalized_readme)
        self.assertIn("bypass failure normalization", normalized_readme)
        self.assertIn("runtime hardening engine", normalized_readme)
        self.assertIn("not active model selection", normalized_readme)
        self.assertIn("do not apply routing", normalized_readme)
        self.assertIn("provider/model switching", normalized_readme)
        self.assertIn("provider execution", normalized_readme)
        self.assertIn("Runtime Evolution", normalized_readme)

        for internal_marker in PUBLIC_README_INTERNAL_MARKERS:
            self.assertNotIn(internal_marker, readme)

    def test_project_docs_cover_current_v4_3_passive_scope(self) -> None:
        project_context = (
            REPO_ROOT / "docs" / "PROJECT_CONTEXT.md"
        ).read_text(encoding="utf-8")
        roadmap = (
            REPO_ROOT / "docs" / "IMPLEMENTATION_ROADMAP.md"
        ).read_text(encoding="utf-8")
        decisions = (
            REPO_ROOT / "docs" / "ARCHITECTURE_DECISIONS.md"
        ).read_text(encoding="utf-8")
        combined = "\n".join((project_context, roadmap, decisions))
        normalized_project_context = re.sub(r"\s+", " ", project_context)
        normalized_combined = re.sub(r"\s+", " ", combined)

        self.assertIn("V4.3 Hybrid Agentic Workflow", combined)
        self.assertIn("Adaptive Multi-Agent Escalation Registry", project_context)
        self.assertIn(
            "Hybrid Workflow Integration source coverage",
            normalized_project_context,
        )
        self.assertIn("completed passive metadata layer", roadmap)
        self.assertIn("V4 Agentic Studio remains future", roadmap)
        self.assertIn("V4.3 Boundary Decision", decisions)
        self.assertIn("passive hybrid workflow metadata", normalized_combined)
        self.assertIn("does not execute agents", normalized_combined)
        self.assertIn("does not execute escalation", normalized_combined)
        self.assertIn("change provider/model routing", normalized_combined)
        self.assertIn("select runtimes", normalized_combined)
        self.assertIn("modify generated output", normalized_combined)
        self.assertIn("compact LangGraph workflow", normalized_combined)

    def test_project_docs_cover_current_v4_4_passive_scope(self) -> None:
        project_context = (
            REPO_ROOT / "docs" / "PROJECT_CONTEXT.md"
        ).read_text(encoding="utf-8")
        roadmap = (
            REPO_ROOT / "docs" / "IMPLEMENTATION_ROADMAP.md"
        ).read_text(encoding="utf-8")
        decisions = (
            REPO_ROOT / "docs" / "ARCHITECTURE_DECISIONS.md"
        ).read_text(encoding="utf-8")
        combined = "\n".join((project_context, roadmap, decisions))
        normalized_project_context = re.sub(r"\s+", " ", project_context)
        normalized_combined = re.sub(r"\s+", " ", combined)

        self.assertIn("V4.4 Hybrid Studio", combined)
        self.assertIn("Hybrid Studio Integration Registry", project_context)
        self.assertIn(
            "Hybrid Studio Integration source coverage",
            normalized_project_context,
        )
        self.assertIn("V4.4 layer is a completed passive metadata layer", roadmap)
        self.assertIn("V4.4 Boundary Decision", decisions)
        self.assertIn("passive hybrid studio metadata", normalized_combined)
        self.assertIn("does not execute providers", normalized_combined)
        self.assertIn("does not activate Studio runtime", normalized_combined)
        self.assertIn("change provider/model routing", normalized_combined)
        self.assertIn("select runtimes", normalized_combined)
        self.assertIn("modify generated output", normalized_combined)
        self.assertIn("compact LangGraph workflow", normalized_combined)

    def test_project_docs_cover_current_v4_5_passive_scope(self) -> None:
        project_context = (
            REPO_ROOT / "docs" / "PROJECT_CONTEXT.md"
        ).read_text(encoding="utf-8")
        roadmap = (
            REPO_ROOT / "docs" / "IMPLEMENTATION_ROADMAP.md"
        ).read_text(encoding="utf-8")
        decisions = (
            REPO_ROOT / "docs" / "ARCHITECTURE_DECISIONS.md"
        ).read_text(encoding="utf-8")
        combined = "\n".join((project_context, roadmap, decisions))
        normalized_project_context = re.sub(r"\s+", " ", project_context)
        normalized_combined = re.sub(r"\s+", " ", combined)

        self.assertIn("V4.5 Multimodal Studio", combined)
        self.assertIn("Multimodal Studio Integration Registry", project_context)
        self.assertIn(
            "Multimodal Studio Integration source coverage",
            normalized_project_context,
        )
        self.assertIn("V4.5 layer is a completed passive metadata layer", roadmap)
        self.assertIn("V4.5 Boundary Decision", decisions)
        self.assertIn("passive multimodal studio metadata", normalized_combined)
        self.assertIn("does not execute rendering", normalized_combined)
        self.assertIn("does not activate Studio runtime", normalized_combined)
        self.assertIn("change provider/model routing", normalized_combined)
        self.assertIn("select runtimes", normalized_combined)
        self.assertIn("modify generated output", normalized_combined)
        self.assertIn("compact LangGraph workflow", normalized_combined)

    def test_project_docs_cover_current_v4_6_passive_scope(self) -> None:
        project_context = (
            REPO_ROOT / "docs" / "PROJECT_CONTEXT.md"
        ).read_text(encoding="utf-8")
        roadmap = (
            REPO_ROOT / "docs" / "IMPLEMENTATION_ROADMAP.md"
        ).read_text(encoding="utf-8")
        decisions = (
            REPO_ROOT / "docs" / "ARCHITECTURE_DECISIONS.md"
        ).read_text(encoding="utf-8")
        combined = "\n".join((project_context, roadmap, decisions))
        normalized_project_context = re.sub(r"\s+", " ", project_context)
        normalized_combined = re.sub(r"\s+", " ", combined)

        self.assertIn("V4.6 Agentic Studio Hardening", combined)
        self.assertIn("LangGraph Error Path Audit", project_context)
        self.assertIn("Final V4 Hardening Registry", project_context)
        self.assertIn(
            "passive agentic studio hardening metadata",
            normalized_combined,
        )
        self.assertIn("V4.6 layer is a completed passive metadata layer", roadmap)
        self.assertIn("V4.6 Boundary Decision", decisions)
        self.assertIn("does not execute hardening checks", normalized_combined)
        self.assertIn("add LangGraph nodes", normalized_combined)
        self.assertIn("bypass failure normalization", normalized_combined)
        self.assertIn("activate passive registries", normalized_combined)
        self.assertIn("change provider/model routing", normalized_combined)
        self.assertIn("select runtimes", normalized_combined)
        self.assertIn("modify generated output", normalized_combined)
        self.assertIn(
            "terminal failure coverage",
            normalized_project_context,
        )

    def test_project_docs_cover_current_v5_2_advisory_scope(self) -> None:
        project_context = (
            REPO_ROOT / "docs" / "PROJECT_CONTEXT.md"
        ).read_text(encoding="utf-8")
        decisions = (
            REPO_ROOT / "docs" / "ARCHITECTURE_DECISIONS.md"
        ).read_text(encoding="utf-8")
        roadmap = (
            REPO_ROOT / "docs" / "IMPLEMENTATION_ROADMAP.md"
        ).read_text(encoding="utf-8")
        combined = "\n".join((project_context, roadmap, decisions))
        normalized_combined = re.sub(r"\s+", " ", combined)

        self.assertIn("V5.2 Intelligent Model Routing Engine", combined)
        self.assertIn("V5.2 Advisory Boundary", project_context)
        self.assertIn("V5.2 Boundary Decision", decisions)
        self.assertIn("completed advisory metadata layer", roadmap)
        self.assertIn("advisory model-routing metadata", normalized_combined)
        self.assertIn("routing explainability", normalized_combined)
        self.assertIn("runtime failure-path audit coverage", normalized_combined)
        self.assertIn("does not apply routing", normalized_combined)
        self.assertIn("switch providers or models", normalized_combined)
        self.assertIn("execute providers", normalized_combined)
        self.assertIn("emit HITL requests", normalized_combined)
        self.assertIn("enforce budgets", normalized_combined)
        self.assertIn("apply Runtime Evolution", normalized_combined)
        self.assertIn("modify generated output", normalized_combined)

    def test_project_docs_cover_current_v5_4_read_only_scope(self) -> None:
        project_context = (
            REPO_ROOT / "docs" / "PROJECT_CONTEXT.md"
        ).read_text(encoding="utf-8")
        decisions = (
            REPO_ROOT / "docs" / "ARCHITECTURE_DECISIONS.md"
        ).read_text(encoding="utf-8")
        roadmap = (
            REPO_ROOT / "docs" / "IMPLEMENTATION_ROADMAP.md"
        ).read_text(encoding="utf-8")
        combined = "\n".join((project_context, decisions, roadmap))
        normalized_combined = re.sub(r"\s+", " ", combined)

        self.assertIn("V5.4 Production Observability", combined)
        self.assertIn("V5.4 Read-Only Observability Boundary", project_context)
        self.assertIn("V5.4 Boundary Decision", decisions)
        self.assertIn("completed read-only metadata layer", normalized_combined)
        self.assertIn("read-only observability metadata", normalized_combined)
        self.assertIn("runtime failure-path audit coverage", normalized_combined)
        self.assertIn("collect live metrics", normalized_combined)
        self.assertIn("emit telemetry or alerts", normalized_combined)
        self.assertIn("capture traces", normalized_combined)
        self.assertIn("execute health checks", normalized_combined)
        self.assertIn("classify live errors", normalized_combined)
        self.assertIn("remediate failures", normalized_combined)
        self.assertIn("change provider/model routing", normalized_combined)
        self.assertIn("apply Runtime Evolution", normalized_combined)
        self.assertIn("modify generated output", normalized_combined)

    def test_project_docs_cover_current_v5_5_advisory_scope(self) -> None:
        project_context = (
            REPO_ROOT / "docs" / "PROJECT_CONTEXT.md"
        ).read_text(encoding="utf-8")
        decisions = (
            REPO_ROOT / "docs" / "ARCHITECTURE_DECISIONS.md"
        ).read_text(encoding="utf-8")
        roadmap = (
            REPO_ROOT / "docs" / "IMPLEMENTATION_ROADMAP.md"
        ).read_text(encoding="utf-8")
        combined = "\n".join((project_context, decisions, roadmap))
        normalized_combined = re.sub(r"\s+", " ", combined)

        self.assertIn("V5.5 Adaptive Execution Intelligence", combined)
        self.assertIn("V5.5 Advisory Adaptive Execution Boundary", project_context)
        self.assertIn("V5.5 Boundary Decision", decisions)
        self.assertIn("completed advisory metadata layer", normalized_combined)
        self.assertIn("advisory adaptive execution metadata", normalized_combined)
        self.assertIn("runtime failure-path audit coverage", normalized_combined)
        self.assertIn("apply adaptive policies or strategies", normalized_combined)
        self.assertIn("switch providers or models", normalized_combined)
        self.assertIn("execute providers", normalized_combined)
        self.assertIn("invoke agents", normalized_combined)
        self.assertIn("allocate agents or resources", normalized_combined)
        self.assertIn("emit HITL requests", normalized_combined)
        self.assertIn("mutate workflow graphs", normalized_combined)
        self.assertIn("apply Runtime Evolution", normalized_combined)
        self.assertIn("modify generated output", normalized_combined)

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
        self.assertIn("V4.4 hybrid studio metadata boundary", mermaid)
        self.assertIn("no Studio runtime", mermaid)
        self.assertIn("V4.5 multimodal studio metadata boundary", mermaid)
        self.assertIn("no rendering execution", mermaid)
        self.assertIn("V4.6 agentic studio hardening metadata boundary", mermaid)
        self.assertIn("terminal failure audit only", mermaid)
        self.assertIn("V5.2 model routing metadata boundary", mermaid)
        self.assertIn("no provider/model switching", mermaid)
        self.assertIn("V5.4 production observability metadata boundary", mermaid)
        self.assertIn("no live telemetry emission", mermaid)
        self.assertIn("V5.5 adaptive execution metadata boundary", mermaid)
        self.assertIn("no adaptive behavior application", mermaid)

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
        self.assertIn(
            "## V4.4 Hybrid Studio Metadata Boundary",
            architecture_doc,
        )
        self.assertIn(
            "## V4.5 Multimodal Studio Metadata Boundary",
            architecture_doc,
        )
        self.assertIn(
            "## V4.6 Agentic Studio Hardening Metadata Boundary",
            architecture_doc,
        )
        self.assertIn(
            "## V5.2 Intelligent Model Routing Metadata Boundary",
            architecture_doc,
        )
        self.assertIn(
            "## V5.4 Production Observability Metadata Boundary",
            architecture_doc,
        )
        self.assertIn(
            "## V5.5 Adaptive Execution Intelligence Metadata Boundary",
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
        for registry_name in V44_HYBRID_STUDIO_REGISTRIES:
            self.assertIn(registry_name, architecture_doc)
        for registry_name in V45_MULTIMODAL_STUDIO_REGISTRIES:
            self.assertIn(registry_name, architecture_doc)
        for registry_name in V46_AGENTIC_STUDIO_HARDENING_REGISTRIES:
            self.assertIn(registry_name, architecture_doc)
        for surface in V52_MODEL_ROUTING_SURFACES:
            self.assertIn(surface, normalized_architecture_doc)
        for surface in V54_PRODUCTION_OBSERVABILITY_SURFACES:
            self.assertIn(surface, normalized_architecture_doc)
        for surface in V55_ADAPTIVE_EXECUTION_SURFACES:
            self.assertIn(surface, normalized_architecture_doc)
        self.assertIn("Hybrid Studio Integration source coverage", architecture_doc)
        self.assertIn(
            "Multimodal Studio Integration source coverage",
            architecture_doc,
        )
        self.assertIn("passive orchestration metadata", normalized_architecture_doc)
        self.assertIn("passive hybrid workflow metadata", normalized_architecture_doc)
        self.assertIn("passive hybrid studio metadata", normalized_architecture_doc)
        self.assertIn("passive multimodal studio metadata", normalized_architecture_doc)
        self.assertIn(
            "passive hardening and audit metadata",
            normalized_architecture_doc,
        )
        self.assertIn("advisory model-routing metadata", normalized_architecture_doc)
        self.assertIn(
            "read-only production observability metadata",
            normalized_architecture_doc,
        )
        self.assertIn(
            "advisory adaptive execution metadata",
            normalized_architecture_doc,
        )
        self.assertIn("do not execute orchestration", normalized_architecture_doc)
        self.assertIn("do not execute escalation", normalized_architecture_doc)
        self.assertIn("does not activate Studio runtime", normalized_architecture_doc)
        self.assertIn("does not execute rendering", normalized_architecture_doc)
        self.assertIn("do not execute hardening checks", normalized_architecture_doc)
        self.assertIn("bypass failure normalization", normalized_architecture_doc)
        self.assertIn("do not enter provider prompts", normalized_architecture_doc)
        self.assertIn("LangGraph node ordering", architecture_doc)
        self.assertIn("do not apply routing", normalized_architecture_doc)
        self.assertIn("no provider/model switching", architecture_doc)

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
            "V4.4",
            "Hybrid Studio",
            "V4.5",
            "Multimodal Studio",
            "V4.6",
            "Agentic Studio Hardening",
            "V4",
            "Agentic Studio",
            "V5",
            "Execution Optimization & Production Intelligence",
            "V5.2",
            "Intelligent Model Routing Engine",
            "V5.4",
            "Production Observability",
            "V5.5",
            "Adaptive Execution Intelligence",
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
        self.assertIn("## V4.4 Hybrid Studio Registries", engine_matrix)
        self.assertIn("## V4.5 Multimodal Studio Registries", engine_matrix)
        self.assertIn("## V4.6 Agentic Studio Hardening Registries", engine_matrix)
        self.assertIn("## V5.2 Intelligent Model Routing Surfaces", engine_matrix)
        self.assertIn("## V5.4 Production Observability Surfaces", engine_matrix)
        self.assertIn("## V5.5 Adaptive Execution Intelligence Surfaces", engine_matrix)

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
            "local_model_registry.v1",
            "cloud_model_registry.v1",
            "hybrid_execution_registry.v1",
            "auto_mode_registry.v1",
            "studio_mode_registry.v1",
            "hitl_decision_registry.v1",
            "provider_selection_registry.v1",
            "execution_simulator_registry.v1",
            "model_profile_registry.v1",
            "cost_profile_registry.v1",
            "quality_profile_registry.v1",
            "local_cloud_comparison_registry.v1",
            "agent_workspace_registry.v1",
            "agent_conversation_view_registry.v1",
            "workspace_snapshot_registry.v1",
            "session_replay_registry.v1",
            "execution_replay_registry.v1",
            "hybrid_studio_integration_registry.v1",
            "multimodal_live_preview_registry.v1",
            "multimodal_multi_preview_registry.v1",
            "multimodal_interactive_canvas_registry.v1",
            "multimodal_visual_workspace_registry.v1",
            "multimodal_runtime_collaboration_registry.v1",
            "multimodal_artifact_collaboration_registry.v1",
            "multimodal_artifact_provenance_registry.v1",
            "multimodal_artifact_lineage_registry.v1",
            "multimodal_cross_agent_workspace_registry.v1",
            "multimodal_shared_artifact_board_registry.v1",
            "multimodal_workspace_history_registry.v1",
            "multimodal_branching_timeline_registry.v1",
            "multimodal_creative_evolution_timeline_registry.v1",
            "multimodal_real_time_workflow_visualization_registry.v1",
            "multimodal_studio_integration_registry.v1",
            "agent_contract_audit_registry.v1",
            "escalation_policy_audit_registry.v1",
            "hybrid_workflow_audit_registry.v1",
            "agent_registry_audit_registry.v1",
            "blackboard_audit_registry.v1",
            "shared_context_audit_registry.v1",
            "agent_collaboration_audit_registry.v1",
            "creative_diversity_audit_registry.v1",
            "agent_explainability_audit_registry.v1",
            "agent_reliability_audit_registry.v1",
            "agent_determinism_audit_registry.v1",
            "agent_telemetry_foundation_registry.v1",
            "agent_cost_tracking_foundation_registry.v1",
            "agent_performance_tracking_foundation_registry.v1",
            "architecture_consistency_pass_registry.v1",
            "final_v4_hardening_registry.v1",
            "langgraph_error_path_audit.v1",
            "model_routing_plan.v1",
            "local_cloud_routing_plan.v1",
            "hybrid_routing_plan.v1",
            "quality_cost_optimization_plan.v1",
            "cost_estimation_plan.v1",
            "budget_policy_plan.v1",
            "hitl_budget_gate_plan.v1",
            "runtime_recommendation_plan.v1",
            "execution_policy_plan.v1",
            "model_recommendation_plan.v1",
            "model_capability_matrix.v1",
            "provider_capability_matrix.v1",
            "quality_prediction_plan.v1",
            "cost_prediction_plan.v1",
            "creative_quality_prediction.v1",
            "creative_diversity_prediction_plan.v1",
            "creative_consistency_prediction_plan.v1",
            "routing_explainability_plan.v1",
            "model_routing_architecture_consistency_registry.v1",
            "model_routing_failure_path_audit_registry.v1",
            "token_dashboard.v1",
            "cost_dashboard.v1",
            "quality_dashboard.v1",
            "performance_dashboard.v1",
            "production_telemetry.v1",
            "workflow_diagnostics.v1",
            "agent_diagnostics.v1",
            "routing_diagnostics.v1",
            "escalation_diagnostics.v1",
            "failure_analysis.v1",
            "error_intelligence.v1",
            "workflow_health_monitoring.v1",
            "system_health_monitoring.v1",
            "creative_analytics.v1",
            "confidence_analytics.v1",
            "creative_diversity_analytics.v1",
            "runtime_timeline.v1",
            "workflow_explainability_dashboard.v1",
            "production_observability_architecture_registry.v1",
            "production_observability_failure_path_audit_registry.v1",
            "adaptive_hybrid_workflow_optimization_plan.v1",
            "adaptive_escalation_optimization_plan.v1",
            "agent_activation_optimization_plan.v1",
            "adaptive_cost_quality_plan.v1",
            "adaptive_latency_plan.v1",
            "adaptive_execution_strategy_selection_plan.v1",
            "dynamic_agent_allocation_plan.v1",
            "dynamic_resource_allocation_plan.v1",
            "workflow_self_tuning_policy_plan.v1",
            "execution_confidence_plan.v1",
            "workflow_risk_plan.v1",
            "creative_exploration_optimization_plan.v1",
            "emergence_optimization_plan.v1",
            "agent_diversity_optimization_plan.v1",
            "reflection_budget_optimization_plan.v1",
            "adaptive_policy_explainability_plan.v1",
            "adaptive_execution_architecture_consistency_registry.v1",
            "adaptive_execution_failure_path_audit_registry.v1",
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
            "agent_contract_audit.py",
            "escalation_policy_audit.py",
            "hybrid_workflow_audit.py",
            "agent_registry_audit.py",
            "blackboard_audit.py",
            "shared_context_audit.py",
            "agent_collaboration_audit.py",
            "creative_diversity_audit.py",
            "agent_explainability_audit.py",
            "agent_reliability_audit.py",
            "agent_determinism_audit.py",
            "agent_telemetry_foundation.py",
            "agent_cost_tracking_foundation.py",
            "agent_performance_tracking_foundation.py",
            "architecture_consistency_pass.py",
            "final_v4_hardening.py",
            "hybrid_studio.py",
            "multimodal_studio.py",
            "model_router.py",
            "local_cloud_routing.py",
            "hybrid_routing.py",
            "quality_cost_optimizer.py",
            "cost_estimator.py",
            "budget_policies.py",
            "hitl_budget_gate.py",
            "runtime_recommendation_engine.py",
            "execution_policy_engine.py",
            "model_recommendation_engine.py",
            "model_capability_matrix.py",
            "provider_capability_matrix.py",
            "quality_prediction_engine.py",
            "cost_prediction_engine.py",
            "creative_quality_prediction.py",
            "creative_diversity_predictor.py",
            "creative_consistency_predictor.py",
            "routing_explainability.py",
            "model_routing_architecture_consistency.py",
            "model_routing_failure_path_audit.py",
            "token_dashboard.py",
            "cost_dashboard.py",
            "quality_dashboard.py",
            "performance_dashboard.py",
            "production_telemetry.py",
            "workflow_diagnostics.py",
            "agent_diagnostics.py",
            "routing_diagnostics.py",
            "escalation_diagnostics.py",
            "failure_analysis.py",
            "error_intelligence.py",
            "workflow_health_monitoring.py",
            "system_health_monitoring.py",
            "creative_analytics.py",
            "confidence_analytics.py",
            "creative_diversity_analytics.py",
            "runtime_timeline.py",
            "workflow_explainability_dashboard.py",
            "production_observability_architecture_consistency.py",
            "production_observability_failure_path_audit.py",
            "adaptive_hybrid_workflow_optimizer.py",
            "adaptive_escalation_optimizer.py",
            "agent_activation_optimizer.py",
            "adaptive_cost_quality_optimizer.py",
            "adaptive_latency_optimizer.py",
            "adaptive_execution_strategy_selection.py",
            "dynamic_agent_allocation.py",
            "dynamic_resource_allocation.py",
            "workflow_self_tuning_policies.py",
            "execution_confidence_engine.py",
            "workflow_risk_engine.py",
            "creative_exploration_optimizer.py",
            "emergence_optimizer.py",
            "agent_diversity_optimizer.py",
            "reflection_budget_optimizer.py",
            "adaptive_policy_explainability.py",
            "adaptive_execution_architecture_consistency.py",
            "adaptive_execution_failure_path_audit.py",
        ):
            self.assertIn(module_path, engine_matrix)

        self.assertIn("export-only metadata surfaces", engine_matrix)
        self.assertIn("not active multi-agent orchestration", normalized_engine_matrix)
        self.assertIn("passive hybrid workflow metadata", normalized_engine_matrix)
        self.assertIn("passive hybrid studio metadata", normalized_engine_matrix)
        self.assertIn("passive multimodal studio metadata", normalized_engine_matrix)
        self.assertIn(
            "passive agentic studio hardening metadata",
            normalized_engine_matrix,
        )
        self.assertIn("advisory model-routing metadata", normalized_engine_matrix)
        self.assertIn(
            "read-only production observability metadata",
            normalized_engine_matrix,
        )
        self.assertIn(
            "advisory adaptive execution metadata",
            normalized_engine_matrix,
        )
        self.assertIn("does not activate Studio runtime", normalized_engine_matrix)
        self.assertIn("does not execute rendering", normalized_engine_matrix)
        self.assertIn("does not execute hardening checks", normalized_engine_matrix)
        self.assertIn("bypass failure normalization", normalized_engine_matrix)
        self.assertIn("activate passive registries", normalized_engine_matrix)
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
        self.assertIn("do not apply routing", normalized_engine_matrix)
        self.assertIn("switching models", normalized_engine_matrix)
        self.assertIn("emitting HITL requests", normalized_engine_matrix)
        self.assertIn("collect live metrics", normalized_engine_matrix)
        self.assertIn("emit telemetry or alerts", normalized_engine_matrix)
        self.assertIn("capture traces", normalized_engine_matrix)
        self.assertIn("apply adaptive policies or strategies", normalized_engine_matrix)
        self.assertIn("adaptive behavior application", normalized_engine_matrix)

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
            "V5 advisory adaptive execution metadata",
            architecture_doc,
        )
        self.assertIn("apply adaptive execution policy", normalized_architecture_doc)
        self.assertIn(
            "does not add provider routing, runtime selection, execution "
            "optimization, artifact execution, artifact modification, "
            "autonomous retries, preview execution, or generated output changes",
            normalized_architecture_doc,
        )


if __name__ == "__main__":
    unittest.main()
