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
        self.assertIn("Next.js workstation", readme)
        self.assertIn("Capability Scope", readme)
        self.assertIn("architecture/artifact_intelligence_graph.md", readme)
        self.assertIn("architecture/workstation_surface_graph.md", readme)
        self.assertIn("architecture/engine_matrix.md", readme)
        self.assertIn("Creative Evaluation", readme)
        self.assertIn("Workstation Dashboard", readme)
        self.assertIn("Creative Timeline", readme)
        self.assertIn("Provenance Engine", readme)
        self.assertIn("metadata-only", readme)
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
