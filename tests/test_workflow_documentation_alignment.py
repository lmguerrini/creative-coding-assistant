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
CAPABILITY_CLOSEOUT_STEPS = (
    "Architecture Update",
    "Documentation Update",
    "Junie Engineering Review",
    "ChatGPT review of Junie report",
    "Codex Review Fixes if approved",
    "Engineering Validation",
    "Create Version Tag",
    "Merge & Push",
)


class WorkflowDocumentationAlignmentTests(unittest.TestCase):
    def test_readme_workflow_order_matches_backend_node_order(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        expected_order = " -> ".join(ASSISTANT_WORKFLOW_NODE_ORDER)

        self.assertIn(f"`{expected_order}`", readme)

    def test_readme_covers_v32_docs_closeout_and_roadmap(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("V3.2 AI-native creative translation", readme)
        self.assertIn("feature/generative-design-core", readme)
        self.assertIn("V3.2 Generative Design Core", readme)
        self.assertIn("architecture/engine_matrix.md", readme)

        for capability in V32_GENERATIVE_DESIGN_CAPABILITIES:
            self.assertIn(capability, readme)

        for step in CAPABILITY_CLOSEOUT_STEPS:
            self.assertIn(step, readme)

        self.assertIn(
            "ChatGPT is architect, planner, and reviewer of Junie reports.",
            readme,
        )
        self.assertIn("Codex is the implementation tool.", readme)
        self.assertIn("Junie is the independent engineering reviewer.", readme)
        self.assertIn("Git and GitHub are the delivery source of truth.", readme)
        self.assertIn("V4: Agentic Studio", readme)
        self.assertIn(
            "V5: Execution Optimization & Production Intelligence",
            readme,
        )
        self.assertIn("V6: Learning & Evolution", readme)

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

        self.assertIn(
            "## Runtime Graph Vs Internal Capability Graph",
            architecture_doc,
        )
        self.assertIn("creative_intelligence_graph.md", architecture_doc)
        self.assertIn("creative_intelligence_graph.mmd", architecture_doc)
        self.assertIn("generative_design_graph.md", architecture_doc)
        self.assertIn("generative_design_graph.mmd", architecture_doc)
        self.assertIn("metadata and design guidance", architecture_doc)
        self.assertIn("not code generation execution", architecture_doc)
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
            "V4",
            "Agentic Studio",
            "V5",
            "Execution Optimization & Production Intelligence",
            "V6",
            "Learning & Evolution",
        ):
            self.assertIn(version_marker, engine_matrix)


if __name__ == "__main__":
    unittest.main()
