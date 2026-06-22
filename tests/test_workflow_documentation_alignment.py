import re
import unittest
from pathlib import Path

from creative_coding_assistant.orchestration import ASSISTANT_WORKFLOW_NODE_ORDER

REPO_ROOT = Path(__file__).resolve().parents[1]
CREATIVE_COGNITION_CORE_CAPABILITIES = (
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
    "Creative Reasoning Engine",
)


class WorkflowDocumentationAlignmentTests(unittest.TestCase):
    def test_readme_workflow_order_matches_backend_node_order(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        expected_order = " -> ".join(ASSISTANT_WORKFLOW_NODE_ORDER)

        self.assertIn(f"`{expected_order}`", readme)

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

    def test_workflow_doc_distinguishes_runtime_and_internal_graphs(self) -> None:
        architecture_doc = (
            REPO_ROOT / "architecture" / "workflow_graph.md"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "## Runtime Graph Vs Internal Capability Graph",
            architecture_doc,
        )
        self.assertIn("creative_intelligence_graph.md", architecture_doc)
        self.assertIn("creative_intelligence_graph.mmd", architecture_doc)
        self.assertIn("true multi-agent", architecture_doc)
        self.assertIn("multi-node runtime graph", architecture_doc)

    def test_creative_intelligence_graph_docs_cover_current_capabilities(self) -> None:
        architecture_doc = (
            REPO_ROOT / "architecture" / "creative_intelligence_graph.md"
        ).read_text(encoding="utf-8")
        mermaid = (
            REPO_ROOT / "architecture" / "creative_intelligence_graph.mmd"
        ).read_text(encoding="utf-8")

        for capability in CREATIVE_COGNITION_CORE_CAPABILITIES:
            self.assertIn(capability, architecture_doc)
            self.assertIn(capability, mermaid)

        self.assertIn("Creative Execution Plan", architecture_doc)
        self.assertIn("Creative Execution Plan", mermaid)
        self.assertIn("Creative Assistant Director runtime node", architecture_doc)
        self.assertIn("Creative Reasoning Engine runtime node", architecture_doc)
        self.assertIn("prompt_rendering", mermaid)
        self.assertIn(
            "Why This Is Not Yet A True Multi-Agent Graph",
            architecture_doc,
        )
        self.assertIn("metadata_store --> director", mermaid)
        self.assertIn("director --> reasoning --> prompt_rendering", mermaid)


if __name__ == "__main__":
    unittest.main()
