from __future__ import annotations

import importlib
import unittest
from pathlib import Path

from creative_coding_assistant.orchestration.runtime.graph_builder import (
    build_assistant_workflow_graph,
)
from creative_coding_assistant.orchestration.runtime.nodes.registry import (
    registered_workflow_conditional_edge_specs,
    registered_workflow_node_specs,
)
from creative_coding_assistant.orchestration.workflow_graph import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class WorkflowRuntimeDecompositionTests(unittest.TestCase):
    def test_workflow_graph_imports_remain_compatible_shims(self) -> None:
        legacy_module = importlib.import_module(
            "creative_coding_assistant.orchestration.workflow_graph"
        )
        runtime_module = importlib.import_module(
            "creative_coding_assistant.orchestration.runtime.workflow_graph"
        )
        handler_module = importlib.import_module(
            "creative_coding_assistant.orchestration.runtime.nodes.handlers"
        )

        self.assertIs(legacy_module, runtime_module)
        self.assertIs(runtime_module, handler_module)

    def test_registered_node_handlers_preserve_order_and_runtime_ownership(
        self,
    ) -> None:
        node_specs = registered_workflow_node_specs()
        expected_modules = {
            "intake": "creative_coding_assistant.orchestration.runtime.nodes.intake",
            "routing": "creative_coding_assistant.orchestration.runtime.nodes.routing",
            "memory": "creative_coding_assistant.orchestration.runtime.nodes.memory",
            "retrieval": (
                "creative_coding_assistant.orchestration.runtime.nodes.retrieval"
            ),
            "context_assembly": (
                "creative_coding_assistant.orchestration.runtime.nodes.context"
            ),
            "prompt_input": (
                "creative_coding_assistant.orchestration.runtime.nodes.context"
            ),
            "planning": "creative_coding_assistant.orchestration.runtime.nodes.planning",
            "director": "creative_coding_assistant.orchestration.runtime.nodes.planning",
            "reasoning": "creative_coding_assistant.orchestration.runtime.nodes.planning",
            "prompt_rendering": (
                "creative_coding_assistant.orchestration.runtime.nodes.generation"
            ),
            "generation": (
                "creative_coding_assistant.orchestration.runtime.nodes.generation"
            ),
            "artifact_extraction": (
                "creative_coding_assistant.orchestration.runtime.nodes.artifacts"
            ),
            "preview_preparation": (
                "creative_coding_assistant.orchestration.runtime.nodes.artifacts"
            ),
            "artifact_critique": (
                "creative_coding_assistant.orchestration.runtime.nodes.artifacts"
            ),
            "review": "creative_coding_assistant.orchestration.runtime.nodes.review",
            "refinement": (
                "creative_coding_assistant.orchestration.runtime.nodes.refinement"
            ),
            "finalization": (
                "creative_coding_assistant.orchestration.runtime.nodes.finalization"
            ),
            "failure": (
                "creative_coding_assistant.orchestration.runtime.nodes.finalization"
            ),
        }

        self.assertEqual(
            tuple(spec.name for spec in node_specs),
            ASSISTANT_WORKFLOW_NODE_ORDER,
        )
        self.assertEqual(
            tuple(spec.handler.__module__ for spec in node_specs),
            tuple(expected_modules[node] for node in ASSISTANT_WORKFLOW_NODE_ORDER),
        )
        self.assertEqual(
            tuple(spec.handler.__name__ for spec in node_specs),
            tuple(f"_{node}_node" for node in ASSISTANT_WORKFLOW_NODE_ORDER),
        )

    def test_representative_node_handlers_import_from_canonical_modules(
        self,
    ) -> None:
        handler_module = importlib.import_module(
            "creative_coding_assistant.orchestration.runtime.nodes.handlers"
        )
        planning_module = importlib.import_module(
            "creative_coding_assistant.orchestration.runtime.nodes.planning"
        )
        artifact_module = importlib.import_module(
            "creative_coding_assistant.orchestration.runtime.nodes.artifacts"
        )
        finalization_module = importlib.import_module(
            "creative_coding_assistant.orchestration.runtime.nodes.finalization"
        )

        self.assertIs(handler_module._planning_node, planning_module._planning_node)
        self.assertIs(
            handler_module._artifact_critique_node,
            artifact_module._artifact_critique_node,
        )
        self.assertIs(
            handler_module._finalization_node,
            finalization_module._finalization_node,
        )

    def test_registered_edges_preserve_failure_targets_and_topology(self) -> None:
        edge_specs = registered_workflow_conditional_edge_specs()
        edge_specs_by_source = {spec.source: spec for spec in edge_specs}

        self.assertEqual(tuple(edge_specs_by_source), ASSISTANT_WORKFLOW_NODE_ORDER[:-1])
        self.assertNotIn("failure", edge_specs_by_source)
        self.assertEqual(
            edge_specs_by_source["prompt_input"].targets,
            {
                "planning": "planning",
                "finalization": "finalization",
                "failure": "failure",
            },
        )
        self.assertEqual(
            edge_specs_by_source["review"].targets,
            {
                "finalization": "finalization",
                "refinement": "refinement",
                "failure": "failure",
            },
        )
        self.assertEqual(
            edge_specs_by_source["finalization"].targets,
            {
                "end": "__end__",
                "failure": "failure",
            },
        )
        for node in ASSISTANT_WORKFLOW_NODE_ORDER[:-1]:
            with self.subTest(node=node):
                self.assertEqual(edge_specs_by_source[node].targets["failure"], "failure")

    def test_graph_builder_owns_langgraph_construction(self) -> None:
        graph = build_assistant_workflow_graph()

        self.assertTrue(hasattr(graph, "stream"))
        self.assertTrue(hasattr(graph, "invoke"))

        workflow_graph_source = (
            REPO_ROOT
            / "src"
            / "creative_coding_assistant"
            / "orchestration"
            / "runtime"
            / "workflow_graph.py"
        ).read_text(encoding="utf-8")
        graph_builder_source = (
            REPO_ROOT
            / "src"
            / "creative_coding_assistant"
            / "orchestration"
            / "runtime"
            / "graph_builder.py"
        ).read_text(encoding="utf-8")
        registry_source = (
            REPO_ROOT
            / "src"
            / "creative_coding_assistant"
            / "orchestration"
            / "runtime"
            / "nodes"
            / "registry.py"
        ).read_text(encoding="utf-8")

        self.assertIn("nodes.handlers", workflow_graph_source)
        self.assertNotIn("def _planning_node", workflow_graph_source)
        self.assertIn("StateGraph", graph_builder_source)
        self.assertIn("registered_workflow_node_specs", graph_builder_source)
        self.assertIn("transitions.next_node_after_review", registry_source)


if __name__ == "__main__":
    unittest.main()
