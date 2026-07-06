from __future__ import annotations

import importlib
import unittest
from pathlib import Path

from creative_coding_assistant.orchestration.runtime.nodes.planning_contracts import (
    PLANNING_EVENT_PAYLOAD_FIELDS,
    PLANNING_RUNTIME_UPDATE_FIELDS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PLANNING_UPDATE_FIELDS = (
    "creative_strategy",
    "creative_intent",
    "creative_hierarchy",
    "creative_techniques",
    "creative_plan",
    "creative_constraints",
    "creative_constraint_priorities",
    "runtime_capabilities",
    "creative_tradeoffs",
    "creative_quality_prediction",
    "symbolic_narrative",
    "creative_composition",
    "procedural_structure",
    "generative_structure",
    "semantic_motif",
    "emotional_consistency",
    "cross_modality",
    "audio_visual_scene",
    "artifact_plan",
    "artifact_dependency_graph",
    "runtime_compatibility",
    "artifact_capability_matrix",
    "multi_artifact_strategy",
    "artifact_critic",
    "artifact_refiner",
    "artifact_intelligence_synthesis",
    "artifact_merge_planner",
    "artifact_export_intelligence",
    "artifact_engine_contracts",
    "evaluation_engine_contracts",
    "creative_critic",
    "self_evaluation",
    "creative_improvement_planner",
    "reflection_loop",
    "creative_confidence",
    "creative_score",
    "consistency_validation",
    "evaluation_report",
)

EXPECTED_PLANNING_EVENT_FIELDS = (
    "creative_intent",
    "creative_hierarchy",
    "creative_strategy",
    "creative_techniques",
    "creative_plan",
    "creative_constraints",
    "creative_constraint_priorities",
    "runtime_capabilities",
    "creative_tradeoffs",
    "creative_quality_prediction",
    "symbolic_narrative",
    "creative_composition",
    "procedural_structure",
    "generative_structure",
    "semantic_motif",
    "emotional_consistency",
    "cross_modality",
    "audio_visual_scene",
    "artifact_plan",
    "artifact_dependency_graph",
    "runtime_compatibility",
    "artifact_capability_matrix",
    "multi_artifact_strategy",
    "artifact_critic",
    "artifact_refiner",
    "artifact_intelligence_synthesis",
    "artifact_merge_planner",
    "artifact_export_intelligence",
    "artifact_engine_contracts",
    "evaluation_engine_contracts",
    "creative_critic",
    "self_evaluation",
    "creative_improvement_planner",
    "reflection_loop",
    "creative_confidence",
    "creative_score",
    "consistency_validation",
    "evaluation_report",
)


class PlanningRuntimeDecompositionTests(unittest.TestCase):
    def test_planning_compatibility_facade_reexports_focused_modules(self) -> None:
        planning_facade = importlib.import_module(
            "creative_coding_assistant.orchestration.runtime.nodes.planning"
        )
        planning_node = importlib.import_module(
            "creative_coding_assistant.orchestration.runtime.nodes.planning_node"
        )
        director = importlib.import_module(
            "creative_coding_assistant.orchestration.runtime.nodes.director"
        )
        reasoning = importlib.import_module(
            "creative_coding_assistant.orchestration.runtime.nodes.reasoning"
        )
        planning_contracts = importlib.import_module(
            "creative_coding_assistant.orchestration.runtime.nodes.planning_contracts"
        )

        self.assertIs(planning_facade._planning_node, planning_node._planning_node)
        self.assertIs(planning_facade._director_node, director._director_node)
        self.assertIs(planning_facade._reasoning_node, reasoning._reasoning_node)
        self.assertIs(
            planning_facade._evaluation_planning_metadata,
            planning_contracts._evaluation_planning_metadata,
        )

    def test_planning_contract_field_order_preserves_runtime_payloads(self) -> None:
        self.assertEqual(
            PLANNING_RUNTIME_UPDATE_FIELDS,
            EXPECTED_PLANNING_UPDATE_FIELDS,
        )
        self.assertEqual(
            PLANNING_EVENT_PAYLOAD_FIELDS,
            EXPECTED_PLANNING_EVENT_FIELDS,
        )

    def test_planning_facade_no_longer_owns_live_handler_implementations(self) -> None:
        node_dir = (
            REPO_ROOT
            / "src"
            / "creative_coding_assistant"
            / "orchestration"
            / "runtime"
            / "nodes"
        )
        planning_facade_source = (node_dir / "planning.py").read_text(encoding="utf-8")
        planning_node_source = (node_dir / "planning_node.py").read_text(
            encoding="utf-8"
        )
        director_source = (node_dir / "director.py").read_text(encoding="utf-8")
        reasoning_source = (node_dir / "reasoning.py").read_text(encoding="utf-8")

        self.assertNotIn("def _planning_node", planning_facade_source)
        self.assertIn("def _planning_node", planning_node_source)
        self.assertIn("def _director_node", director_source)
        self.assertIn("def _reasoning_node", reasoning_source)


if __name__ == "__main__":
    unittest.main()
