import unittest

from creative_coding_assistant.orchestration import (
    CognitiveExplanationEnginePlan,
    build_cognitive_blackboard,
    build_cognitive_explanation_engine,
    cognitive_explanation_trace_by_id,
    cognitive_explanation_traces_for_agent,
    cognitive_explanation_traces_for_layer,
    cognitive_explanation_traces_for_posture,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class CognitiveExplanationEngineTests(unittest.TestCase):
    def test_cognitive_explanation_engine_builds_read_only_traces(self) -> None:
        blackboard = build_cognitive_blackboard()
        engine = build_cognitive_explanation_engine(cognitive_blackboard=blackboard)

        self.assertEqual(engine.role, "cognitive_explanation_engine")
        self.assertEqual(
            engine.serialization_version,
            "cognitive_explanation_engine.v1",
        )
        self.assertEqual(engine.cognitive_blackboard_role, blackboard.role)
        self.assertEqual(
            engine.cognitive_blackboard_serialization_version,
            blackboard.serialization_version,
        )
        self.assertEqual(engine.cognitive_router_role, blackboard.cognitive_router_role)
        self.assertEqual(
            engine.cognitive_planner_role,
            blackboard.cognitive_planner_role,
        )
        self.assertEqual(
            engine.cognitive_scheduler_role,
            blackboard.cognitive_scheduler_role,
        )
        self.assertEqual(engine.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(engine.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(engine.capability_ids, blackboard.capability_ids)
        self.assertEqual(engine.capability_count, 6)
        self.assertEqual(
            engine.source_blackboard_entry_ids,
            blackboard.blackboard_entry_ids,
        )
        self.assertEqual(engine.source_blackboard_entry_count, 6)
        self.assertEqual(
            engine.source_route_decision_ids,
            blackboard.source_route_decision_ids,
        )
        self.assertEqual(engine.source_route_decision_count, 6)
        self.assertEqual(engine.source_plan_ids, blackboard.source_plan_ids)
        self.assertEqual(engine.source_plan_count, 6)
        self.assertEqual(
            engine.source_schedule_ids,
            blackboard.source_schedule_ids,
        )
        self.assertEqual(engine.source_schedule_count, 6)
        self.assertEqual(
            engine.source_emergence_ids,
            blackboard.source_emergence_ids,
        )
        self.assertEqual(engine.source_emergence_count, 6)
        self.assertEqual(len(engine.explanation_traces), 6)
        self.assertEqual(engine.explanation_count, 6)
        self.assertEqual(engine.candidate_explanation_count, 0)
        self.assertEqual(engine.review_required_explanation_count, 0)
        self.assertEqual(engine.guarded_explanation_count, 6)
        self.assertEqual(engine.linked_agent_ids, blackboard.linked_agent_ids)
        self.assertEqual(
            engine.covered_roadmap_items,
            ("Cognitive Explanation Engine",),
        )
        self.assertEqual(engine.covered_roadmap_item_count, 1)
        self.assertEqual(engine.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            engine.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(engine.cognitive_explanation_engine_implemented)
        self.assertTrue(engine.cognitive_blackboard_integrated)
        self.assertTrue(engine.explanation_trace_contract_implemented)
        self.assertTrue(engine.explanation_dependency_traceability_implemented)
        self.assertTrue(engine.explanation_governance_contract_implemented)
        self.assertTrue(engine.explanation_hitl_contract_implemented)
        self.assertFalse(engine.explanation_generation_implemented)
        self.assertFalse(engine.explanation_text_generation_implemented)
        self.assertFalse(engine.explanation_audit_record_write_implemented)
        self.assertFalse(engine.prompt_mutation_implemented)
        self.assertFalse(engine.memory_mutation_implemented)
        self.assertFalse(engine.retrieval_mutation_implemented)
        self.assertFalse(engine.storage_mutation_implemented)
        self.assertFalse(engine.provider_model_routing_implemented)
        self.assertFalse(engine.provider_execution_implemented)
        self.assertFalse(engine.generated_output_mutation_implemented)
        self.assertFalse(engine.runtime_evolution_implemented)
        self.assertFalse(engine.generated_explanation_ids)
        self.assertFalse(engine.written_explanation_record_ids)
        self.assertFalse(engine.mutated_explanation_ids)
        self.assertFalse(engine.emitted_hitl_request_ids)
        self.assertTrue(engine.advisory_only)

    def test_cognitive_explanation_lookup_helpers_are_scope_aware(self) -> None:
        engine = build_cognitive_explanation_engine()

        core_trace = cognitive_explanation_trace_by_id(
            "cognitive_explanation::v6_6_cognitive_core",
            engine,
        )
        self.assertIsNotNone(core_trace)
        assert core_trace is not None
        self.assertEqual(core_trace.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_trace.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_trace.linked_agent_ids)
        self.assertEqual(core_trace.explanation_rank, 6)
        self.assertEqual(core_trace.dependency_depth, 5)
        self.assertEqual(
            core_trace.source_trace_ids[0],
            "cognitive_blackboard::v6_6_cognitive_core",
        )
        self.assertIn(
            "cognitive_router::v6_6_cognitive_core",
            core_trace.source_trace_ids,
        )
        self.assertIn(
            "cognitive_planner::v6_6_cognitive_core",
            core_trace.source_trace_ids,
        )
        self.assertIn(
            "cognitive_scheduler::v6_6_cognitive_core",
            core_trace.source_trace_ids,
        )
        self.assertFalse(core_trace.explanation_generation_authorized)
        self.assertIn("generate explanations", core_trace.governance_contracts[0])

        research_traces = cognitive_explanation_traces_for_layer("research", engine)
        self.assertEqual(len(research_traces), 1)
        self.assertEqual(
            research_traces[0].capability_id,
            "v6_4_autonomous_research",
        )

        planner_traces = cognitive_explanation_traces_for_agent(
            "planner_agent",
            engine,
        )
        self.assertEqual(
            tuple(trace.capability_id for trace in planner_traces),
            ("v6_6_cognitive_core",),
        )
        guarded_traces = cognitive_explanation_traces_for_posture(
            "guarded",
            engine,
        )
        self.assertEqual(
            tuple(trace.explanation_id for trace in guarded_traces),
            engine.guarded_explanation_ids,
        )
        self.assertIsNone(cognitive_explanation_trace_by_id("missing", engine))

    def test_cognitive_explanation_engine_rejects_generation_and_drift(self) -> None:
        engine = build_cognitive_explanation_engine()
        payload = engine.model_dump(mode="json")
        payload["explanation_ids"] = ("missing",) + tuple(
            payload["explanation_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "explanation_ids must match"):
            CognitiveExplanationEnginePlan(**payload)

        payload = engine.model_dump(mode="json")
        payload["generated_explanation_ids"] = (
            "cognitive_explanation::v6_6_cognitive_core",
        )

        with self.assertRaisesRegex(
            ValueError,
            "explanation generation, writes, mutation, and HITL ids must be empty",
        ):
            CognitiveExplanationEnginePlan(**payload)

    def test_cognitive_explanation_engine_reuses_supplied_blackboard(self) -> None:
        blackboard = build_cognitive_blackboard(route="generate")
        engine = build_cognitive_explanation_engine(cognitive_blackboard=blackboard)

        self.assertEqual(engine.route_name, blackboard.route_name)
        self.assertEqual(engine.task_type, blackboard.task_type)
        self.assertEqual(
            engine.source_blackboard_entry_ids,
            blackboard.blackboard_entry_ids,
        )
        self.assertEqual(
            engine.source_route_decision_ids,
            blackboard.source_route_decision_ids,
        )
        self.assertEqual(engine.source_plan_ids, blackboard.source_plan_ids)
        self.assertEqual(
            engine.source_schedule_ids,
            blackboard.source_schedule_ids,
        )
        self.assertEqual(
            engine.source_emergence_ids,
            blackboard.source_emergence_ids,
        )


if __name__ == "__main__":
    unittest.main()
