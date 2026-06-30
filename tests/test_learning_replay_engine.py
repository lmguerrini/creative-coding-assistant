import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    LearningReplayPlan,
    build_learning_replay_engine,
    learning_replay_scenario_by_id,
    learning_replay_scenarios_for_confidence,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")


class LearningReplayEngineTests(unittest.TestCase):
    def test_plan_builds_replay_scenarios_without_execution(self) -> None:
        plan = build_learning_replay_engine(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "learning_replay_engine")
        self.assertEqual(plan.serialization_version, "learning_replay_plan.v1")
        self.assertEqual(
            plan.source_adaptive_learning_serialization_version,
            "adaptive_learning_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.scenario_count, 5)
        self.assertEqual(plan.overall_replay_posture, "guarded")
        self.assertIn("does not execute learning replay", plan.authority_boundary)
        self.assertTrue(plan.learning_replay_engine_implemented)
        self.assertTrue(plan.replay_metadata_implemented)
        self.assertTrue(plan.adaptive_learning_metadata_used)
        self.assertFalse(plan.replay_executed)
        self.assertFalse(plan.workflow_replay_executed)
        self.assertFalse(plan.provider_calls_executed)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_scenarios_score_replay_confidence_without_replay(self) -> None:
        plan = build_learning_replay_engine(route="generate")

        for scenario in plan.scenarios:
            expected_score = min(
                1000,
                max(0, scenario.learning_priority_score // 2 + scenario.replay_weight),
            )
            self.assertEqual(scenario.replay_confidence_score, expected_score)
            self.assertIn(
                "workflow_replay_execution",
                scenario.blocked_runtime_behaviors,
            )
            self.assertIn(
                scenario.source_learning_signal_id,
                " ".join(scenario.evidence),
            )
            self.assertTrue(scenario.expected_replay_insight)
            self.assertTrue(scenario.replay_safety_boundary)
            self.assertFalse(scenario.replay_executed)
            self.assertFalse(scenario.workflow_replay_executed)
            self.assertFalse(scenario.provider_calls_executed)
            self.assertFalse(scenario.persistent_storage_write_implemented)
            self.assertFalse(scenario.generated_output_mutation_implemented)
            self.assertFalse(scenario.runtime_evolution_implemented)

        guarded = learning_replay_scenarios_for_confidence("guarded", plan)
        routing = learning_replay_scenario_by_id(
            "learning_replay::routing_boundary_replay",
            plan,
        )
        self.assertGreaterEqual(len(guarded), 1)
        self.assertIsNotNone(routing)
        assert routing is not None
        self.assertEqual(routing.replay_confidence, "guarded")

    def test_plan_rejects_mismatched_replay_metadata(self) -> None:
        plan = build_learning_replay_engine()
        payload = plan.model_dump(mode="json")
        payload["scenario_ids"] = ("missing",) + tuple(payload["scenario_ids"][1:])

        with self.assertRaisesRegex(ValueError, "scenario_ids must match"):
            LearningReplayPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_replay_confidence_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_replay_confidence_score must match",
        ):
            LearningReplayPlan(**payload)

    def test_replay_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review learning replay for a creative workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
        )

        build_learning_replay_engine(route=RouteName.GENERATE)
        provider = build_generation_provider(settings)
        after_decision = route_request(request)

        self.assertEqual(after_decision, baseline_decision)
        self.assertIsInstance(provider, OpenAIGenerationProvider)

    def test_replay_metadata_does_not_declare_active_terms(self) -> None:
        plan = build_learning_replay_engine(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for scenario in plan.scenarios
                    for field in (
                        scenario.scenario_id,
                        scenario.expected_replay_insight,
                        scenario.replay_safety_boundary,
                        *scenario.advisory_actions,
                        *scenario.evidence,
                        *scenario.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_workflow_replay(",
            "call_provider(",
            "write_learning_replay(",
            "mutate_generated_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
