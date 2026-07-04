import unittest

from creative_coding_assistant.contracts import StreamEventType
from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    build_assistant_workflow_graph,
    build_rendered_prompt_request,
    creative_improvement_planner_prompt_lines,
    derive_creative_assistant_director_brief,
    derive_creative_improvement_planner_profile,
    derive_creative_reasoning_result,
    derive_self_evaluation_profile,
    stream_assistant_workflow_events,
)
from test_creative_critic_engine import _stack
from test_langgraph_workflow_integration import (
    _first_event,
    _request,
    _runtime,
    _stream_completed_generation,
    _stream_prompt_inputs_with_builder,
)


class CreativeImprovementPlannerTests(unittest.TestCase):
    def test_derives_improvement_priorities_from_evaluation_metadata(self) -> None:
        stack = _stack(
            "Generate a luminous p5.js mandala with improvement planner metadata."
        )
        self_evaluation = _self_evaluation(
            stack,
            generated_response="Here is a luminous p5.js mandala with caveats.",
        )
        planner = _planner(stack, self_evaluation)

        self.assertEqual(planner.role, "creative_improvement_planner")
        self.assertEqual(planner.serialization_version, "v1")
        self.assertGreater(planner.confidence, 0)
        self.assertTrue(planner.improvement_priorities)
        self.assertTrue(planner.highest_impact_opportunities)
        self.assertTrue(planner.low_risk_improvements)
        self.assertTrue(planner.trade_off_recommendations)
        self.assertIn("does not modify prompts", planner.authority_boundary)
        self.assertTrue(creative_improvement_planner_prompt_lines(planner))

    def test_prioritizes_high_risk_evaluation_signals(self) -> None:
        stack = _stack("Generate a risky artifact with unsupported assumptions.")
        base_evaluation = _self_evaluation(
            stack,
            generated_response=(
                "This production-ready artifact is guaranteed and tested across "
                "all runtimes with no caveats."
            ),
        )
        risky_evaluation = base_evaluation.model_copy(
            update={
                "hallucination_risk": "high",
                "underdelivery_risk": "high",
                "quality_gaps": (
                    "Unsupported certainty language needs explicit caveats.",
                ),
            }
        )
        planner = _planner(stack, risky_evaluation)

        self.assertEqual(planner.improvement_priorities[0].priority, "critical")
        self.assertTrue(planner.hitl_questions)
        self.assertTrue(planner.future_refinement_candidates)

    def test_derives_low_risk_and_experimental_improvements(self) -> None:
        stack = _stack("Generate an experimental p5.js mandala with clear caveats.")
        self_evaluation = _self_evaluation(
            stack,
            generated_response="Here is an experimental p5.js mandala with caveats.",
        )
        planner = _planner(stack, self_evaluation)

        self.assertTrue(planner.low_risk_improvements)
        self.assertTrue(planner.experimental_improvements)
        self.assertTrue(
            any(
                "optional" in item.lower() for item in planner.experimental_improvements
            ),
            planner.experimental_improvements,
        )

    def test_integrates_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack(
            "Generate a symbolic p5.js phoenix mandala with improvement planner metadata."
        )
        self_evaluation = _self_evaluation(
            stack,
            generated_response="Here is a symbolic p5.js phoenix mandala with caveats.",
        )
        planner = _planner(stack, self_evaluation)
        director = _director(stack, self_evaluation, planner)
        reasoning = _reasoning(stack, self_evaluation, planner, director)
        prompt_input = stack.runtime_base.prompt_input.model_copy(
            update={
                "creative_intent": stack.runtime_base.intent,
                "creative_hierarchy": stack.runtime_base.hierarchy,
                "creative_strategy": stack.runtime_base.strategy,
                "creative_techniques": stack.runtime_base.techniques,
                "creative_plan": stack.runtime_base.plan,
                "creative_constraints": stack.runtime_base.constraints,
                "creative_constraint_priorities": stack.runtime_base.prioritization,
                "runtime_capabilities": stack.runtime_base.runtime_capabilities,
                "creative_tradeoffs": stack.runtime_base.tradeoffs,
                "creative_quality_prediction": getattr(
                    stack.runtime_base,
                    "quality_prediction",
                    None,
                ),
                "artifact_plan": stack.runtime_base.artifact_plan,
                "artifact_dependency_graph": (
                    stack.runtime_base.artifact_dependency_graph
                ),
                "runtime_compatibility": stack.runtime_base.runtime_compatibility,
                "artifact_capability_matrix": stack.artifact_capability_matrix,
                "multi_artifact_strategy": stack.multi_artifact_strategy,
                "artifact_critic": stack.artifact_critic,
                "artifact_refiner": stack.artifact_refiner,
                "artifact_intelligence_synthesis": (
                    stack.artifact_intelligence_synthesis
                ),
                "artifact_merge_planner": stack.artifact_merge_planner,
                "artifact_export_intelligence": stack.artifact_export_intelligence,
                "creative_critic": stack.creative_critic,
                "self_evaluation": self_evaluation,
                "creative_improvement_planner": planner,
                "creative_director": director,
                "creative_reasoning": reasoning,
            }
        )
        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=stack.runtime_base.route,
                prompt_input=prompt_input,
            )
        )
        system = rendered.sections[0].content

        self.assertIn("Creative Improvement Planner:", system)
        self.assertIn("Improvement priority:", system)
        self.assertTrue(
            any(
                "Creative improvement planner:" in item
                for item in director.planning_focus
            ),
            director.planning_focus,
        )
        self.assertIn(
            "creative_improvement_planner",
            {item.source for item in reasoning.evidence_chain},
        )
        self.assertEqual(
            planner.model_dump(mode="json")["role"],
            "creative_improvement_planner",
        )
        self.assertNotIn("workflow loop enabled", system.lower())
        self.assertNotIn("runtime auto-selection enabled", system.lower())

    def test_workflow_and_final_payload_serialize_improvement_planner(self) -> None:
        graph = build_assistant_workflow_graph()
        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(
                    query=(
                        "Generate a p5.js sketch with creative improvement "
                        "planner metadata."
                    ),
                ),
                runtime=_runtime(
                    stream_prompt_inputs=_stream_prompt_inputs_with_builder,
                    stream_generation=_stream_completed_generation,
                ),
            )
        )

        planning_event = _first_event(
            events,
            StreamEventType.PLANNING,
            "creative_plan_prepared",
        )
        final_event = events[-1]
        planning_planner = planning_event.payload["creative_improvement_planner"]
        final_planner = final_event.payload["creative_improvement_planner"]

        self.assertEqual(planning_planner["role"], "creative_improvement_planner")
        self.assertTrue(
            planning_event.payload["workflow"]["creative_improvement_planner_available"]
        )
        self.assertEqual(
            planning_event.payload["workflow"]["creative_improvement_planner"],
            planning_planner,
        )
        self.assertEqual(final_planner["role"], "creative_improvement_planner")
        self.assertEqual(
            final_event.payload["workflow"]["creative_improvement_planner"],
            final_planner,
        )


def _self_evaluation(
    stack: object,
    *,
    generated_response: str | None = None,
):
    runtime_base = stack.runtime_base
    return derive_self_evaluation_profile(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        creative_translation=runtime_base.prompt_input.creative_translation,
        creative_intent=runtime_base.intent,
        creative_hierarchy=runtime_base.hierarchy,
        creative_plan=runtime_base.plan,
        creative_constraints=runtime_base.constraints,
        creative_constraint_priorities=runtime_base.prioritization,
        creative_strategy=runtime_base.strategy,
        creative_techniques=runtime_base.techniques,
        runtime_capabilities=runtime_base.runtime_capabilities,
        creative_tradeoffs=runtime_base.tradeoffs,
        creative_quality_prediction=getattr(runtime_base, "quality_prediction", None),
        artifact_plan=runtime_base.artifact_plan,
        artifact_dependency_graph=runtime_base.artifact_dependency_graph,
        runtime_compatibility=runtime_base.runtime_compatibility,
        artifact_capability_matrix=stack.artifact_capability_matrix,
        multi_artifact_strategy=stack.multi_artifact_strategy,
        artifact_critic=stack.artifact_critic,
        artifact_refiner=stack.artifact_refiner,
        artifact_intelligence_synthesis=stack.artifact_intelligence_synthesis,
        artifact_merge_planner=stack.artifact_merge_planner,
        artifact_export_intelligence=stack.artifact_export_intelligence,
        creative_critic=stack.creative_critic,
        generated_response=generated_response,
    )


def _planner(stack: object, self_evaluation: object):
    runtime_base = stack.runtime_base
    return derive_creative_improvement_planner_profile(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        creative_critic=stack.creative_critic,
        self_evaluation=self_evaluation,
    )


def _director(stack: object, self_evaluation: object, planner: object):
    runtime_base = stack.runtime_base
    return derive_creative_assistant_director_brief(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        creative_translation=runtime_base.prompt_input.creative_translation,
        creative_intent=runtime_base.intent,
        creative_hierarchy=runtime_base.hierarchy,
        creative_strategy=runtime_base.strategy,
        creative_techniques=runtime_base.techniques,
        creative_plan=runtime_base.plan,
        creative_constraints=runtime_base.constraints,
        creative_constraint_priorities=runtime_base.prioritization,
        runtime_capabilities=runtime_base.runtime_capabilities,
        creative_tradeoffs=runtime_base.tradeoffs,
        creative_quality_prediction=getattr(runtime_base, "quality_prediction", None),
        artifact_plan=runtime_base.artifact_plan,
        artifact_dependency_graph=runtime_base.artifact_dependency_graph,
        runtime_compatibility=runtime_base.runtime_compatibility,
        artifact_capability_matrix=stack.artifact_capability_matrix,
        multi_artifact_strategy=stack.multi_artifact_strategy,
        artifact_critic=stack.artifact_critic,
        artifact_refiner=stack.artifact_refiner,
        artifact_intelligence_synthesis=stack.artifact_intelligence_synthesis,
        artifact_merge_planner=stack.artifact_merge_planner,
        artifact_export_intelligence=stack.artifact_export_intelligence,
        creative_critic=stack.creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=planner,
    )


def _reasoning(
    stack: object,
    self_evaluation: object,
    planner: object,
    director: object,
):
    runtime_base = stack.runtime_base
    return derive_creative_reasoning_result(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        creative_translation=runtime_base.prompt_input.creative_translation,
        creative_intent=runtime_base.intent,
        creative_hierarchy=runtime_base.hierarchy,
        creative_plan=runtime_base.plan,
        creative_director=director,
        creative_constraints=runtime_base.constraints,
        creative_constraint_priorities=runtime_base.prioritization,
        creative_strategy=runtime_base.strategy,
        creative_techniques=runtime_base.techniques,
        runtime_capabilities=runtime_base.runtime_capabilities,
        creative_tradeoffs=runtime_base.tradeoffs,
        creative_quality_prediction=getattr(runtime_base, "quality_prediction", None),
        artifact_plan=runtime_base.artifact_plan,
        artifact_dependency_graph=runtime_base.artifact_dependency_graph,
        runtime_compatibility=runtime_base.runtime_compatibility,
        artifact_capability_matrix=stack.artifact_capability_matrix,
        multi_artifact_strategy=stack.multi_artifact_strategy,
        artifact_critic=stack.artifact_critic,
        artifact_refiner=stack.artifact_refiner,
        artifact_intelligence_synthesis=stack.artifact_intelligence_synthesis,
        artifact_merge_planner=stack.artifact_merge_planner,
        artifact_export_intelligence=stack.artifact_export_intelligence,
        creative_critic=stack.creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=planner,
    )


if __name__ == "__main__":
    unittest.main()
