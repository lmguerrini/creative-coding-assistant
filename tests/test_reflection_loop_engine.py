import unittest

from creative_coding_assistant.contracts import StreamEventType
from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    build_assistant_workflow_graph,
    build_rendered_prompt_request,
    derive_creative_assistant_director_brief,
    derive_creative_improvement_planner_profile,
    derive_creative_reasoning_result,
    derive_reflection_loop_profile,
    reflection_loop_prompt_lines,
    stream_assistant_workflow_events,
)
from test_creative_critic_engine import _stack
from test_creative_improvement_planner import _self_evaluation
from test_langgraph_workflow_integration import (
    _first_event,
    _request,
    _runtime,
    _stream_completed_generation,
    _stream_prompt_inputs_with_builder,
)


class ReflectionLoopEngineTests(unittest.TestCase):
    def test_high_quality_plan_does_not_require_reflection(self) -> None:
        stack = _stack("Generate a clear p5.js mandala with bounded metadata.")
        critic = _high_quality_critic(stack)
        evaluation = _high_quality_evaluation(stack, critic=critic)
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)

        self.assertEqual(reflection.role, "reflection_loop_engine")
        self.assertEqual(reflection.serialization_version, "v1")
        self.assertFalse(reflection.reflection_required)
        self.assertEqual(reflection.reflection_priority, "none")
        self.assertEqual(reflection.reflection_depth, "none")
        self.assertEqual(reflection.expected_quality_gain, "none")
        self.assertEqual(reflection.expected_risk_reduction, "none")
        self.assertIn("does not regenerate responses", reflection.authority_boundary)
        self.assertTrue(reflection_loop_prompt_lines(reflection))

    def test_low_quality_plan_recommends_high_gain_reflection(self) -> None:
        stack = _stack("Generate a fragile multi-runtime artifact with gaps.")
        critic = stack.creative_critic.model_copy(
            update={
                "risk_assessment": "high",
                "execution_quality": 0.42,
                "artifact_quality": 0.44,
                "runtime_fit_quality": 0.38,
                "feasibility_quality": 0.4,
                "creative_weaknesses": (
                    "Runtime assumptions are unsupported.",
                    "The requested artifact scope is underdelivered.",
                ),
                "unsupported_assumptions": (
                    "Unsupported runtime behavior is treated as available.",
                ),
                "improvement_opportunities": (
                    "Caveat unsupported runtime behavior before expanding scope.",
                ),
            }
        )
        evaluation = _self_evaluation(
            stack, generated_response="Incomplete."
        ).model_copy(
            update={
                "request_alignment": 0.45,
                "intent_alignment": 0.48,
                "constraint_alignment": 0.42,
                "runtime_alignment": 0.36,
                "creative_coherence": 0.46,
                "technical_coherence": 0.4,
                "completeness_assessment": "partial",
                "ambiguity_assessment": "high",
                "hallucination_risk": "high",
                "underdelivery_risk": "high",
                "quality_gaps": ("Unsupported runtime claims need explicit caveats.",),
                "improvement_opportunities": (
                    "Resolve runtime caveats before adding visual complexity.",
                ),
            }
        )
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)

        self.assertTrue(reflection.reflection_required)
        self.assertIn(reflection.reflection_priority, {"critical", "high"})
        self.assertIn(reflection.reflection_depth, {"deep", "moderate"})
        self.assertEqual(reflection.expected_quality_gain, "high")
        self.assertEqual(reflection.expected_risk_reduction, "high")
        self.assertGreater(reflection.confidence_after_reflection, 0.5)

    def test_low_gain_reflection_remains_advisory(self) -> None:
        stack = _stack("Generate a clean p5.js sketch with one small ambiguity.")
        critic = _high_quality_critic(stack)
        evaluation = _high_quality_evaluation(stack, critic=critic).model_copy(
            update={
                "ambiguity_assessment": "medium",
                "missing_information": ("Final color preference is unspecified.",),
            }
        )
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)

        self.assertFalse(reflection.reflection_required)
        self.assertEqual(reflection.reflection_priority, "low")
        self.assertEqual(reflection.expected_quality_gain, "low")
        self.assertIn("Do not trigger", " ".join(reflection.stop_conditions))

    def test_integrates_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack(
            "Generate a symbolic p5.js phoenix mandala with Reflection Loop metadata."
        )
        evaluation = _self_evaluation(
            stack,
            generated_response="Here is a symbolic p5.js phoenix mandala with caveats.",
        )
        planner = _planner(stack, stack.creative_critic, evaluation)
        reflection = _reflection(stack, stack.creative_critic, evaluation, planner)
        director = _director(stack, evaluation, planner, reflection)
        reasoning = _reasoning(stack, evaluation, planner, reflection, director)
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
                "self_evaluation": evaluation,
                "creative_improvement_planner": planner,
                "reflection_loop": reflection,
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

        self.assertIn("Reflection Loop Engine:", system)
        self.assertIn("Reflection priority:", system)
        self.assertTrue(
            any("Reflection loop:" in item for item in director.planning_focus),
            director.planning_focus,
        )
        self.assertIn(
            "reflection_loop",
            {item.source for item in reasoning.evidence_chain},
        )
        self.assertEqual(
            reflection.model_dump(mode="json")["role"],
            "reflection_loop_engine",
        )
        self.assertNotIn("workflow loop enabled", system.lower())
        self.assertNotIn("runtime auto-selection enabled", system.lower())

    def test_workflow_and_final_payload_serialize_reflection_loop(self) -> None:
        graph = build_assistant_workflow_graph()
        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(
                    query="Generate a p5.js sketch with reflection loop metadata.",
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
        planning_reflection = planning_event.payload["reflection_loop"]
        final_reflection = final_event.payload["reflection_loop"]

        self.assertEqual(planning_reflection["role"], "reflection_loop_engine")
        self.assertTrue(planning_event.payload["workflow"]["reflection_loop_available"])
        self.assertEqual(
            planning_event.payload["workflow"]["reflection_loop"],
            planning_reflection,
        )
        self.assertEqual(final_reflection["role"], "reflection_loop_engine")
        self.assertEqual(
            final_event.payload["workflow"]["reflection_loop"],
            final_reflection,
        )


def _high_quality_critic(stack: object):
    return stack.creative_critic.model_copy(
        update={
            "critic_confidence": 0.9,
            "concept_quality": 0.92,
            "execution_quality": 0.9,
            "artifact_quality": 0.9,
            "coherence_quality": 0.91,
            "runtime_fit_quality": 0.9,
            "originality_quality": 0.88,
            "clarity_quality": 0.92,
            "feasibility_quality": 0.9,
            "risk_assessment": "low",
            "creative_weaknesses": (),
            "missing_information": (),
            "unsupported_assumptions": (),
            "improvement_opportunities": (),
            "hitl_questions": (),
        }
    )


def _high_quality_evaluation(stack: object, *, critic: object):
    return _self_evaluation(
        stack,
        generated_response="Complete bounded p5.js sketch with clear caveats.",
    ).model_copy(
        update={
            "self_evaluation_confidence": 0.9,
            "request_alignment": 0.92,
            "intent_alignment": 0.9,
            "constraint_alignment": 0.9,
            "artifact_alignment": 0.9,
            "runtime_alignment": 0.9,
            "creative_coherence": 0.91,
            "technical_coherence": 0.9,
            "completeness_assessment": "complete",
            "ambiguity_assessment": "low",
            "hallucination_risk": "low",
            "overreach_risk": "low",
            "underdelivery_risk": "low",
            "missing_information": (),
            "unsupported_assumptions": (),
            "quality_gaps": (),
            "improvement_opportunities": (),
            "hitl_questions": (),
        }
    )


def _planner(stack: object, critic: object, evaluation: object):
    runtime_base = stack.runtime_base
    return derive_creative_improvement_planner_profile(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        creative_critic=critic,
        self_evaluation=evaluation,
    )


def _reflection(
    stack: object,
    critic: object,
    evaluation: object,
    planner: object,
):
    runtime_base = stack.runtime_base
    return derive_reflection_loop_profile(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        creative_critic=critic,
        self_evaluation=evaluation,
        creative_improvement_planner=planner,
    )


def _director(
    stack: object,
    evaluation: object,
    planner: object,
    reflection: object,
):
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
        self_evaluation=evaluation,
        creative_improvement_planner=planner,
        reflection_loop=reflection,
    )


def _reasoning(
    stack: object,
    evaluation: object,
    planner: object,
    reflection: object,
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
        self_evaluation=evaluation,
        creative_improvement_planner=planner,
        reflection_loop=reflection,
    )


if __name__ == "__main__":
    unittest.main()
