import unittest

from creative_coding_assistant.contracts import StreamEventType
from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    build_assistant_workflow_graph,
    build_rendered_prompt_request,
    creative_confidence_prompt_lines,
    derive_creative_assistant_director_brief,
    derive_creative_confidence_profile,
    derive_creative_improvement_planner_profile,
    derive_creative_reasoning_result,
    derive_reflection_loop_profile,
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
from test_reflection_loop_engine import (
    _high_quality_critic,
    _high_quality_evaluation,
)


class CreativeConfidenceEngineTests(unittest.TestCase):
    def test_derives_high_confidence_from_aligned_metadata(self) -> None:
        stack = _stack("Generate a clear p5.js mandala with confidence metadata.")
        critic = _high_quality_critic(stack)
        evaluation = _high_quality_evaluation(stack, critic=critic)
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)
        confidence = _confidence(stack, critic, evaluation, planner, reflection)

        self.assertEqual(confidence.role, "creative_confidence_engine")
        self.assertEqual(confidence.serialization_version, "v1")
        self.assertGreaterEqual(confidence.confidence_score, 0.8)
        self.assertIn(confidence.confidence_level, {"high", "very_high"})
        self.assertEqual(confidence.expected_human_review_need, "not_needed")
        self.assertEqual(confidence.escalation_recommendation, "none")
        self.assertIn("does not change outputs", confidence.authority_boundary)
        self.assertTrue(creative_confidence_prompt_lines(confidence))

    def test_derives_low_confidence_from_high_risk_metadata(self) -> None:
        stack = _stack("Generate a fragile artifact with unsupported claims.")
        critic = stack.creative_critic.model_copy(
            update={
                "risk_assessment": "high",
                "concept_quality": 0.4,
                "execution_quality": 0.34,
                "artifact_quality": 0.35,
                "coherence_quality": 0.38,
                "runtime_fit_quality": 0.32,
                "clarity_quality": 0.36,
                "feasibility_quality": 0.3,
                "creative_weaknesses": (
                    "Unsupported runtime claims dominate the plan.",
                ),
                "unsupported_assumptions": (
                    "Unsupported runtime behavior is treated as available.",
                ),
            }
        )
        evaluation = _self_evaluation(stack, generated_response="Incomplete.").model_copy(
            update={
                "request_alignment": 0.38,
                "intent_alignment": 0.4,
                "constraint_alignment": 0.32,
                "artifact_alignment": 0.35,
                "runtime_alignment": 0.28,
                "creative_coherence": 0.36,
                "technical_coherence": 0.3,
                "completeness_assessment": "partial",
                "ambiguity_assessment": "high",
                "hallucination_risk": "high",
                "underdelivery_risk": "high",
                "quality_gaps": ("Runtime claims need explicit caveats.",),
                "hitl_questions": (
                    "Should unsupported runtime claims be resolved first?",
                ),
            }
        )
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)
        confidence = _confidence(stack, critic, evaluation, planner, reflection)

        self.assertLess(confidence.confidence_score, 0.55)
        self.assertIn(confidence.confidence_level, {"critical", "low"})
        self.assertEqual(confidence.hitl_recommendation, "required")
        self.assertIn(
            confidence.escalation_recommendation,
            {"hitl_review", "future_escalation"},
        )
        self.assertTrue(confidence.confidence_weaknesses)

    def test_detects_conflicting_evidence(self) -> None:
        stack = _stack("Generate a p5.js sketch with mixed confidence signals.")
        critic = _high_quality_critic(stack)
        evaluation = _self_evaluation(stack, generated_response="Incomplete.").model_copy(
            update={
                "request_alignment": 0.36,
                "intent_alignment": 0.35,
                "constraint_alignment": 0.34,
                "artifact_alignment": 0.32,
                "runtime_alignment": 0.3,
                "creative_coherence": 0.35,
                "technical_coherence": 0.31,
                "completeness_assessment": "partial",
                "ambiguity_assessment": "high",
                "hallucination_risk": "medium",
                "underdelivery_risk": "high",
            }
        )
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)
        confidence = _confidence(stack, critic, evaluation, planner, reflection)

        self.assertEqual(confidence.confidence_trend, "conflicting")
        self.assertTrue(
            any(item.source == "creative_critic" for item in confidence.confidence_components)
        )
        self.assertTrue(
            any(item.source == "self_evaluation" for item in confidence.confidence_components)
        )

    def test_uncertain_evaluation_recommends_hitl_without_execution_changes(self) -> None:
        stack = _stack("Generate a clean sketch with unresolved success criteria.")
        critic = _high_quality_critic(stack)
        evaluation = _high_quality_evaluation(stack, critic=critic).model_copy(
            update={
                "missing_information": (
                    "Success criteria are not fully specified.",
                    "Final runtime expectations need confirmation.",
                ),
                "hitl_questions": (
                    "Should runtime caveats be reviewed by a human?",
                ),
                "ambiguity_assessment": "medium",
            }
        )
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)
        confidence = _confidence(stack, critic, evaluation, planner, reflection)

        self.assertTrue(confidence.confidence_uncertainties)
        self.assertIn(
            confidence.hitl_recommendation,
            {"optional", "recommended", "required"},
        )
        self.assertTrue(
            any("Do not change outputs" in item for item in confidence.prompt_guidance),
            confidence.prompt_guidance,
        )

    def test_integrates_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack(
            "Generate a symbolic p5.js phoenix mandala with confidence metadata."
        )
        evaluation = _self_evaluation(
            stack,
            generated_response="Here is a symbolic p5.js phoenix mandala with caveats.",
        )
        planner = _planner(stack, stack.creative_critic, evaluation)
        reflection = _reflection(stack, stack.creative_critic, evaluation, planner)
        confidence = _confidence(
            stack,
            stack.creative_critic,
            evaluation,
            planner,
            reflection,
        )
        director = _director(stack, evaluation, planner, reflection, confidence)
        reasoning = _reasoning(
            stack,
            evaluation,
            planner,
            reflection,
            confidence,
            director,
        )
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
                "creative_confidence": confidence,
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

        self.assertIn("Creative Confidence Engine:", system)
        self.assertIn("Confidence score:", system)
        self.assertTrue(
            any("Creative confidence:" in item for item in director.planning_focus),
            director.planning_focus,
        )
        self.assertIn(
            "creative_confidence",
            {item.source for item in reasoning.evidence_chain},
        )
        self.assertEqual(
            confidence.model_dump(mode="json")["role"],
            "creative_confidence_engine",
        )
        self.assertNotIn("automatic refinement enabled", system.lower())
        self.assertNotIn("provider routing enabled", system.lower())

    def test_workflow_and_final_payload_serialize_creative_confidence(self) -> None:
        graph = build_assistant_workflow_graph()
        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(
                    query="Generate a p5.js sketch with creative confidence metadata.",
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
        planning_confidence = planning_event.payload["creative_confidence"]
        final_confidence = final_event.payload["creative_confidence"]

        self.assertEqual(planning_confidence["role"], "creative_confidence_engine")
        self.assertTrue(
            planning_event.payload["workflow"]["creative_confidence_available"]
        )
        self.assertEqual(
            planning_event.payload["workflow"]["creative_confidence"],
            planning_confidence,
        )
        self.assertEqual(final_confidence["role"], "creative_confidence_engine")
        self.assertEqual(
            final_event.payload["workflow"]["creative_confidence"],
            final_confidence,
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


def _confidence(
    stack: object,
    critic: object,
    evaluation: object,
    planner: object,
    reflection: object,
):
    runtime_base = stack.runtime_base
    return derive_creative_confidence_profile(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        creative_critic=critic,
        self_evaluation=evaluation,
        creative_improvement_planner=planner,
        reflection_loop=reflection,
    )


def _director(
    stack: object,
    evaluation: object,
    planner: object,
    reflection: object,
    confidence: object,
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
        creative_confidence=confidence,
    )


def _reasoning(
    stack: object,
    evaluation: object,
    planner: object,
    reflection: object,
    confidence: object,
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
        creative_confidence=confidence,
    )


if __name__ == "__main__":
    unittest.main()
