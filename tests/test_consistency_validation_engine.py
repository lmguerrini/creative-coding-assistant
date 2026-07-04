import unittest

from creative_coding_assistant.contracts import StreamEventType
from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    build_assistant_workflow_graph,
    build_rendered_prompt_request,
    consistency_validation_prompt_lines,
    derive_consistency_validation_profile,
    derive_creative_assistant_director_brief,
    derive_creative_reasoning_result,
    stream_assistant_workflow_events,
)
from test_creative_confidence_engine import _confidence, _planner, _reflection
from test_creative_critic_engine import _stack
from test_creative_improvement_planner import _self_evaluation
from test_creative_score_engine import _score
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


class ConsistencyValidationEngineTests(unittest.TestCase):
    def test_derives_consistent_evaluation_integrity(self) -> None:
        stack = _stack("Generate a clear p5.js mandala with consistency metadata.")
        critic = _high_quality_critic(stack)
        evaluation = _high_quality_evaluation(stack, critic=critic)
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)
        confidence = _confidence(stack, critic, evaluation, planner, reflection)
        score = _score(stack, critic, evaluation, planner, reflection, confidence)
        consistency = _consistency(
            stack,
            critic,
            evaluation,
            planner,
            reflection,
            confidence,
            score,
        )

        self.assertEqual(consistency.role, "consistency_validation_engine")
        self.assertEqual(consistency.serialization_version, "v1")
        self.assertEqual(consistency.contradiction_level, "none")
        self.assertEqual(consistency.detected_conflicts, ())
        self.assertIn(consistency.evaluation_integrity, {"strong", "adequate"})
        self.assertIn(consistency.consistency_status, {"consistent", "needs_attention"})
        self.assertEqual(consistency.score_consistency.status, "aligned")
        self.assertIn("does not modify outputs", consistency.authority_boundary)
        self.assertTrue(consistency_validation_prompt_lines(consistency))

    def test_detects_conflicting_score_and_confidence(self) -> None:
        stack = _stack("Generate a high score with low confidence contradiction.")
        critic = _high_quality_critic(stack)
        evaluation = _high_quality_evaluation(stack, critic=critic)
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)
        confidence = _confidence(stack, critic, evaluation, planner, reflection)
        score = _score(stack, critic, evaluation, planner, reflection, confidence)
        low_confidence = confidence.model_copy(
            update={
                "confidence_score": 0.42,
                "confidence_level": "low",
                "confidence_trend": "conflicting",
                "expected_output_reliability": "low",
                "expected_execution_readiness": "needs_hitl",
                "expected_human_review_need": "required",
                "hitl_recommendation": "required",
            }
        )
        consistency = _consistency(
            stack,
            critic,
            evaluation,
            planner,
            reflection,
            low_confidence,
            score,
        )

        self.assertEqual(consistency.score_consistency.status, "conflict")
        self.assertEqual(consistency.confidence_consistency.status, "conflict")
        self.assertIn(consistency.contradiction_level, {"medium", "high"})
        self.assertTrue(
            any(
                "confidence" in item.lower() for item in consistency.detected_conflicts
            ),
            consistency.detected_conflicts,
        )

    def test_detects_reflection_disagreement(self) -> None:
        stack = _stack("Generate a strong artifact with excessive reflection pressure.")
        critic = _high_quality_critic(stack)
        evaluation = _high_quality_evaluation(stack, critic=critic)
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)
        high_pressure_reflection = reflection.model_copy(
            update={
                "reflection_required": True,
                "reflection_priority": "critical",
                "reflection_depth": "deep",
                "hitl_recommendation": "required",
            }
        )
        confidence = _confidence(stack, critic, evaluation, planner, reflection)
        score = _score(stack, critic, evaluation, planner, reflection, confidence)
        consistency = _consistency(
            stack,
            critic,
            evaluation,
            planner,
            high_pressure_reflection,
            confidence,
            score,
        )

        self.assertEqual(consistency.reflection_consistency.status, "conflict")
        self.assertTrue(
            any(
                "reflection" in item.lower() for item in consistency.detected_conflicts
            ),
            consistency.detected_conflicts,
        )

    def test_detects_critic_disagreement(self) -> None:
        stack = _stack("Generate a critic disagreement metadata check.")
        critic = _high_quality_critic(stack).model_copy(
            update={"risk_assessment": "high"}
        )
        evaluation = _high_quality_evaluation(stack, critic=critic)
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)
        confidence = _confidence(stack, critic, evaluation, planner, reflection)
        score = _score(stack, critic, evaluation, planner, reflection, confidence)
        consistency = _consistency(
            stack,
            critic,
            evaluation,
            planner,
            reflection,
            confidence,
            score,
        )

        self.assertEqual(consistency.critic_consistency.status, "conflict")
        self.assertTrue(
            any("critic" in item.lower() for item in consistency.detected_conflicts),
            consistency.detected_conflicts,
        )

    def test_detects_ambiguity_and_unsupported_conclusions(self) -> None:
        stack = _stack("Generate a sketch with ambiguous validation criteria.")
        critic = _high_quality_critic(stack).model_copy(
            update={
                "missing_information": (
                    "Success criteria are implicit.",
                    "Runtime support expectations need confirmation.",
                ),
                "unsupported_assumptions": (
                    "Unsupported runtime behavior is treated as available.",
                ),
                "hitl_questions": ("Should assumptions be reviewed?",),
            }
        )
        evaluation = _high_quality_evaluation(stack, critic=critic).model_copy(
            update={
                "ambiguity_assessment": "high",
                "missing_information": (
                    "Final evaluation criteria are unresolved.",
                    "Visual success threshold is ambiguous.",
                ),
                "unsupported_assumptions": (
                    "The answer assumes a live preview is available.",
                ),
                "hitl_questions": ("Should ambiguity be resolved first?",),
            }
        )
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)
        confidence = _confidence(stack, critic, evaluation, planner, reflection)
        score = _score(stack, critic, evaluation, planner, reflection, confidence)
        consistency = _consistency(
            stack,
            critic,
            evaluation,
            planner,
            reflection,
            confidence,
            score,
        )

        self.assertEqual(consistency.ambiguity_level, "high")
        self.assertTrue(consistency.unsupported_conclusions)
        self.assertIn(
            consistency.hitl_recommendation,
            {"optional", "recommended", "required"},
        )

    def test_integrates_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack(
            "Generate a symbolic p5.js phoenix mandala with consistency metadata."
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
        score = _score(
            stack,
            stack.creative_critic,
            evaluation,
            planner,
            reflection,
            confidence,
        )
        consistency = _consistency(
            stack,
            stack.creative_critic,
            evaluation,
            planner,
            reflection,
            confidence,
            score,
        )
        director = _director(
            stack,
            evaluation,
            planner,
            reflection,
            confidence,
            score,
            consistency,
        )
        reasoning = _reasoning(
            stack,
            evaluation,
            planner,
            reflection,
            confidence,
            score,
            consistency,
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
                "creative_score": score,
                "consistency_validation": consistency,
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

        self.assertIn("Consistency Validation Engine:", system)
        self.assertIn("Consistency status:", system)
        self.assertTrue(
            any("Consistency validation:" in item for item in director.planning_focus),
            director.planning_focus,
        )
        self.assertIn(
            "consistency_validation",
            {item.source for item in reasoning.evidence_chain},
        )
        self.assertEqual(
            consistency.model_dump(mode="json")["role"],
            "consistency_validation_engine",
        )
        self.assertNotIn("automatic refinement enabled", system.lower())
        self.assertNotIn("runtime selection enabled", system.lower())

    def test_workflow_and_final_payload_serialize_consistency_validation(self) -> None:
        graph = build_assistant_workflow_graph()
        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(
                    query="Generate a p5.js sketch with consistency metadata.",
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
        planning_consistency = planning_event.payload["consistency_validation"]
        final_consistency = final_event.payload["consistency_validation"]

        self.assertEqual(
            planning_consistency["role"],
            "consistency_validation_engine",
        )
        self.assertIn("score_consistency", planning_consistency)
        self.assertIn("confidence_consistency", planning_consistency)
        self.assertIn("reasoning_consistency", planning_consistency)
        self.assertTrue(
            planning_event.payload["workflow"]["consistency_validation_available"]
        )
        self.assertEqual(
            planning_event.payload["workflow"]["consistency_validation"],
            planning_consistency,
        )
        self.assertEqual(final_consistency["role"], "consistency_validation_engine")
        self.assertEqual(
            final_event.payload["workflow"]["consistency_validation"],
            final_consistency,
        )


def _consistency(
    stack: object,
    critic: object,
    evaluation: object,
    planner: object,
    reflection: object,
    confidence: object,
    score: object,
):
    runtime_base = stack.runtime_base
    return derive_consistency_validation_profile(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        creative_critic=critic,
        self_evaluation=evaluation,
        creative_improvement_planner=planner,
        reflection_loop=reflection,
        creative_confidence=confidence,
        creative_score=score,
    )


def _director(
    stack: object,
    evaluation: object,
    planner: object,
    reflection: object,
    confidence: object,
    score: object,
    consistency: object,
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
        creative_score=score,
        consistency_validation=consistency,
    )


def _reasoning(
    stack: object,
    evaluation: object,
    planner: object,
    reflection: object,
    confidence: object,
    score: object,
    consistency: object,
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
        creative_score=score,
        consistency_validation=consistency,
    )


if __name__ == "__main__":
    unittest.main()
