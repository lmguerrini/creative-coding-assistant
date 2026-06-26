import unittest

from creative_coding_assistant.contracts import StreamEventType
from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    build_assistant_workflow_graph,
    build_rendered_prompt_request,
    creative_score_prompt_lines,
    derive_creative_assistant_director_brief,
    derive_creative_reasoning_result,
    derive_creative_score_profile,
    stream_assistant_workflow_events,
)
from test_creative_confidence_engine import _confidence, _planner, _reflection
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


class CreativeScoreEngineTests(unittest.TestCase):
    def test_derives_high_score_from_aligned_metadata(self) -> None:
        stack = _stack("Generate a clear p5.js mandala with score metadata.")
        critic = _high_quality_critic(stack)
        evaluation = _high_quality_evaluation(stack, critic=critic)
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)
        confidence = _confidence(stack, critic, evaluation, planner, reflection)
        score = _score(stack, critic, evaluation, planner, reflection, confidence)

        self.assertEqual(score.role, "creative_score_engine")
        self.assertEqual(score.serialization_version, "v1")
        self.assertGreaterEqual(score.overall_creative_score, 75)
        self.assertIn(score.score_band, {"strong", "excellent"})
        self.assertEqual(score.hitl_recommendation, "not_needed")
        self.assertEqual(len(score.score_breakdown), 6)
        self.assertIn("does not modify outputs", score.authority_boundary)
        self.assertTrue(creative_score_prompt_lines(score))

    def test_derives_low_score_from_high_risk_metadata(self) -> None:
        stack = _stack("Generate a fragile artifact with unsupported claims.")
        critic = stack.creative_critic.model_copy(
            update={
                "risk_assessment": "high",
                "concept_quality": 0.36,
                "execution_quality": 0.3,
                "artifact_quality": 0.34,
                "coherence_quality": 0.32,
                "runtime_fit_quality": 0.28,
                "originality_quality": 0.38,
                "clarity_quality": 0.31,
                "feasibility_quality": 0.27,
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
                "request_alignment": 0.34,
                "intent_alignment": 0.35,
                "constraint_alignment": 0.28,
                "artifact_alignment": 0.32,
                "runtime_alignment": 0.25,
                "creative_coherence": 0.3,
                "technical_coherence": 0.27,
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
        score = _score(stack, critic, evaluation, planner, reflection, confidence)

        self.assertLess(score.overall_creative_score, 55)
        self.assertIn(score.score_band, {"weak", "critical"})
        self.assertIn(score.hitl_recommendation, {"recommended", "required"})
        self.assertGreater(score.risk_penalty, 8)
        self.assertTrue(score.weaknesses)

    def test_conflicting_evaluation_signals_lower_score_and_surface_weaknesses(
        self,
    ) -> None:
        stack = _stack("Generate a p5.js sketch with mixed scoring signals.")
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
                "quality_gaps": ("Self evaluation conflicts with critic strength.",),
            }
        )
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)
        confidence = _confidence(stack, critic, evaluation, planner, reflection)
        score = _score(stack, critic, evaluation, planner, reflection, confidence)

        spread = max(item.score for item in score.score_breakdown) - min(
            item.score for item in score.score_breakdown
        )
        self.assertGreater(spread, 10)
        self.assertLess(score.overall_creative_score, 75)
        self.assertTrue(score.weaknesses)

    def test_high_uncertainty_adds_penalty_and_human_review_guidance(self) -> None:
        stack = _stack("Generate a clean sketch with unresolved score criteria.")
        critic = _high_quality_critic(stack)
        evaluation = _high_quality_evaluation(stack, critic=critic).model_copy(
            update={
                "missing_information": (
                    "Success criteria are not fully specified.",
                    "Runtime expectations need confirmation.",
                    "Final artifact evaluation target is ambiguous.",
                ),
                "hitl_questions": (
                    "Should score criteria be confirmed by a human?",
                ),
                "ambiguity_assessment": "high",
            }
        )
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)
        confidence = _confidence(stack, critic, evaluation, planner, reflection)
        score = _score(stack, critic, evaluation, planner, reflection, confidence)

        self.assertGreaterEqual(score.uncertainty_penalty, 8)
        self.assertIn(
            score.hitl_recommendation,
            {"optional", "recommended", "required"},
        )
        self.assertTrue(
            any("human review" in item.lower() for item in score.prompt_guidance),
            score.prompt_guidance,
        )

    def test_high_confidence_increases_confidence_weight(self) -> None:
        stack = _stack("Generate a clear p5.js score-weighted mandala.")
        critic = _high_quality_critic(stack)
        evaluation = _high_quality_evaluation(stack, critic=critic)
        planner = _planner(stack, critic, evaluation)
        reflection = _reflection(stack, critic, evaluation, planner)
        confidence = _confidence(stack, critic, evaluation, planner, reflection)
        score = _score(stack, critic, evaluation, planner, reflection, confidence)

        self.assertGreaterEqual(confidence.confidence_score, 0.8)
        self.assertGreaterEqual(score.confidence_weight, 0.96)
        self.assertGreaterEqual(score.overall_creative_score, 75)

    def test_integrates_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack("Generate a symbolic p5.js phoenix mandala with score metadata.")
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
        director = _director(stack, evaluation, planner, reflection, confidence, score)
        reasoning = _reasoning(
            stack,
            evaluation,
            planner,
            reflection,
            confidence,
            score,
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

        self.assertIn("Creative Score Engine:", system)
        self.assertIn("Overall creative score:", system)
        self.assertTrue(
            any("Creative score:" in item for item in director.planning_focus),
            director.planning_focus,
        )
        self.assertIn("creative_score", {item.source for item in reasoning.evidence_chain})
        self.assertEqual(
            score.model_dump(mode="json")["role"],
            "creative_score_engine",
        )
        self.assertNotIn("automatic refinement enabled", system.lower())
        self.assertNotIn("runtime selection enabled", system.lower())

    def test_workflow_and_final_payload_serialize_creative_score(self) -> None:
        graph = build_assistant_workflow_graph()
        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(
                    query="Generate a p5.js sketch with creative score metadata.",
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
        planning_score = planning_event.payload["creative_score"]
        final_score = final_event.payload["creative_score"]

        self.assertEqual(planning_score["role"], "creative_score_engine")
        self.assertTrue(planning_event.payload["workflow"]["creative_score_available"])
        self.assertEqual(
            planning_event.payload["workflow"]["creative_score"],
            planning_score,
        )
        self.assertEqual(final_score["role"], "creative_score_engine")
        self.assertEqual(final_event.payload["workflow"]["creative_score"], final_score)


def _score(
    stack: object,
    critic: object,
    evaluation: object,
    planner: object,
    reflection: object,
    confidence: object,
):
    runtime_base = stack.runtime_base
    return derive_creative_score_profile(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        creative_critic=critic,
        self_evaluation=evaluation,
        creative_improvement_planner=planner,
        reflection_loop=reflection,
        creative_confidence=confidence,
    )


def _director(
    stack: object,
    evaluation: object,
    planner: object,
    reflection: object,
    confidence: object,
    score: object,
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
    )


def _reasoning(
    stack: object,
    evaluation: object,
    planner: object,
    reflection: object,
    confidence: object,
    score: object,
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
    )


if __name__ == "__main__":
    unittest.main()
