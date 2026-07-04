import unittest

from creative_coding_assistant.contracts import StreamEventType
from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    build_assistant_workflow_graph,
    build_rendered_prompt_request,
    derive_creative_assistant_director_brief,
    derive_creative_reasoning_result,
    derive_evaluation_report_profile,
    evaluation_report_prompt_lines,
    stream_assistant_workflow_events,
)
from test_consistency_validation_engine import _consistency
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


class EvaluationReportsTests(unittest.TestCase):
    def test_derives_report_sections_from_evaluation_stack(self) -> None:
        stack, evaluation, planner, reflection, confidence, score, consistency = (
            _evaluation_stack("Generate a p5.js mandala with evaluation reports.")
        )
        report = _report(
            stack,
            evaluation,
            planner,
            reflection,
            confidence,
            score,
            consistency,
        )

        self.assertEqual(report.role, "evaluation_reports")
        self.assertEqual(report.serialization_version, "v1")
        self.assertTrue(report.executive_summary)
        self.assertTrue(report.quality_summary)
        self.assertTrue(report.confidence_summary)
        self.assertTrue(report.consistency_summary)
        self.assertTrue(report.improvement_summary)
        self.assertTrue(report.score_summary)
        self.assertTrue(report.recommendations)
        self.assertIn(
            report.hitl_recommendation,
            {"not_needed", "optional", "recommended", "required"},
        )
        self.assertIn("metadata only", report.authority_boundary)

    def test_trace_provenance_dependencies_and_evidence_are_ordered(self) -> None:
        stack, evaluation, planner, reflection, confidence, score, consistency = (
            _evaluation_stack("Generate a report with traceable provenance.")
        )
        report = _report(
            stack,
            evaluation,
            planner,
            reflection,
            confidence,
            score,
            consistency,
        )
        expected_sources = (
            "creative_critic",
            "self_evaluation",
            "creative_improvement_planner",
            "reflection_loop",
            "creative_confidence",
            "creative_score",
            "consistency_validation",
        )

        self.assertEqual(
            tuple(item.source for item in report.evaluation_trace),
            expected_sources,
        )
        self.assertEqual(
            tuple(item.source for item in report.evaluation_dependencies),
            expected_sources,
        )
        self.assertTrue(
            all(item.status == "available" for item in report.evaluation_dependencies)
        )
        self.assertEqual(
            {item.source for item in report.evaluation_provenance},
            set(expected_sources),
        )
        self.assertIn(
            "consistency_validation",
            {item.source for item in report.evidence_chain},
        )

    def test_prompt_lines_and_serialization_are_inspectable(self) -> None:
        stack, evaluation, planner, reflection, confidence, score, consistency = (
            _evaluation_stack("Generate a report with prompt guidance.")
        )
        report = _report(
            stack,
            evaluation,
            planner,
            reflection,
            confidence,
            score,
            consistency,
        )
        lines = evaluation_report_prompt_lines(report)
        payload = report.model_dump(mode="json")

        self.assertIn("evaluation_trace", payload)
        self.assertIn("evaluation_provenance", payload)
        self.assertIn("evaluation_dependencies", payload)
        self.assertTrue(any("Evaluation trace:" in item for item in lines))
        self.assertTrue(any("Evaluation provenance:" in item for item in lines))
        self.assertTrue(any("Evaluation dependency:" in item for item in lines))
        self.assertTrue(
            any(
                "evaluation pipeline" in item.lower()
                for item in report.evaluation_explainability
            ),
            report.evaluation_explainability,
        )

    def test_integrates_prompt_director_reasoning_and_serialization(self) -> None:
        stack, evaluation, planner, reflection, confidence, score, consistency = (
            _evaluation_stack("Generate a symbolic phoenix with evaluation report.")
        )
        report = _report(
            stack,
            evaluation,
            planner,
            reflection,
            confidence,
            score,
            consistency,
        )
        director = _director(
            stack,
            evaluation,
            planner,
            reflection,
            confidence,
            score,
            consistency,
            report,
        )
        reasoning = _reasoning(
            stack,
            evaluation,
            planner,
            reflection,
            confidence,
            score,
            consistency,
            report,
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
                "evaluation_report": report,
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

        self.assertIn("Evaluation Reports:", system)
        self.assertIn("Executive summary:", system)
        self.assertTrue(
            any("Evaluation report:" in item for item in director.planning_focus),
            director.planning_focus,
        )
        self.assertTrue(
            any("Evaluation Reports" in item for item in director.critique_focus),
            director.critique_focus,
        )
        self.assertIn(
            "evaluation_report",
            {item.source for item in reasoning.evidence_chain},
        )
        self.assertTrue(
            any(
                "Evaluation Reports" in item
                for item in reasoning.implementation_guidance
            ),
            reasoning.implementation_guidance,
        )
        self.assertEqual(
            report.model_dump(mode="json")["role"],
            "evaluation_reports",
        )
        self.assertNotIn("automatic refinement enabled", system.lower())
        self.assertNotIn("runtime selection enabled", system.lower())

    def test_workflow_and_final_payload_serialize_evaluation_report(self) -> None:
        graph = build_assistant_workflow_graph()
        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(
                    query="Generate a p5.js sketch with evaluation report metadata.",
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
        planning_report = planning_event.payload["evaluation_report"]
        final_report = final_event.payload["evaluation_report"]

        self.assertEqual(planning_report["role"], "evaluation_reports")
        self.assertIn("evaluation_trace", planning_report)
        self.assertIn("evaluation_provenance", planning_report)
        self.assertIn("evaluation_dependencies", planning_report)
        self.assertIn("evidence_chain", planning_report)
        self.assertTrue(
            planning_event.payload["workflow"]["evaluation_report_available"]
        )
        self.assertEqual(
            planning_event.payload["workflow"]["evaluation_report"],
            planning_report,
        )
        self.assertEqual(final_report["role"], "evaluation_reports")
        self.assertEqual(
            final_event.payload["workflow"]["evaluation_report"],
            final_report,
        )


def _evaluation_stack(query: str):
    stack = _stack(query)
    evaluation = _self_evaluation(
        stack,
        generated_response="Here is a symbolic p5.js sketch with evaluation caveats.",
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
    return stack, evaluation, planner, reflection, confidence, score, consistency


def _report(
    stack: object,
    evaluation: object,
    planner: object,
    reflection: object,
    confidence: object,
    score: object,
    consistency: object,
):
    runtime_base = stack.runtime_base
    return derive_evaluation_report_profile(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        creative_critic=stack.creative_critic,
        self_evaluation=evaluation,
        creative_improvement_planner=planner,
        reflection_loop=reflection,
        creative_confidence=confidence,
        creative_score=score,
        consistency_validation=consistency,
        planning_metadata=(
            runtime_base.intent,
            runtime_base.hierarchy,
            runtime_base.strategy,
            runtime_base.techniques,
            runtime_base.plan,
            runtime_base.constraints,
            runtime_base.prioritization,
            runtime_base.runtime_capabilities,
            runtime_base.tradeoffs,
            getattr(runtime_base, "quality_prediction", None),
            runtime_base.artifact_plan,
            runtime_base.artifact_dependency_graph,
            runtime_base.runtime_compatibility,
            stack.artifact_capability_matrix,
            stack.multi_artifact_strategy,
            stack.artifact_critic,
            stack.artifact_refiner,
            stack.artifact_intelligence_synthesis,
            stack.artifact_merge_planner,
            stack.artifact_export_intelligence,
        ),
    )


def _director(
    stack: object,
    evaluation: object,
    planner: object,
    reflection: object,
    confidence: object,
    score: object,
    consistency: object,
    report: object,
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
        evaluation_report=report,
    )


def _reasoning(
    stack: object,
    evaluation: object,
    planner: object,
    reflection: object,
    confidence: object,
    score: object,
    consistency: object,
    report: object,
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
        evaluation_report=report,
    )


if __name__ == "__main__":
    unittest.main()
