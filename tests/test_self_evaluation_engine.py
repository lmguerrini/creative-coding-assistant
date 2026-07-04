import unittest

from creative_coding_assistant.contracts import StreamEventType
from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    WorkflowArtifact,
    build_assistant_workflow_graph,
    build_rendered_prompt_request,
    derive_creative_assistant_director_brief,
    derive_creative_reasoning_result,
    derive_self_evaluation_profile,
    self_evaluation_prompt_lines,
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


class SelfEvaluationEngineTests(unittest.TestCase):
    def test_derives_high_alignment_response_context(self) -> None:
        stack = _stack(
            "Generate a luminous p5.js mandala with hierarchy, motion, "
            "runtime caveats, and artifact notes."
        )
        evaluation = _derive(
            stack,
            generated_response=(
                "Here is a luminous p5.js mandala sketch with radial hierarchy, "
                "gentle motion, runtime caveats, and explicit artifact notes."
            ),
        )

        self.assertEqual(evaluation.role, "self_evaluation_engine")
        self.assertGreaterEqual(evaluation.request_alignment, 0.6)
        self.assertIn(
            evaluation.completeness_assessment,
            {"partial", "mostly_complete", "complete"},
        )
        self.assertIn(evaluation.hallucination_risk, {"low", "medium"})
        self.assertIn("does not modify outputs", evaluation.authority_boundary)
        self.assertTrue(self_evaluation_prompt_lines(evaluation))

    def test_derives_low_alignment_response_context(self) -> None:
        stack = _stack(
            "Generate a p5.js phoenix mandala with runtime caveats and artifact notes."
        )
        evaluation = _derive(
            stack,
            generated_response=(
                "This response explains a spreadsheet macro and deployment "
                "workflow without addressing the requested sketch."
            ),
        )

        self.assertLess(evaluation.request_alignment, 0.75)
        self.assertTrue(evaluation.quality_gaps)
        self.assertTrue(evaluation.improvement_opportunities)

    def test_handles_missing_information_scenario(self) -> None:
        stack = _stack("Evaluate an underspecified creative coding response.")
        evaluation = derive_self_evaluation_profile(
            request=stack.runtime_base.request,
            route_decision=None,
        )

        self.assertEqual(evaluation.completeness_assessment, "blocked")
        self.assertEqual(evaluation.ambiguity_assessment, "high")
        self.assertTrue(evaluation.missing_information)
        self.assertTrue(evaluation.hitl_questions)

    def test_flags_hallucination_risk_scenario(self) -> None:
        stack = _stack(
            "Generate a fragile multi-runtime artifact with unsupported runtime caveats."
        )
        risky_critic = stack.creative_critic.model_copy(
            update={
                "risk_assessment": "high",
                "unsupported_assumptions": (
                    "Unsupported runtime remains non-viable metadata: glsl.",
                    "Runtime availability remains unverified by the plan.",
                ),
            }
        )
        evaluation = _derive(
            stack,
            creative_critic=risky_critic,
            generated_response=(
                "This production-ready artifact is guaranteed tested and runs "
                "everywhere across all runtimes with no caveats."
            ),
        )

        self.assertEqual(evaluation.hallucination_risk, "high")
        self.assertTrue(evaluation.unsupported_assumptions)
        self.assertTrue(evaluation.hitl_questions)

    def test_supports_artifact_aware_self_evaluation(self) -> None:
        stack = _stack("Generate a p5.js mandala artifact for self evaluation.")
        artifact = WorkflowArtifact(
            id="artifact_1",
            title="Mandala sketch",
            name="mandala.js",
            language="javascript",
            source_language="javascript",
            content="function setup() { createCanvas(400, 400); }",
            summary="A bounded p5.js mandala artifact.",
            source_order=1,
            domain="p5_js",
            is_creative=True,
            preview_eligible=True,
            runtime="p5",
            renderer_id="p5",
            preview_target="p5",
            content_hash="artifacthash",
            quality_score=0.86,
        )
        evaluation = _derive(
            stack,
            generated_response="Here is a p5.js mandala artifact with caveats.",
            artifacts=(artifact,),
        )

        self.assertGreaterEqual(evaluation.artifact_alignment, 0.6)
        self.assertTrue(
            any("Artifacts available:" in item for item in evaluation.evidence)
        )

    def test_supports_critic_aware_self_evaluation(self) -> None:
        stack = _stack("Generate a p5.js artifact with critic-aware evaluation.")
        evaluation = _derive(stack)

        self.assertTrue(
            any("Creative critic:" in item for item in evaluation.evidence),
            evaluation.evidence,
        )
        self.assertTrue(evaluation.unsupported_assumptions)
        self.assertTrue(evaluation.prompt_guidance)

    def test_integrates_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack(
            "Generate a symbolic p5.js phoenix mandala with Self Evaluation "
            "metadata after Creative Critic."
        )
        evaluation = _derive(
            stack,
            generated_response=(
                "Here is a symbolic p5.js phoenix mandala with runtime caveats."
            ),
        )
        director = _director(stack, evaluation)
        reasoning = _reasoning(stack, evaluation, director)
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

        self.assertIn("Self Evaluation Engine:", system)
        self.assertIn("Self evaluation status:", system)
        self.assertTrue(
            any("Self evaluation:" in item for item in director.planning_focus),
            director.planning_focus,
        )
        self.assertIn(
            "self_evaluation",
            {item.source for item in reasoning.evidence_chain},
        )
        self.assertEqual(
            evaluation.model_dump(mode="json")["role"],
            "self_evaluation_engine",
        )
        self.assertNotIn("reflection loop enabled", system.lower())
        self.assertNotIn("runtime auto-selection enabled", system.lower())

    def test_workflow_and_final_payload_serialize_self_evaluation(self) -> None:
        graph = build_assistant_workflow_graph()
        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(
                    query="Generate a p5.js sketch with self evaluation metadata.",
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
        planning_evaluation = planning_event.payload["self_evaluation"]
        final_evaluation = final_event.payload["self_evaluation"]

        self.assertEqual(planning_evaluation["role"], "self_evaluation_engine")
        self.assertTrue(planning_event.payload["workflow"]["self_evaluation_available"])
        self.assertEqual(
            planning_event.payload["workflow"]["self_evaluation"],
            planning_evaluation,
        )
        self.assertEqual(final_evaluation["role"], "self_evaluation_engine")
        self.assertEqual(
            final_event.payload["workflow"]["self_evaluation"],
            final_evaluation,
        )


def _derive(
    stack: object,
    **overrides: object,
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
        creative_quality_prediction=overrides.pop(
            "creative_quality_prediction",
            getattr(runtime_base, "quality_prediction", None),
        ),
        artifact_plan=runtime_base.artifact_plan,
        artifact_dependency_graph=runtime_base.artifact_dependency_graph,
        runtime_compatibility=runtime_base.runtime_compatibility,
        artifact_capability_matrix=stack.artifact_capability_matrix,
        multi_artifact_strategy=stack.multi_artifact_strategy,
        artifact_critic=overrides.pop("artifact_critic", stack.artifact_critic),
        artifact_refiner=overrides.pop("artifact_refiner", stack.artifact_refiner),
        artifact_intelligence_synthesis=stack.artifact_intelligence_synthesis,
        artifact_merge_planner=stack.artifact_merge_planner,
        artifact_export_intelligence=stack.artifact_export_intelligence,
        creative_critic=overrides.pop("creative_critic", stack.creative_critic),
        generated_response=overrides.pop("generated_response", None),
        artifacts=overrides.pop("artifacts", ()),
    )


def _director(stack: object, evaluation: object):
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
    )


def _reasoning(stack: object, evaluation: object, director: object):
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
    )


if __name__ == "__main__":
    unittest.main()
