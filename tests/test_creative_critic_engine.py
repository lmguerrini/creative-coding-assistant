import unittest
from dataclasses import dataclass

from creative_coding_assistant.contracts import StreamEventType
from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    WorkflowArtifact,
    build_assistant_workflow_graph,
    build_rendered_prompt_request,
    creative_critic_prompt_lines,
    derive_creative_assistant_director_brief,
    derive_creative_critic_profile,
    derive_creative_reasoning_result,
    stream_assistant_workflow_events,
)
from test_artifact_export_intelligence import _stack as _export_stack
from test_langgraph_workflow_integration import (
    _first_event,
    _request,
    _runtime,
    _stream_completed_generation,
    _stream_prompt_inputs_with_builder,
)


class CreativeCriticEngineTests(unittest.TestCase):
    def test_derives_strong_creative_plan_critique(self) -> None:
        stack = _stack(
            "Generate a luminous p5.js mandala with clear hierarchy, motion, "
            "runtime caveats, and artifact packaging notes."
        )
        critic = stack.creative_critic

        self.assertEqual(critic.role, "creative_critic_engine")
        self.assertGreater(critic.critic_confidence, 0)
        self.assertTrue(critic.critique_summary)
        self.assertTrue(critic.creative_strengths)
        self.assertGreaterEqual(critic.concept_quality, 0.5)
        self.assertGreaterEqual(critic.artifact_quality, 0.5)
        self.assertIn(critic.risk_assessment, {"low", "medium"})
        self.assertIn("does not modify artifacts", critic.authority_boundary)
        self.assertTrue(creative_critic_prompt_lines(critic))

    def test_derives_weak_creative_plan_critique(self) -> None:
        stack = _stack(
            "Generate a dense, vague, high-complexity visual with uncertain "
            "runtime and no clear constraints."
        )
        weak_artifact_critic = stack.artifact_critic.model_copy(
            update={
                "risk_assessment": "medium",
                "weaknesses": (
                    "Runtime and complexity pressure can overwhelm the output.",
                    "Intent clarity is weak for the requested scope.",
                ),
                "missing_information": (
                    "Creative output success criteria are underspecified.",
                ),
            }
        )
        critic = _derive(stack, artifact_critic=weak_artifact_critic)

        self.assertIn(critic.risk_assessment, {"medium", "high"})
        self.assertTrue(critic.creative_weaknesses)
        self.assertTrue(critic.improvement_opportunities)

    def test_handles_missing_information_scenario(self) -> None:
        stack = _stack("Explain a possible creative coding artifact.")
        critic = derive_creative_critic_profile(
            request=stack.runtime_base.request,
            route_decision=None,
        )

        self.assertEqual(critic.risk_assessment, "blocked")
        self.assertTrue(critic.missing_information)
        self.assertTrue(critic.hitl_questions)

    def test_flags_high_risk_scenario(self) -> None:
        stack = _stack(
            "Generate a fragile multi-runtime artifact with unsupported runtime "
            "assumptions."
        )
        high_risk_artifact_critic = stack.artifact_critic.model_copy(
            update={
                "risk_assessment": "high",
                "weaknesses": (
                    "Unsupported runtime assumptions are central to the plan.",
                    "Artifact critique has unresolved execution caveats.",
                ),
                "runtime_concerns": (
                    "Unsupported runtimes should not be treated as viable targets.",
                ),
            }
        )
        critic = _derive(stack, artifact_critic=high_risk_artifact_critic)

        self.assertEqual(critic.risk_assessment, "high")
        self.assertTrue(critic.unsupported_assumptions)
        self.assertTrue(critic.hitl_questions)

    def test_supports_artifact_aware_critique(self) -> None:
        stack = _stack("Generate a p5.js artifact for critic inspection.")
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
            quality_score=0.84,
        )
        critic = _derive(
            stack,
            generated_response="Here is a p5.js mandala artifact.",
            artifacts=(artifact,),
        )

        self.assertTrue(
            any("1 generated artifact" in item for item in critic.creative_strengths),
            critic.creative_strengths,
        )
        self.assertTrue(any("Artifacts available:" in item for item in critic.evidence))
        self.assertGreaterEqual(critic.artifact_quality, 0.6)

    def test_integrates_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack(
            "Generate a symbolic p5.js phoenix mandala with Creative Critic "
            "metadata before Director and Reasoning."
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
                "creative_director": stack.director,
                "creative_reasoning": stack.reasoning,
            }
        )
        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=stack.runtime_base.route,
                prompt_input=prompt_input,
            )
        )
        system = rendered.sections[0].content

        self.assertIn("Creative Critic Engine:", system)
        self.assertIn("Creative critic risk assessment:", system)
        self.assertTrue(
            any(
                "Creative critic:" in item
                for item in stack.director.planning_focus
            ),
            stack.director.planning_focus,
        )
        self.assertIn(
            "creative_critic",
            {item.source for item in stack.reasoning.evidence_chain},
        )
        self.assertEqual(
            stack.creative_critic.model_dump(mode="json")["role"],
            "creative_critic_engine",
        )
        self.assertNotIn("runtime auto-selection enabled", system.lower())
        self.assertNotIn("provider routing enabled", system.lower())

    def test_workflow_and_final_payload_serialize_creative_critic(self) -> None:
        graph = build_assistant_workflow_graph()
        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(
                    query="Generate a p5.js sketch with creative critic metadata.",
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
        critic = planning_event.payload["creative_critic"]

        self.assertEqual(critic["role"], "creative_critic_engine")
        self.assertTrue(planning_event.payload["workflow"]["creative_critic_available"])
        self.assertEqual(planning_event.payload["workflow"]["creative_critic"], critic)
        self.assertEqual(final_event.payload["creative_critic"], critic)
        self.assertEqual(final_event.payload["workflow"]["creative_critic"], critic)


@dataclass(frozen=True)
class _DerivedStack:
    export_stack: object
    runtime_base: object
    artifact_capability_matrix: object
    multi_artifact_strategy: object
    artifact_critic: object
    artifact_refiner: object
    artifact_intelligence_synthesis: object
    artifact_merge_planner: object
    artifact_export_intelligence: object
    creative_critic: object
    director: object
    reasoning: object


def _stack(query: str) -> _DerivedStack:
    export_stack = _export_stack(query)
    raw_artifact_critic = (
        export_stack.merge_stack.synthesis_stack.refiner_stack.critic_stack.artifact_critic
    )
    artifact_critic = raw_artifact_critic.model_copy(
        update={
            "risk_assessment": "medium",
            "weaknesses": raw_artifact_critic.weaknesses[:2],
            "runtime_concerns": raw_artifact_critic.runtime_concerns[:1],
        }
    )
    artifact_refiner = (
        export_stack.merge_stack.synthesis_stack.refiner_stack.artifact_refiner
    )
    creative_critic = _derive(
        export_stack,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
    )
    director = derive_creative_assistant_director_brief(
        request=export_stack.runtime_base.request,
        route_decision=export_stack.runtime_base.route,
        creative_translation=export_stack.runtime_base.prompt_input.creative_translation,
        creative_intent=export_stack.runtime_base.intent,
        creative_hierarchy=export_stack.runtime_base.hierarchy,
        creative_strategy=export_stack.runtime_base.strategy,
        creative_techniques=export_stack.runtime_base.techniques,
        creative_plan=export_stack.runtime_base.plan,
        creative_constraints=export_stack.runtime_base.constraints,
        creative_constraint_priorities=export_stack.runtime_base.prioritization,
        runtime_capabilities=export_stack.runtime_base.runtime_capabilities,
        creative_tradeoffs=export_stack.runtime_base.tradeoffs,
        creative_quality_prediction=getattr(
            export_stack.runtime_base,
            "quality_prediction",
            None,
        ),
        artifact_plan=export_stack.runtime_base.artifact_plan,
        artifact_dependency_graph=export_stack.runtime_base.artifact_dependency_graph,
        runtime_compatibility=export_stack.runtime_base.runtime_compatibility,
        artifact_capability_matrix=export_stack.artifact_capability_matrix,
        multi_artifact_strategy=export_stack.multi_artifact_strategy,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
        artifact_intelligence_synthesis=(
            export_stack.artifact_intelligence_synthesis
        ),
        artifact_merge_planner=export_stack.artifact_merge_planner,
        artifact_export_intelligence=export_stack.artifact_export_intelligence,
        creative_critic=creative_critic,
    )
    reasoning = derive_creative_reasoning_result(
        request=export_stack.runtime_base.request,
        route_decision=export_stack.runtime_base.route,
        creative_translation=export_stack.runtime_base.prompt_input.creative_translation,
        creative_intent=export_stack.runtime_base.intent,
        creative_hierarchy=export_stack.runtime_base.hierarchy,
        creative_plan=export_stack.runtime_base.plan,
        creative_director=director,
        creative_constraints=export_stack.runtime_base.constraints,
        creative_constraint_priorities=export_stack.runtime_base.prioritization,
        creative_strategy=export_stack.runtime_base.strategy,
        creative_techniques=export_stack.runtime_base.techniques,
        runtime_capabilities=export_stack.runtime_base.runtime_capabilities,
        creative_tradeoffs=export_stack.runtime_base.tradeoffs,
        creative_quality_prediction=getattr(
            export_stack.runtime_base,
            "quality_prediction",
            None,
        ),
        artifact_plan=export_stack.runtime_base.artifact_plan,
        artifact_dependency_graph=export_stack.runtime_base.artifact_dependency_graph,
        runtime_compatibility=export_stack.runtime_base.runtime_compatibility,
        artifact_capability_matrix=export_stack.artifact_capability_matrix,
        multi_artifact_strategy=export_stack.multi_artifact_strategy,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
        artifact_intelligence_synthesis=(
            export_stack.artifact_intelligence_synthesis
        ),
        artifact_merge_planner=export_stack.artifact_merge_planner,
        artifact_export_intelligence=export_stack.artifact_export_intelligence,
        creative_critic=creative_critic,
    )
    return _DerivedStack(
        export_stack=export_stack,
        runtime_base=export_stack.runtime_base,
        artifact_capability_matrix=export_stack.artifact_capability_matrix,
        multi_artifact_strategy=export_stack.multi_artifact_strategy,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
        artifact_intelligence_synthesis=export_stack.artifact_intelligence_synthesis,
        artifact_merge_planner=export_stack.artifact_merge_planner,
        artifact_export_intelligence=export_stack.artifact_export_intelligence,
        creative_critic=creative_critic,
        director=director,
        reasoning=reasoning,
    )


def _derive(
    stack: object,
    **overrides: object,
):
    runtime_base = stack.runtime_base
    return derive_creative_critic_profile(
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
        artifact_critic=overrides.pop(
            "artifact_critic",
            getattr(stack, "artifact_critic", None),
        ),
        artifact_refiner=overrides.pop(
            "artifact_refiner",
            getattr(stack, "artifact_refiner", None),
        ),
        artifact_intelligence_synthesis=stack.artifact_intelligence_synthesis,
        artifact_merge_planner=stack.artifact_merge_planner,
        artifact_export_intelligence=stack.artifact_export_intelligence,
        generated_response=overrides.pop("generated_response", None),
        artifacts=overrides.pop("artifacts", ()),
    )


if __name__ == "__main__":
    unittest.main()
