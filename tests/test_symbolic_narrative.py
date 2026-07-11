import unittest
from dataclasses import dataclass

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    RouteCapability,
    RouteDecision,
    RouteName,
    StructuredPromptInputBuilder,
    build_prompt_input_request,
    build_rendered_prompt_request,
    derive_creative_assistant_director_brief,
    derive_creative_constraint_priorities,
    derive_creative_constraint_solution,
    derive_creative_execution_plan,
    derive_creative_hierarchy_plan,
    derive_creative_intent_decomposition,
    derive_creative_quality_prediction,
    derive_creative_reasoning_result,
    derive_creative_strategy_profile,
    derive_creative_technique_profile,
    derive_creative_tradeoff_profile,
    derive_runtime_capability_profile,
    derive_symbolic_narrative_plan,
    symbolic_narrative_prompt_lines,
)


class SymbolicNarrativePlannerTests(unittest.TestCase):
    def test_derives_phased_symbolic_arc_with_progression(self) -> None:
        stack = _stack(
            "Generate a p5.js dark-to-light phoenix rebirth ritual with "
            "fragmented geometry, gold sparks, slow audio pulse, and a "
            "visible transformation climax."
        )
        narrative = stack.symbolic_narrative

        self.assertEqual(narrative.role, "symbolic_narrative_planner")
        self.assertEqual(narrative.narrative_archetype, "death_and_rebirth")
        self.assertIn("without unsupported doctrine", narrative.symbolic_arc)
        self.assertEqual(
            [phase.phase for phase in narrative.phases],
            ["opening", "development", "threshold", "climax", "resolution"],
        )
        self.assertEqual(len({phase.title for phase in narrative.phases}), 5)
        self.assertEqual(len(narrative.symbolic_transitions), 4)
        self.assertEqual(len(narrative.emotional_progression), 5)
        self.assertEqual(len(narrative.visual_progression), 5)
        self.assertEqual(len(narrative.motion_progression), 5)
        self.assertEqual(len(narrative.audio_progression), 5)
        self.assertTrue(narrative.prompt_guidance)
        self.assertTrue(symbolic_narrative_prompt_lines(narrative))
        self.assertNotEqual(
            narrative.opening_phase.visual_state,
            narrative.climax_phase.visual_state,
        )

    def test_surfaces_ambiguous_symbolic_gaps_and_hitl_questions(self) -> None:
        stack = _stack(
            "Make something profound and symbolic, maybe interactive or "
            "audio-reactive, but keep it simple and cinematic."
        )
        narrative = stack.symbolic_narrative

        self.assertEqual(narrative.narrative_archetype, "symbolic_vignette")
        self.assertTrue(narrative.unresolved_narrative_gaps)
        self.assertTrue(narrative.hitl_questions)
        self.assertTrue(
            any(
                "transformation" in item.lower()
                or "journey" in item.lower()
                or "audio" in item.lower()
                for item in narrative.unresolved_narrative_gaps
            ),
            narrative.unresolved_narrative_gaps,
        )
        self.assertTrue(
            any("?" in item for item in narrative.hitl_questions),
            narrative.hitl_questions,
        )

    def test_clips_long_symbolic_arc_context_to_its_contract_limit(self) -> None:
        stack = _stack("Generate a bounded p5.js threshold visual.")
        long_intent = stack.intent.model_copy(
            update={
                "primary_expression": (
                    "Preserve a layered descent through a luminous threshold with "
                    "bounded visual change and clear interactive cues. "
                    * 4
                )[:360]
            }
        )

        narrative = derive_symbolic_narrative_plan(
            request=stack.request,
            route_decision=stack.route,
            creative_translation=stack.prompt_input.creative_translation,
            creative_intent=long_intent,
            creative_hierarchy=stack.hierarchy,
            creative_plan=stack.plan,
            creative_constraints=stack.constraints,
            creative_constraint_priorities=stack.prioritization,
            creative_strategy=stack.strategy,
            creative_techniques=stack.techniques,
            runtime_capabilities=stack.runtime_capabilities,
            creative_tradeoffs=stack.tradeoffs,
            creative_quality_prediction=stack.quality_prediction,
        )

        self.assertLessEqual(len(narrative.symbolic_arc), 420)
        self.assertIn("without unsupported doctrine", narrative.symbolic_arc)

    def test_integrates_with_prompt_director_and_reasoning_metadata(self) -> None:
        stack = _stack(
            "Generate a symbolic spiral threshold crossing in p5.js with "
            "sacred geometry, slow orbiting motion, and an explicit resolution."
        )
        prompt_input = stack.prompt_input.model_copy(
            update={
                "creative_intent": stack.intent,
                "creative_hierarchy": stack.hierarchy,
                "creative_strategy": stack.strategy,
                "creative_techniques": stack.techniques,
                "creative_plan": stack.plan,
                "creative_constraints": stack.constraints,
                "creative_constraint_priorities": stack.prioritization,
                "runtime_capabilities": stack.runtime_capabilities,
                "creative_tradeoffs": stack.tradeoffs,
                "creative_quality_prediction": stack.quality_prediction,
                "symbolic_narrative": stack.symbolic_narrative,
                "creative_director": stack.director,
                "creative_reasoning": stack.reasoning,
            }
        )

        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=stack.route,
                prompt_input=prompt_input,
            )
        )
        system = rendered.sections[0].content

        self.assertIn("Symbolic Narrative Planner:", system)
        self.assertIn("Narrative guidance:", system)
        self.assertTrue(
            any("Narrative arc:" in item for item in stack.director.planning_focus),
            stack.director.planning_focus,
        )
        self.assertIn(
            "symbolic_narrative",
            {item.source for item in stack.reasoning.evidence_chain},
        )
        self.assertIn(
            "Shape symbolic arc:",
            stack.reasoning.recommended_creative_direction,
        )
        self.assertNotIn("runtime auto-selection enabled", system)


@dataclass(frozen=True)
class _DerivedStack:
    request: AssistantRequest
    route: RouteDecision
    prompt_input: object
    intent: object
    hierarchy: object
    strategy: object
    techniques: object
    plan: object
    constraints: object
    runtime_capabilities: object
    tradeoffs: object
    prioritization: object
    quality_prediction: object
    symbolic_narrative: object
    director: object
    reasoning: object


def _stack(query: str) -> _DerivedStack:
    request = AssistantRequest(
        query=query,
        mode=AssistantMode.GENERATE,
        domain=CreativeCodingDomain.P5_JS,
    )
    route = RouteDecision(
        route=RouteName.GENERATE,
        mode=AssistantMode.GENERATE,
        domain=CreativeCodingDomain.P5_JS,
        domains=(CreativeCodingDomain.P5_JS,),
        capabilities=(RouteCapability.TOOL_USE,),
    )
    prompt_input = StructuredPromptInputBuilder().build(
        build_prompt_input_request(
            assistant_request=request,
            route_decision=route,
            assembled_context=None,
        )
    )
    intent = derive_creative_intent_decomposition(
        request=request,
        route_decision=route,
        creative_translation=prompt_input.creative_translation,
    )
    hierarchy = derive_creative_hierarchy_plan(
        request=request,
        route_decision=route,
        creative_intent=intent,
        creative_translation=prompt_input.creative_translation,
    )
    strategy = derive_creative_strategy_profile(
        request=request,
        route_decision=route,
        creative_intent=intent,
        creative_hierarchy=hierarchy,
        creative_translation=prompt_input.creative_translation,
    )
    techniques = derive_creative_technique_profile(
        request=request,
        route_decision=route,
        creative_intent=intent,
        creative_hierarchy=hierarchy,
        creative_translation=prompt_input.creative_translation,
        creative_strategy=strategy,
    )
    plan = derive_creative_execution_plan(
        request=request,
        route_decision=route,
        creative_intent=intent,
        creative_hierarchy=hierarchy,
        creative_translation=prompt_input.creative_translation,
        creative_strategy=strategy,
        creative_techniques=techniques,
    )
    constraints = derive_creative_constraint_solution(
        request=request,
        route_decision=route,
        creative_intent=intent,
        creative_hierarchy=hierarchy,
        creative_translation=prompt_input.creative_translation,
        creative_plan=plan,
        creative_strategy=strategy,
        creative_techniques=techniques,
    )
    runtime_capabilities = derive_runtime_capability_profile(
        request=request,
        route_decision=route,
        creative_translation=prompt_input.creative_translation,
        creative_strategy=strategy,
        creative_techniques=techniques,
        creative_plan=plan,
        creative_constraints=constraints,
    )
    tradeoffs = derive_creative_tradeoff_profile(
        request=request,
        route_decision=route,
        creative_translation=prompt_input.creative_translation,
        creative_strategy=strategy,
        creative_techniques=techniques,
        creative_plan=plan,
        creative_constraints=constraints,
        runtime_capabilities=runtime_capabilities,
    )
    prioritization = derive_creative_constraint_priorities(
        request=request,
        route_decision=route,
        creative_translation=prompt_input.creative_translation,
        creative_intent=intent,
        creative_hierarchy=hierarchy,
        creative_plan=plan,
        creative_constraints=constraints,
        creative_strategy=strategy,
        creative_techniques=techniques,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=tradeoffs,
    )
    quality_prediction = derive_creative_quality_prediction(
        request=request,
        route_decision=route,
        creative_translation=prompt_input.creative_translation,
        creative_intent=intent,
        creative_hierarchy=hierarchy,
        creative_plan=plan,
        creative_constraints=constraints,
        creative_constraint_priorities=prioritization,
        creative_strategy=strategy,
        creative_techniques=techniques,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=tradeoffs,
    )
    symbolic_narrative = derive_symbolic_narrative_plan(
        request=request,
        route_decision=route,
        creative_translation=prompt_input.creative_translation,
        creative_intent=intent,
        creative_hierarchy=hierarchy,
        creative_plan=plan,
        creative_constraints=constraints,
        creative_constraint_priorities=prioritization,
        creative_strategy=strategy,
        creative_techniques=techniques,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=tradeoffs,
        creative_quality_prediction=quality_prediction,
    )
    director = derive_creative_assistant_director_brief(
        request=request,
        route_decision=route,
        creative_translation=prompt_input.creative_translation,
        creative_intent=intent,
        creative_hierarchy=hierarchy,
        creative_strategy=strategy,
        creative_techniques=techniques,
        creative_plan=plan,
        creative_constraints=constraints,
        creative_constraint_priorities=prioritization,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=tradeoffs,
        creative_quality_prediction=quality_prediction,
        symbolic_narrative=symbolic_narrative,
    )
    reasoning = derive_creative_reasoning_result(
        request=request,
        route_decision=route,
        creative_translation=prompt_input.creative_translation,
        creative_intent=intent,
        creative_hierarchy=hierarchy,
        creative_plan=plan,
        creative_director=director,
        creative_constraints=constraints,
        creative_constraint_priorities=prioritization,
        creative_strategy=strategy,
        creative_techniques=techniques,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=tradeoffs,
        creative_quality_prediction=quality_prediction,
        symbolic_narrative=symbolic_narrative,
    )
    return _DerivedStack(
        request=request,
        route=route,
        prompt_input=prompt_input,
        intent=intent,
        hierarchy=hierarchy,
        strategy=strategy,
        techniques=techniques,
        plan=plan,
        constraints=constraints,
        runtime_capabilities=runtime_capabilities,
        tradeoffs=tradeoffs,
        prioritization=prioritization,
        quality_prediction=quality_prediction,
        symbolic_narrative=symbolic_narrative,
        director=director,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    unittest.main()
