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
    derive_creative_composition_plan,
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
    derive_procedural_structure_plan,
    derive_runtime_capability_profile,
    derive_symbolic_narrative_plan,
    procedural_structure_prompt_lines,
)


class ProceduralStructurePlannerTests(unittest.TestCase):
    def test_derives_recursive_family_and_runtime_notes(self) -> None:
        stack = _stack(
            "Generate a p5.js recursive spiral mandala with nested geometry, "
            "sacred symmetry, controlled rebirth phases, and slow orbital motion."
        )
        procedural = stack.procedural_structure

        self.assertEqual(procedural.role, "procedural_structure_planner")
        self.assertEqual(procedural.primary_structure.family, "recursive_geometry")
        self.assertIn("recursive_geometry", procedural.recommended_families)
        self.assertTrue(procedural.secondary_structures)
        self.assertTrue(
            {
                "polar_radial_systems",
                "sacred_geometry_pattern_systems",
            }.intersection({item.family for item in procedural.secondary_structures}),
            procedural.secondary_structures,
        )
        self.assertIn("spine", procedural.combination_strategy.lower())
        self.assertIn("recursive", procedural.spatial_structure_plan.lower())
        self.assertIn("animate", procedural.temporal_structure_plan.lower())
        self.assertTrue(
            any("p5" in item.lower() for item in procedural.runtime_suitability_notes),
            procedural.runtime_suitability_notes,
        )
        self.assertTrue(procedural.fallback_structure_options)
        self.assertTrue(procedural_structure_prompt_lines(procedural))

    def test_derives_particle_structure_with_fallbacks_and_risks(self) -> None:
        stack = _stack(
            "Generate a p5.js phoenix that dissolves into ember particles, "
            "flows through turbulent noise, then reforms from a storm of sparks."
        )
        procedural = stack.procedural_structure

        self.assertEqual(procedural.primary_structure.family, "particle_systems")
        self.assertIn("particle_systems", procedural.recommended_families)
        self.assertTrue(
            any(
                item.family in {"noise_fields", "flow_fields"}
                for item in procedural.secondary_structures
            ),
            procedural.secondary_structures,
        )
        self.assertTrue(procedural.performance_risks)
        self.assertTrue(procedural.implementation_risks)
        self.assertTrue(procedural.fallback_structure_options)
        self.assertTrue(
            any(
                "count" in item.lower() or "cost" in item.lower()
                for item in procedural.performance_risks
            ),
            procedural.performance_risks,
        )

    def test_surfaces_ambiguous_procedural_gaps_and_hitl(self) -> None:
        stack = _stack(
            "Make something profound and maybe interactive or audio-reactive, "
            "with a deep vibe, but keep it simple and cinematic."
        )
        procedural = stack.procedural_structure

        self.assertTrue(procedural.unresolved_procedural_gaps)
        self.assertTrue(procedural.hitl_questions)
        self.assertIsNotNone(procedural.interaction_structure_plan)
        self.assertIsNotNone(procedural.audiovisual_structure_plan)
        self.assertTrue(
            any("gesture" in item.lower() for item in procedural.hitl_questions),
            procedural.hitl_questions,
        )
        self.assertTrue(
            any(
                "beat" in item.lower() or "tempo" in item.lower()
                for item in procedural.hitl_questions
            ),
            procedural.hitl_questions,
        )

    def test_clips_long_retrieval_answer_structure_rationales(self) -> None:
        stack = _stack(
            "Answer a creative-coding runtime question with registered source "
            "grounding. Explain which retrieved sources shaped the response, "
            "what the source boundaries are, and how the answer should be "
            "validated before using it in a browser sketch."
        )
        procedural = stack.procedural_structure
        choices = (
            procedural.primary_structure,
            *procedural.secondary_structures,
            *procedural.fallback_structure_options,
        )

        self.assertTrue(choices)
        self.assertTrue(
            all(len(choice.rationale) <= 320 for choice in choices),
            [choice.rationale for choice in choices],
        )
        self.assertLessEqual(len(procedural.combination_strategy), 360)
        self.assertLessEqual(len(procedural.spatial_structure_plan), 360)
        self.assertLessEqual(len(procedural.temporal_structure_plan), 360)
        if procedural.interaction_structure_plan is not None:
            self.assertLessEqual(len(procedural.interaction_structure_plan), 320)
        if procedural.audiovisual_structure_plan is not None:
            self.assertLessEqual(len(procedural.audiovisual_structure_plan), 320)

    def test_integrates_with_prompt_director_and_reasoning_metadata(self) -> None:
        stack = _stack(
            "Generate a symbolic recursive spiral threshold in p5.js with "
            "sacred geometry, orbiting rings, and a visible resolution phase."
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
                "creative_composition": stack.creative_composition,
                "procedural_structure": stack.procedural_structure,
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

        self.assertIn("Procedural Structure Planner:", system)
        self.assertIn("Primary procedural structure:", system)
        self.assertTrue(
            any(
                "Procedural structure:" in item
                for item in stack.director.planning_focus
            ),
            stack.director.planning_focus,
        )
        self.assertIn(
            "procedural_structure",
            {item.source for item in stack.reasoning.evidence_chain},
        )
        self.assertIn(
            "Structure procedurally as",
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
    creative_composition: object
    procedural_structure: object
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
    creative_composition = derive_creative_composition_plan(
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
        symbolic_narrative=symbolic_narrative,
    )
    procedural_structure = derive_procedural_structure_plan(
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
        symbolic_narrative=symbolic_narrative,
        creative_composition=creative_composition,
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
        creative_composition=creative_composition,
        procedural_structure=procedural_structure,
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
        creative_composition=creative_composition,
        procedural_structure=procedural_structure,
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
        creative_composition=creative_composition,
        procedural_structure=procedural_structure,
        director=director,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    unittest.main()
