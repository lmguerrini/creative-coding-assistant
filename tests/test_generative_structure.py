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
    derive_generative_structure_blueprint,
    derive_procedural_structure_plan,
    derive_runtime_capability_profile,
    derive_symbolic_narrative_plan,
    generative_structure_prompt_lines,
)


class GenerativeStructureEngineTests(unittest.TestCase):
    def test_derives_modules_parameters_and_evolution_rules(self) -> None:
        stack = _stack(
            "Generate a p5.js recursive spiral phoenix mandala with particles "
            "dissolving into embers and reassembling through a radial rebirth."
        )
        blueprint = stack.generative_structure
        module_kinds = {module.kind for module in blueprint.procedural_modules}
        relationship_types = {
            relationship.relationship_type
            for relationship in blueprint.module_relationships
        }
        parameter_names = {parameter.name for parameter in blueprint.parameter_schema}
        evolution_phases = {rule.phase for rule in blueprint.evolution_rules}

        self.assertEqual(blueprint.role, "generative_structure_engine")
        self.assertTrue(blueprint.blueprint_name)
        self.assertIn("seed_system", module_kinds)
        self.assertIn("recursive_module", module_kinds)
        self.assertIn("particle_emitter", module_kinds)
        self.assertIn("geometry_reassembly_layer", module_kinds)
        self.assertTrue(
            {"attracts", "reassembles"}.intersection(relationship_types),
            relationship_types,
        )
        self.assertTrue(
            {"recursion_depth", "particle_count", "reassembly_speed"}.issubset(
                parameter_names
            ),
            parameter_names,
        )
        self.assertTrue(
            {"fragmentation", "reassembly", "stabilization"}.issubset(evolution_phases),
            evolution_phases,
        )
        self.assertTrue(generative_structure_prompt_lines(blueprint))

    def test_derives_runtime_guidance_safeguards_and_fallback(self) -> None:
        stack = _stack(
            "Generate a p5.js phoenix that dissolves into ember particles, "
            "flows through turbulent noise, then reforms from a storm of sparks."
        )
        blueprint = stack.generative_structure
        guidance = " ".join(blueprint.runtime_implementation_guidance).lower()
        safeguards = " ".join(blueprint.performance_safeguards).lower()
        reductions = " ".join(blueprint.fallback_blueprint.parameter_reductions).lower()

        self.assertIn("feasibility guidance only", guidance)
        self.assertIn("p5", guidance)
        self.assertIn("particle", safeguards)
        self.assertIn("frame_budget_ms", safeguards)
        self.assertTrue(blueprint.fallback_blueprint.module_kinds)
        self.assertTrue(
            "max_particle_count" in reductions
            or "module count" in reductions
            or "frame_budget_ms" in reductions,
            blueprint.fallback_blueprint.parameter_reductions,
        )

    def test_surfaces_ambiguous_hooks_and_hitl_questions(self) -> None:
        stack = _stack(
            "Make something profound and maybe interactive or audio-reactive, "
            "with a deep vibe, but keep it simple and cinematic."
        )
        blueprint = stack.generative_structure

        self.assertTrue(blueprint.interaction_hooks)
        self.assertTrue(blueprint.audiovisual_hooks)
        self.assertTrue(blueprint.unresolved_implementation_gaps)
        self.assertTrue(blueprint.hitl_questions)
        self.assertTrue(
            any("gesture" in item.lower() for item in blueprint.hitl_questions),
            blueprint.hitl_questions,
        )
        self.assertTrue(
            any(
                "beat" in item.lower()
                or "pulse" in item.lower()
                or "audio" in item.lower()
                for item in blueprint.hitl_questions
            ),
            blueprint.hitl_questions,
        )

    def test_bounds_seed_module_purpose_for_long_intent_context(self) -> None:
        stack = _stack("Generate a bounded p5.js threshold visual.")
        long_intent = stack.intent.model_copy(
            update={
                "primary_expression": (
                    "Define a pointer-led Chladni field with clear visual intent and "
                    "one runnable browser artifact. "
                    * 4
                )[:360]
            }
        )
        blueprint = derive_generative_structure_blueprint(
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
            symbolic_narrative=stack.symbolic_narrative,
            creative_composition=stack.creative_composition,
            procedural_structure=stack.procedural_structure,
        )

        seed_module = next(
            module for module in blueprint.procedural_modules if module.kind == "seed_system"
        )
        self.assertLessEqual(len(seed_module.purpose), 360)

    def test_integrates_with_prompt_director_reasoning_and_serialization(
        self,
    ) -> None:
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
                "generative_structure": stack.generative_structure,
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

        self.assertIn("Generative Structure Engine:", system)
        self.assertIn("Generative architecture:", system)
        self.assertTrue(
            any(
                "Generative blueprint:" in item
                for item in stack.director.planning_focus
            ),
            stack.director.planning_focus,
        )
        self.assertIn(
            "generative_structure",
            {item.source for item in stack.reasoning.evidence_chain},
        )
        self.assertIn(
            "Blueprint as",
            stack.reasoning.recommended_creative_direction,
        )
        self.assertEqual(
            stack.generative_structure.model_dump(mode="json")["role"],
            "generative_structure_engine",
        )
        self.assertNotIn("runtime auto-selection enabled", system)
        self.assertNotIn("provider routing", system.lower())


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
    generative_structure: object
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
    generative_structure = derive_generative_structure_blueprint(
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
        procedural_structure=procedural_structure,
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
        generative_structure=generative_structure,
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
        generative_structure=generative_structure,
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
        generative_structure=generative_structure,
        director=director,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    unittest.main()
