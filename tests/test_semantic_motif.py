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
    derive_semantic_motif_system,
    derive_symbolic_narrative_plan,
    semantic_motif_prompt_lines,
)


class SemanticMotifEngineTests(unittest.TestCase):
    def test_derives_motif_hierarchy_roles_and_recurrence(self) -> None:
        stack = _stack(
            "Generate a p5.js recursive spiral phoenix mandala with particles "
            "dissolving into embers and reassembling through a radial rebirth."
        )
        motif = stack.semantic_motif
        primary_ids = {item.motif_id for item in motif.primary_motifs}
        secondary_ids = {item.motif_id for item in motif.secondary_motifs}
        roles = {item.role for item in (*motif.primary_motifs, *motif.secondary_motifs)}

        self.assertEqual(motif.role, "semantic_motif_engine")
        self.assertIn("fragmentation", primary_ids)
        self.assertIn("reintegration", primary_ids)
        self.assertTrue({"spiral", "mandala", "flame"}.intersection(secondary_ids))
        self.assertIn("transformation", roles)
        self.assertTrue(motif.motif_hierarchy)
        self.assertTrue(motif.motif_recurrence_plan)
        self.assertTrue(motif.motif_transformation_plan)
        self.assertTrue(semantic_motif_prompt_lines(motif))

    def test_maps_motifs_to_structure_composition_narrative_and_parameters(
        self,
    ) -> None:
        stack = _stack(
            "Generate a p5.js phoenix that dissolves into ember particles, "
            "flows through turbulent noise, then reforms from a storm of sparks."
        )
        motif = stack.semantic_motif
        structure_by_id = {
            item.motif_id: item for item in motif.motif_to_structure_mapping
        }
        parameter_by_id = {
            item.motif_id: item for item in motif.motif_to_parameter_mapping
        }
        narrative_by_id = {
            item.motif_id: item for item in motif.motif_to_narrative_mapping
        }

        self.assertIn("fragmentation", structure_by_id)
        self.assertIn(
            "particle_emitter",
            set(structure_by_id["fragmentation"].generative_module_kinds),
        )
        self.assertTrue(
            {
                "fragmentation_amount",
                "particle_count",
                "max_particle_count",
            }.intersection(parameter_by_id["fragmentation"].parameter_names),
            parameter_by_id["fragmentation"].parameter_names,
        )
        self.assertIn("reintegration", parameter_by_id)
        self.assertIn("reassembly_speed", parameter_by_id["reintegration"].parameter_names)
        self.assertIn("reintegration", narrative_by_id)
        self.assertTrue(narrative_by_id["reintegration"].phase_alignment)
        self.assertTrue(motif.motif_to_composition_mapping)
        self.assertTrue(motif.motif_fallback_plan.prompt_guidance)

    def test_flags_unsupported_symbolic_claims_and_hitl(self) -> None:
        stack = _stack(
            "Create a sacred chakra eye mandala that proves ancient cosmic "
            "truth, but maybe keep the symbolism subtle."
        )
        motif = stack.semantic_motif
        motif_ids = {
            item.motif_id for item in (*motif.primary_motifs, *motif.secondary_motifs)
        }

        self.assertTrue({"eye", "mandala"}.intersection(motif_ids))
        self.assertTrue(motif.unsupported_symbolic_claims)
        self.assertTrue(
            any("doctrine" in item.lower() for item in motif.unsupported_symbolic_claims),
            motif.unsupported_symbolic_claims,
        )
        self.assertTrue(motif.overuse_risks)
        self.assertTrue(motif.unresolved_motif_gaps)
        self.assertTrue(
            any("user-authored" in item for item in motif.hitl_questions),
            motif.hitl_questions,
        )

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
                "semantic_motif": stack.semantic_motif,
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

        self.assertIn("Semantic Motif Engine:", system)
        self.assertIn("Semantic motif system:", system)
        self.assertTrue(
            any("Semantic motifs:" in item for item in stack.director.planning_focus),
            stack.director.planning_focus,
        )
        self.assertIn(
            "semantic_motif",
            {item.source for item in stack.reasoning.evidence_chain},
        )
        self.assertIn("Motifs as", stack.reasoning.recommended_creative_direction)
        self.assertEqual(
            stack.semantic_motif.model_dump(mode="json")["role"],
            "semantic_motif_engine",
        )
        self.assertNotIn("runtime auto-selection enabled", system)
        self.assertNotIn("provider routing enabled", system.lower())


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
    semantic_motif: object
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
    semantic_motif = derive_semantic_motif_system(
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
        creative_tradeoffs=tradeoffs,
        creative_quality_prediction=quality_prediction,
        symbolic_narrative=symbolic_narrative,
        creative_composition=creative_composition,
        procedural_structure=procedural_structure,
        generative_structure=generative_structure,
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
        semantic_motif=semantic_motif,
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
        semantic_motif=semantic_motif,
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
        semantic_motif=semantic_motif,
        director=director,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    unittest.main()
