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
    cross_modality_prompt_lines,
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
    derive_cross_modality_composition_profile,
    derive_emotional_consistency_profile,
    derive_generative_structure_blueprint,
    derive_procedural_structure_plan,
    derive_runtime_capability_profile,
    derive_semantic_motif_system,
    derive_symbolic_narrative_plan,
)


class CrossModalityComposerTests(unittest.TestCase):
    def test_derives_hierarchy_roles_and_synchronization(self) -> None:
        stack = _stack(
            "Generate an audiovisual p5.js phoenix mandala with particles "
            "dissolving into embers, pulse-synced motion, threshold stillness, "
            "and luminous reintegration."
        )
        profile = stack.cross_modality

        self.assertEqual(profile.role, "cross_modality_composer")
        self.assertEqual(profile.primary_modality, "visual_structure")
        self.assertIn("audio", profile.supporting_modalities)
        self.assertIn("motion", profile.supporting_modalities)
        self.assertGreaterEqual(len(profile.modality_hierarchy), 3)
        self.assertTrue(profile.visual_role)
        self.assertTrue(profile.motion_role)
        self.assertTrue(profile.audio_role)
        self.assertTrue(profile.rhythm_role)
        self.assertTrue(profile.structure_role)
        self.assertTrue(profile.motif_role)
        self.assertTrue(profile.emotion_role)
        self.assertTrue(profile.modality_synchronization_plan)
        self.assertTrue(cross_modality_prompt_lines(profile))

    def test_maps_audio_motion_structure_motif_and_emotion(self) -> None:
        stack = _stack(
            "Create an audio-reactive phoenix that fragments into particle "
            "embers, follows a low drone pulse, then reforms into a luminous "
            "spiral with reintegration."
        )
        profile = stack.cross_modality

        self.assertTrue(profile.visual_to_audio_mapping)
        self.assertTrue(profile.audio_to_motion_mapping)
        self.assertTrue(profile.motion_to_structure_mapping)
        self.assertTrue(profile.motif_to_modality_mapping)
        self.assertTrue(profile.emotional_to_modality_mapping)
        self.assertTrue(
            any(
                "pulse" in cue
                for mapping in profile.audio_to_motion_mapping
                for cue in mapping.cues
            ),
            profile.audio_to_motion_mapping,
        )
        self.assertTrue(
            {"fragmentation", "reintegration"}.intersection(
                {
                    mapping.motif_id
                    for mapping in profile.motif_to_modality_mapping
                    if mapping.motif_id is not None
                }
            ),
            profile.motif_to_modality_mapping,
        )
        self.assertIn(
            profile.emotional_to_modality_mapping[0].emotional_tone,
            {"transformation", "rupture", "integration", "dissolution"},
        )
        self.assertTrue(profile.fallback_multimodal_strategy.prompt_guidance)

    def test_flags_conflicts_overload_underuse_and_hitl(self) -> None:
        stack = _stack(
            "Maybe make an audiovisual vibe with dense particle visuals, loud "
            "audio, camera motion, and profound ritual symbolism."
        )
        profile = stack.cross_modality

        self.assertTrue(profile.modality_conflicts, profile)
        self.assertTrue(profile.overload_risks, profile)
        self.assertTrue(profile.underuse_risks, profile)
        self.assertTrue(profile.unresolved_modality_gaps, profile)
        self.assertTrue(profile.hitl_questions, profile)
        self.assertTrue(
            any("modality should lead" in item for item in profile.hitl_questions),
            profile.hitl_questions,
        )

    def test_integrates_with_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack(
            "Generate a symbolic recursive spiral threshold in p5.js with "
            "sacred geometry, orbiting rings, solemn emotional pacing, a "
            "visible release phase, and subtle pulse-driven audio cues."
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
                "emotional_consistency": stack.emotional_consistency,
                "cross_modality": stack.cross_modality,
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

        self.assertIn("Cross-Modality Composer:", system)
        self.assertIn("Primary modality:", system)
        self.assertTrue(
            any("Cross-modality:" in item for item in stack.director.planning_focus),
            stack.director.planning_focus,
        )
        self.assertIn(
            "cross_modality",
            {item.source for item in stack.reasoning.evidence_chain},
        )
        self.assertIn("Modalities as", stack.reasoning.recommended_creative_direction)
        self.assertEqual(
            stack.cross_modality.model_dump(mode="json")["role"],
            "cross_modality_composer",
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
    emotional_consistency: object
    cross_modality: object
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
        retrieval_chunk_count=0,
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
        clarification=None,
        retrieval_chunk_count=0,
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
    emotional_consistency = derive_emotional_consistency_profile(
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
        generative_structure=generative_structure,
        semantic_motif=semantic_motif,
    )
    cross_modality = derive_cross_modality_composition_profile(
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
        generative_structure=generative_structure,
        semantic_motif=semantic_motif,
        emotional_consistency=emotional_consistency,
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
        emotional_consistency=emotional_consistency,
        cross_modality=cross_modality,
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
        emotional_consistency=emotional_consistency,
        cross_modality=cross_modality,
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
        emotional_consistency=emotional_consistency,
        cross_modality=cross_modality,
        director=director,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    unittest.main()
