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
    audio_visual_scene_prompt_lines,
    build_prompt_input_request,
    build_rendered_prompt_request,
    derive_audio_visual_scene_profile,
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


class AudioVisualSceneSystemTests(unittest.TestCase):
    def test_derives_scene_arc_phases_cues_and_prompt_guidance(self) -> None:
        stack = _stack(
            "Generate an audiovisual p5.js phoenix mandala with particles "
            "dissolving into embers, pulse-synced motion, threshold stillness, "
            "and luminous reintegration."
        )
        scene = stack.audio_visual_scene

        self.assertEqual(scene.role, "audio_visual_scene_system")
        self.assertIn(
            scene.scene_pattern,
            {
                "fragmentation_to_reintegration",
                "ritual_opening_to_climax",
                "pulse_escalation",
                "seed_to_expansion",
            },
        )
        self.assertEqual([phase.phase for phase in scene.scene_phases], _PHASES)
        self.assertEqual(scene.opening_scene.phase, "opening")
        self.assertEqual(scene.development_scene.phase, "development")
        self.assertEqual(scene.threshold_scene.phase, "threshold")
        self.assertEqual(scene.climax_scene.phase, "climax")
        self.assertEqual(scene.resolution_scene.phase, "resolution")
        self.assertGreaterEqual(len(scene.cue_plan), 5)
        self.assertEqual(len(scene.transition_plan), 4)
        self.assertTrue(scene.climax_strategy)
        self.assertTrue(scene.resolution_strategy)
        self.assertTrue(audio_visual_scene_prompt_lines(scene))

    def test_maps_timing_plans_and_synchronization(self) -> None:
        stack = _stack(
            "Create an audio-reactive phoenix with camera orbit, low drone "
            "pulse, particle embers, threshold silence, and luminous "
            "reintegration."
        )
        scene = stack.audio_visual_scene

        self.assertTrue(scene.visual_timing_plan)
        self.assertTrue(scene.motion_timing_plan)
        self.assertTrue(scene.audio_timing_plan)
        self.assertTrue(scene.rhythm_timing_plan)
        self.assertTrue(scene.camera_timing_plan)
        self.assertTrue(scene.motif_timing_plan)
        self.assertTrue(scene.emotional_timing_plan)
        self.assertTrue(scene.procedural_timing_plan)
        self.assertTrue(scene.synchronization_checkpoints)
        self.assertTrue(
            any(
                {"drone", "pulse", "silence", "cadence"}.intersection(
                    set(item.lower().split())
                )
                for item in scene.audio_timing_plan
            ),
            scene.audio_timing_plan,
        )
        self.assertTrue(
            any(cue.cue_type == "synchronization" for cue in scene.cue_plan),
            scene.cue_plan,
        )

    def test_flags_pacing_overload_fallback_and_hitl(self) -> None:
        stack = _stack(
            "Maybe make an audiovisual scene vibe with dense chaotic particle "
            "visuals, loud intense audio, cinematic camera motion, many ritual "
            "symbols, and overwhelming strobe-like climax timing."
        )
        scene = stack.audio_visual_scene

        self.assertTrue(scene.scene_risks, scene)
        self.assertTrue(scene.pacing_risks, scene)
        self.assertTrue(scene.overload_risks, scene)
        self.assertTrue(scene.unresolved_scene_gaps, scene)
        self.assertTrue(scene.hitl_questions, scene)
        self.assertTrue(scene.fallback_scene_strategy.reduced_elements)
        self.assertIn(
            "audio timing",
            scene.fallback_scene_strategy.reduced_elements,
        )
        self.assertIn(
            "camera/viewpoint timing",
            scene.fallback_scene_strategy.reduced_elements,
        )

    def test_integrates_with_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack(
            "Generate a symbolic recursive spiral threshold in p5.js with "
            "sacred geometry, orbiting rings, solemn emotional pacing, a "
            "visible release phase, camera threshold shift, and subtle "
            "pulse-driven audio cues."
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
                "audio_visual_scene": stack.audio_visual_scene,
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

        self.assertIn("Audio-Visual Scene System:", system)
        self.assertIn("Scene pattern:", system)
        self.assertTrue(
            any(
                "Audio-visual scene:" in item
                for item in stack.director.planning_focus
            ),
            stack.director.planning_focus,
        )
        self.assertIn(
            "audio_visual_scene",
            {item.source for item in stack.reasoning.evidence_chain},
        )
        self.assertIn("Scenes as", stack.reasoning.recommended_creative_direction)
        self.assertEqual(
            stack.audio_visual_scene.model_dump(mode="json")["role"],
            "audio_visual_scene_system",
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
    audio_visual_scene: object
    director: object
    reasoning: object


_PHASES = ["opening", "development", "threshold", "climax", "resolution"]


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
    audio_visual_scene = derive_audio_visual_scene_profile(
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
        cross_modality=cross_modality,
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
        audio_visual_scene=audio_visual_scene,
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
        audio_visual_scene=audio_visual_scene,
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
        audio_visual_scene=audio_visual_scene,
        director=director,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    unittest.main()
