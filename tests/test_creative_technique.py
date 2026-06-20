import unittest

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
    creative_technique_prompt_lines,
    derive_creative_assistant_director_brief,
    derive_creative_constraint_solution,
    derive_creative_execution_plan,
    derive_creative_strategy_profile,
    derive_creative_technique_profile,
)


class CreativeTechniqueSelectorTests(unittest.TestCase):
    def test_selector_chooses_particle_systems_for_particle_strategy(self) -> None:
        request = AssistantRequest(
            query="Generate a particle nebula with orbiting star clusters.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.THREE_JS,
        )
        route = _route(CreativeCodingDomain.THREE_JS)
        prompt_input = _prompt_input(request, route)
        strategy = derive_creative_strategy_profile(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
        )

        techniques = derive_creative_technique_profile(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_strategy=strategy,
        )

        self.assertEqual(techniques.role, "creative_technique_selector")
        self.assertEqual(techniques.primary_technique, "particle_systems")
        self.assertEqual(techniques.strategy_alignment, "particle_cosmology")
        self.assertEqual(techniques.compatibility, "strong")
        self.assertIn("does not choose runtime", techniques.selection_boundary)
        self.assertTrue(creative_technique_prompt_lines(techniques))

    def test_selector_keeps_audio_reactive_mapping_bounded(self) -> None:
        request = AssistantRequest(
            query="Create an audio-reactive field driven by beat and drone energy.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route = _route(CreativeCodingDomain.P5_JS)
        prompt_input = _prompt_input(request, route)
        strategy = derive_creative_strategy_profile(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
        )

        techniques = derive_creative_technique_profile(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_strategy=strategy,
        )

        self.assertEqual(techniques.primary_technique, "audio_reactive_mappings")
        self.assertTrue(
            any("audio input" in item for item in techniques.implementation_notes),
            techniques.implementation_notes,
        )
        self.assertTrue(
            any("runtime" in item for item in techniques.technique_constraints),
            techniques.technique_constraints,
        )

    def test_techniques_integrate_with_plan_constraints_and_director(self) -> None:
        request = AssistantRequest(
            query="Generate a quiet field of drifting magnetic waves.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route = _route(CreativeCodingDomain.P5_JS)
        prompt_input = _prompt_input(request, route)
        strategy = derive_creative_strategy_profile(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
        )
        techniques = derive_creative_technique_profile(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_strategy=strategy,
        )
        plan = derive_creative_execution_plan(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_strategy=strategy,
            creative_techniques=techniques,
        )
        constraints = derive_creative_constraint_solution(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_plan=plan,
            creative_strategy=strategy,
            creative_techniques=techniques,
        )
        director = derive_creative_assistant_director_brief(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_plan=plan,
            creative_strategy=strategy,
            creative_techniques=techniques,
            creative_constraints=constraints,
        )

        self.assertIn(
            "Do not treat creative technique as runtime or renderer selection.",
            plan.constraints,
        )
        self.assertIn(
            f"Use {techniques.primary_technique} as bounded technique guidance.",
            constraints.prompt_guidance,
        )
        self.assertIn(
            f"Primary technique: {techniques.primary_technique}.",
            director.planning_focus,
        )

    def test_prompt_renderer_includes_technique_guidance(self) -> None:
        request = AssistantRequest(
            query="Generate recursive geometry for a luminous mandala.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route = _route(CreativeCodingDomain.P5_JS)
        prompt_input = _prompt_input(request, route)
        strategy = derive_creative_strategy_profile(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
        )
        techniques = derive_creative_technique_profile(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_strategy=strategy,
        )
        prompt_input = prompt_input.model_copy(
            update={
                "creative_strategy": strategy,
                "creative_techniques": techniques,
            }
        )

        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=route,
                prompt_input=prompt_input,
            )
        )

        system = rendered.sections[0].content
        self.assertIn("Creative Technique Selector:", system)
        self.assertIn("Primary technique:", system)
        self.assertIn("Technique constraint:", system)
        self.assertNotIn("Recommended runtime:", system)


def _route(domain: CreativeCodingDomain) -> RouteDecision:
    return RouteDecision(
        route=RouteName.GENERATE,
        mode=AssistantMode.GENERATE,
        domain=domain,
        domains=(domain,),
        capabilities=(RouteCapability.TOOL_USE,),
    )


def _prompt_input(request: AssistantRequest, route: RouteDecision):
    return StructuredPromptInputBuilder().build(
        build_prompt_input_request(
            assistant_request=request,
            route_decision=route,
            assembled_context=None,
        )
    )


if __name__ == "__main__":
    unittest.main()
