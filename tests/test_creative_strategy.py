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
    creative_strategy_prompt_lines,
    derive_creative_assistant_director_brief,
    derive_creative_constraint_solution,
    derive_creative_execution_plan,
    derive_creative_strategy_profile,
)


class CreativeStrategyEngineTests(unittest.TestCase):
    def test_strategy_selects_sacred_geometry_without_runtime_choice(self) -> None:
        request = AssistantRequest(
            query="Create a luminous mandala using sacred geometry and golden ratio.",
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

        self.assertEqual(strategy.role, "creative_strategy_engine")
        self.assertEqual(strategy.primary_strategy, "sacred_geometry")
        self.assertGreater(strategy.confidence, 0.5)
        self.assertIn("Sacred Geometry", strategy.symbolic_alignment)
        self.assertIn("does not choose runtimes", strategy.implementation_boundary)
        self.assertTrue(creative_strategy_prompt_lines(strategy))

    def test_strategy_engine_prefers_particle_cosmology_for_particle_brief(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Generate a particle nebula with orbiting star clusters.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.THREE_JS,
        )

        strategy = derive_creative_strategy_profile(
            request=request,
            route_decision=_route(CreativeCodingDomain.THREE_JS),
            creative_translation=None,
        )

        self.assertEqual(strategy.primary_strategy, "particle_cosmology")
        self.assertTrue(strategy.alternative_strategies)
        self.assertTrue(
            any("particle" in item.lower() for item in strategy.evidence),
            strategy.evidence,
        )

    def test_strategy_integrates_with_plan_constraints_and_director(self) -> None:
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
        plan = derive_creative_execution_plan(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_strategy=strategy,
        )
        constraints = derive_creative_constraint_solution(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_plan=plan,
            creative_strategy=strategy,
        )
        director = derive_creative_assistant_director_brief(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_plan=plan,
            creative_strategy=strategy,
            creative_constraints=constraints,
        )

        self.assertIn(
            "Do not treat creative strategy as runtime or implementation technique.",
            plan.constraints,
        )
        self.assertIn(
            f"Preserve {strategy.primary_strategy} as high-level strategy.",
            constraints.prompt_guidance,
        )
        self.assertIn(
            f"High-level strategy: {strategy.primary_strategy}.",
            director.planning_focus,
        )

    def test_prompt_renderer_includes_strategy_guidance(self) -> None:
        request = AssistantRequest(
            query="Generate a minimal generative system of quiet monochrome lines.",
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
        prompt_input = prompt_input.model_copy(update={"creative_strategy": strategy})

        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=route,
                prompt_input=prompt_input,
            )
        )

        system = rendered.sections[0].content
        self.assertIn("Creative Strategy Engine:", system)
        self.assertIn("Primary strategy: minimal_generative_systems.", system)
        self.assertIn("Authority boundary:", system)
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
