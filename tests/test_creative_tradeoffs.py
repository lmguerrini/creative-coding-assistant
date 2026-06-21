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
    creative_tradeoff_prompt_lines,
    derive_creative_assistant_director_brief,
    derive_creative_constraint_solution,
    derive_creative_execution_plan,
    derive_creative_strategy_profile,
    derive_creative_technique_profile,
    derive_creative_tradeoff_profile,
    derive_runtime_capability_profile,
)


class CreativeTradeoffExplorerTests(unittest.TestCase):
    def test_explorer_structures_expressiveness_complexity_tradeoff(self) -> None:
        request = AssistantRequest(
            query="Generate a p5.js particle nebula with dense luminous trails.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route = _route(CreativeCodingDomain.P5_JS)
        context = _creative_context(request, route)

        profile = derive_creative_tradeoff_profile(**context)

        self.assertEqual(profile.role, "creative_tradeoff_explorer")
        self.assertTrue(profile.primary_tradeoffs)
        self.assertIn(
            "does not select final artifacts",
            profile.authority_boundary,
        )
        self.assertTrue(
            any(
                item.source_axis == "creative_expressiveness"
                and item.target_axis == "implementation_complexity"
                for item in profile.primary_tradeoffs
            ),
            profile.primary_tradeoffs,
        )
        self.assertTrue(profile.creative_benefits)
        self.assertTrue(profile.technical_costs)
        self.assertTrue(creative_tradeoff_prompt_lines(profile))

    def test_explorer_surfaces_hitl_for_runtime_ambiguity(self) -> None:
        request = AssistantRequest(
            query="Generate a Tone.js rhythm pulse.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.TONE_JS,
        )
        route = _route(CreativeCodingDomain.TONE_JS)
        context = _creative_context(request, route)

        profile = derive_creative_tradeoff_profile(**context)

        self.assertTrue(profile.hitl_advisable)
        self.assertIn("live preview runtime", profile.hitl_reason or "")
        self.assertTrue(
            any(item.hitl_recommended for item in profile.primary_tradeoffs),
            profile.primary_tradeoffs,
        )
        self.assertTrue(profile.director_discussion_points)

    def test_explorer_integrates_with_director_and_prompt_guidance(self) -> None:
        request = AssistantRequest(
            query="Generate recursive geometry for a luminous mandala.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route = _route(CreativeCodingDomain.P5_JS)
        context = _creative_context(request, route)
        tradeoffs = derive_creative_tradeoff_profile(**context)
        director = derive_creative_assistant_director_brief(
            request=request,
            route_decision=route,
            creative_translation=context["creative_translation"],
            creative_strategy=context["creative_strategy"],
            creative_techniques=context["creative_techniques"],
            creative_plan=context["creative_plan"],
            creative_constraints=context["creative_constraints"],
            runtime_capabilities=context["runtime_capabilities"],
            creative_tradeoffs=tradeoffs,
        )
        prompt_input = _prompt_input(request, route).model_copy(
            update={
                "creative_strategy": context["creative_strategy"],
                "creative_techniques": context["creative_techniques"],
                "creative_plan": context["creative_plan"],
                "creative_constraints": context["creative_constraints"],
                "runtime_capabilities": context["runtime_capabilities"],
                "creative_tradeoffs": tradeoffs,
                "creative_director": director,
            }
        )

        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=route,
                prompt_input=prompt_input,
            )
        )
        system = rendered.sections[0].content

        self.assertIn("Creative Trade-off Explorer:", system)
        self.assertIn("Primary trade-off:", system)
        self.assertTrue(
            any(
                "Trade-off discussion:" in item
                for item in director.planning_focus
            ),
            director.planning_focus,
        )
        self.assertNotIn("Artifact selection", system)


def _creative_context(
    request: AssistantRequest,
    route: RouteDecision,
) -> dict[str, object]:
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
        creative_strategy=strategy,
        creative_techniques=techniques,
        creative_plan=plan,
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
    return {
        "request": request,
        "route_decision": route,
        "creative_translation": prompt_input.creative_translation,
        "creative_strategy": strategy,
        "creative_techniques": techniques,
        "creative_plan": plan,
        "creative_constraints": constraints,
        "runtime_capabilities": runtime_capabilities,
    }


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
