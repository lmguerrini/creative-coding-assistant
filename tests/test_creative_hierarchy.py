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
    creative_hierarchy_plan_prompt_lines,
    derive_creative_hierarchy_plan,
    derive_creative_intent_decomposition,
)


class CreativeHierarchyPlannerTests(unittest.TestCase):
    def test_ranks_priorities_beyond_decomposed_intent_order(self) -> None:
        request = AssistantRequest(
            query=(
                "Create a simple browser-safe ritual mandala where symbolic "
                "geometry must dominate over motion. Keep a calm gold glow and "
                "a slow pulse, but make performance flexible."
            ),
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route = _route(CreativeCodingDomain.P5_JS)
        prompt_input = _prompt_input(request, route)
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

        primary_dimensions = [
            item.dimension for item in hierarchy.primary_creative_priorities
        ]
        decomposed_order = [
            item.name for item in intent.atomic_dimensions if item.signals
        ]

        self.assertEqual(hierarchy.role, "creative_hierarchy_planner")
        self.assertNotEqual(
            primary_dimensions,
            decomposed_order[: len(primary_dimensions)],
        )
        self.assertIn("symbolism", primary_dimensions)
        self.assertIn("geometry", primary_dimensions)
        self.assertTrue(hierarchy.non_negotiable_dimensions)
        self.assertIn("audio", hierarchy.flexible_dimensions)
        self.assertTrue(hierarchy.priority_rationale)
        self.assertGreater(hierarchy.hierarchy_confidence, 0.5)
        self.assertTrue(creative_hierarchy_plan_prompt_lines(hierarchy))
        self.assertIn("does not select runtimes", hierarchy.authority_boundary)

    def test_detects_conflicts_and_hitl_questions(self) -> None:
        request = AssistantRequest(
            query=(
                "Generate a cinematic visual spectacle with dense complex "
                "layers, but it must stay 60 fps on mobile and browser-safe."
            ),
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route = _route(CreativeCodingDomain.P5_JS)
        prompt_input = _prompt_input(request, route)
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

        self.assertIn(
            "Visual impact may compete with performance priority.",
            hierarchy.priority_conflicts,
        )
        self.assertTrue(hierarchy.hitl_questions)
        self.assertTrue(
            any("visual" in item.lower() for item in hierarchy.hitl_questions),
            hierarchy.hitl_questions,
        )

    def test_prompt_rendering_includes_hierarchy_guidance(self) -> None:
        request = AssistantRequest(
            query="Generate a glowing abstract spiral where motion should lead.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route = _route(CreativeCodingDomain.P5_JS)
        prompt_input = _prompt_input(request, route)
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
        prompt_input = prompt_input.model_copy(
            update={
                "creative_intent": intent,
                "creative_hierarchy": hierarchy,
            }
        )

        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=route,
                prompt_input=prompt_input,
            )
        )
        system = rendered.sections[0].content

        self.assertIn("Creative Hierarchy Planner:", system)
        self.assertIn("Primary priority", system)
        self.assertIn("Hierarchy guidance:", system)
        self.assertNotIn("runtime auto-selection", system)


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
