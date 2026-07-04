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
    creative_intent_decomposition_prompt_lines,
    derive_creative_intent_decomposition,
)


class CreativeIntentDecomposerTests(unittest.TestCase):
    def test_decomposes_atomic_creative_dimensions(self) -> None:
        request = AssistantRequest(
            query=(
                "Create an audiovisual ritual: a symbolic lotus mandala that "
                "feels serene and full of awe, with radial geometry, pulsing "
                "spiral motion, polyrhythm drone beats, cyan and gold glowing "
                "particles, mouse interaction, and a bloom from darkness to light."
            ),
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS, CreativeCodingDomain.TONE_JS),
        )
        route = _route(CreativeCodingDomain.P5_JS, CreativeCodingDomain.TONE_JS)
        prompt_input = _prompt_input(request, route)

        decomposition = derive_creative_intent_decomposition(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
        )

        self.assertEqual(decomposition.role, "creative_intent_decomposer")
        self.assertNotEqual(decomposition.primary_expression, request.query)
        self.assertEqual(len(decomposition.atomic_dimensions), 10)
        self.assertEqual(decomposition.symbolic_intent.explicitness, "explicit")
        self.assertTrue(decomposition.symbolic_intent.signals)
        self.assertEqual(decomposition.emotional_intent.explicitness, "explicit")
        self.assertTrue({"calm", "awe"} & set(decomposition.emotional_intent.signals))
        self.assertEqual(decomposition.geometric_intent.explicitness, "explicit")
        self.assertTrue(decomposition.motion_intent.signals)
        self.assertEqual(decomposition.rhythm_intent.explicitness, "explicit")
        self.assertEqual(decomposition.audio_intent.explicitness, "explicit")
        self.assertEqual(decomposition.interaction_intent.explicitness, "explicit")
        self.assertEqual(
            decomposition.climax_transformation_intent.explicitness,
            "explicit",
        )
        self.assertIn("audio", decomposition.primary_expression)
        self.assertIn("does not choose strategy", decomposition.authority_boundary)
        self.assertTrue(creative_intent_decomposition_prompt_lines(decomposition))

    def test_surfaces_unresolved_gaps_and_hitl_questions(self) -> None:
        request = AssistantRequest(
            query="Make it beautiful.",
            mode=AssistantMode.GENERATE,
        )
        route = _route()
        prompt_input = _prompt_input(request, route)

        decomposition = derive_creative_intent_decomposition(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
        )

        self.assertIn(
            "Core creative subject or motif is underspecified.",
            decomposition.unresolved_intent_gaps,
        )
        self.assertIn(
            "Emotional tone is not explicit.",
            decomposition.unresolved_intent_gaps,
        )
        self.assertTrue(decomposition.hitl_questions)
        self.assertTrue(
            any("subject" in question for question in decomposition.hitl_questions)
        )

    def test_prompt_rendering_includes_decomposition_guidance(self) -> None:
        request = AssistantRequest(
            query="Generate a glowing abstract spiral that slowly morphs.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route = _route(CreativeCodingDomain.P5_JS)
        prompt_input = _prompt_input(request, route)
        decomposition = derive_creative_intent_decomposition(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
        )
        prompt_input = prompt_input.model_copy(
            update={"creative_intent": decomposition}
        )

        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=route,
                prompt_input=prompt_input,
            )
        )
        system = rendered.sections[0].content

        self.assertIn("Creative Intent Decomposer:", system)
        self.assertIn("Primary expression:", system)
        self.assertIn("Intent guidance:", system)
        self.assertNotIn("runtime auto-selection", system)


def _route(*domains: CreativeCodingDomain) -> RouteDecision:
    return RouteDecision(
        route=RouteName.GENERATE,
        mode=AssistantMode.GENERATE,
        domain=domains[0] if len(domains) == 1 else None,
        domains=domains,
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
