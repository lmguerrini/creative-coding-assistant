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
    creative_reasoning_prompt_lines,
    derive_creative_assistant_director_brief,
    derive_creative_constraint_solution,
    derive_creative_execution_plan,
    derive_creative_reasoning_result,
    derive_creative_strategy_profile,
    derive_creative_technique_profile,
    derive_creative_tradeoff_profile,
    derive_runtime_capability_profile,
)


class CreativeReasoningEngineTests(unittest.TestCase):
    def test_reasoning_synthesizes_direction_across_existing_signals(self) -> None:
        request = AssistantRequest(
            query="Generate a recursive p5.js mandala.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route = _route(CreativeCodingDomain.P5_JS)
        context = _creative_context(request, route)

        reasoning = derive_creative_reasoning_result(**context)

        self.assertEqual(reasoning.role, "creative_reasoning_engine")
        self.assertIn(
            context["creative_strategy"].primary_strategy,
            reasoning.recommended_creative_direction,
        )
        self.assertIn(
            context["creative_techniques"].primary_technique,
            reasoning.recommended_creative_direction,
        )
        self.assertEqual(
            [step.stage for step in reasoning.reasoning_path],
            ["strategy", "technique", "runtime", "tradeoff", "recommendation"],
        )
        self.assertEqual(
            reasoning.reasoning_path[-1].claim,
            reasoning.recommended_creative_direction,
        )
        self.assertTrue(
            {
                "creative_strategy",
                "creative_technique",
                "runtime_capability",
                "tradeoff_explorer",
            }.issubset({item.source for item in reasoning.evidence_chain})
        )
        self.assertTrue(reasoning.strongest_supporting_signals)
        self.assertTrue(reasoning.implementation_guidance)
        self.assertTrue(reasoning.rejected_alternatives)
        self.assertEqual(
            reasoning.future_knowledge_context["status"],
            "not_attached",
        )
        self.assertIn("does not generate code", reasoning.authority_boundary)
        self.assertTrue(creative_reasoning_prompt_lines(reasoning))

    def test_reasoning_does_not_treat_bounded_tone_preview_as_unavailable(self) -> None:
        request = AssistantRequest(
            query="Generate a Tone.js rhythm pulse.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.TONE_JS,
        )
        route = _route(CreativeCodingDomain.TONE_JS)
        reasoning = derive_creative_reasoning_result(
            **_creative_context(request, route)
        )

        self.assertTrue(reasoning.hitl_questions)
        self.assertFalse(
            any("preview runtime" in item for item in reasoning.unresolved_decisions),
            reasoning.unresolved_decisions,
        )
        self.assertTrue(
            any("HITL" in item for item in reasoning.prompt_guidance),
            reasoning.prompt_guidance,
        )
        self.assertNotIn("No blocking", " ".join(reasoning.unresolved_decisions))

    def test_reasoning_integrates_with_prompt_rendering(self) -> None:
        request = AssistantRequest(
            query="Generate a luminous particle field in p5.js.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route = _route(CreativeCodingDomain.P5_JS)
        context = _creative_context(request, route)
        reasoning = derive_creative_reasoning_result(**context)
        prompt_input = _prompt_input(request, route).model_copy(
            update={
                "creative_strategy": context["creative_strategy"],
                "creative_techniques": context["creative_techniques"],
                "creative_plan": context["creative_plan"],
                "creative_constraints": context["creative_constraints"],
                "runtime_capabilities": context["runtime_capabilities"],
                "creative_tradeoffs": context["creative_tradeoffs"],
                "creative_director": context["creative_director"],
                "creative_reasoning": reasoning,
            }
        )

        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=route,
                prompt_input=prompt_input,
            )
        )
        system = rendered.sections[0].content

        self.assertIn("Creative Reasoning Engine:", system)
        self.assertIn("Reasoning recommendation:", system)
        self.assertIn(
            "strategy -> technique -> runtime -> trade-off -> recommendation",
            system,
        )
        self.assertNotIn("runtime auto-selection enabled", system)

    def test_reasoning_clips_long_demo_prompt_evidence(self) -> None:
        request = AssistantRequest(
            query=(
                "Create an audio-reactive Three.js visual system for a capstone "
                "demo. Use concentric geometry, subtle bloom, FFT-driven motion "
                "accents, camera movement, browser-safe runtime notes, "
                "interaction guidance, and a clear fallback if live audio or "
                "preview is unavailable."
            ),
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.THREE_JS,
        )
        route = _route(CreativeCodingDomain.THREE_JS)

        reasoning = derive_creative_reasoning_result(
            **_creative_context(request, route)
        )

        self.assertIn(
            "translation",
            {item.source for item in reasoning.evidence_chain},
        )
        self.assertTrue(
            all(len(item.signal) <= 240 for item in reasoning.evidence_chain)
        )
        self.assertTrue(
            all(len(item.interpretation) <= 360 for item in reasoning.evidence_chain)
        )


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
    director = derive_creative_assistant_director_brief(
        request=request,
        route_decision=route,
        creative_translation=prompt_input.creative_translation,
        creative_strategy=strategy,
        creative_techniques=techniques,
        creative_plan=plan,
        creative_constraints=constraints,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=tradeoffs,
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
        "creative_tradeoffs": tradeoffs,
        "creative_director": director,
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
