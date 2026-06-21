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
    derive_creative_assistant_director_brief,
    derive_creative_constraint_solution,
    derive_creative_execution_plan,
    derive_creative_strategy_profile,
    derive_creative_technique_profile,
    derive_runtime_capability_profile,
    runtime_capability_prompt_lines,
)


class RuntimeCapabilityReasonerTests(unittest.TestCase):
    def test_reasoner_scores_p5_without_runtime_auto_selection(self) -> None:
        request = AssistantRequest(
            query="Generate a p5.js particle field with drifting luminous trails.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route = _route(CreativeCodingDomain.P5_JS)
        context = _creative_context(request, route)

        profile = derive_runtime_capability_profile(**context)

        self.assertEqual(profile.role, "runtime_capability_reasoner")
        self.assertEqual(profile.likely_candidates[0], "p5_js")
        self.assertEqual(len(profile.candidate_runtimes), 9)
        self.assertIn("does not auto-select runtimes", profile.authority_boundary)
        p5_candidate = _candidate(profile, "p5_js")
        self.assertEqual(p5_candidate.suitability, "strong")
        self.assertEqual(p5_candidate.preview_support, "backend_preview_supported")
        self.assertTrue(runtime_capability_prompt_lines(profile))
        self.assertNotIn(
            "Recommended runtime",
            "\n".join(runtime_capability_prompt_lines(profile)),
        )

    def test_reasoner_surfaces_tone_js_hitl_advisory_for_audio_scope(self) -> None:
        request = AssistantRequest(
            query="Generate a Tone.js rhythm pulse.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.TONE_JS,
        )
        route = _route(CreativeCodingDomain.TONE_JS)
        context = _creative_context(request, route)

        profile = derive_runtime_capability_profile(**context)

        self.assertEqual(profile.likely_candidates[0], "tone_js")
        self.assertTrue(profile.hitl_advisable)
        self.assertIn("live preview runtime", profile.hitl_reason or "")
        tone_candidate = _candidate(profile, "tone_js")
        self.assertEqual(tone_candidate.output_goal_fit, "strong")
        self.assertEqual(tone_candidate.preview_support, "workstation_preview_bounded")
        self.assertTrue(
            any("Audio context" in item for item in tone_candidate.risks),
            tone_candidate.risks,
        )

    def test_reasoner_integrates_with_director_and_prompt_guidance(self) -> None:
        request = AssistantRequest(
            query="Generate recursive geometry for a luminous mandala.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route = _route(CreativeCodingDomain.P5_JS)
        context = _creative_context(request, route)
        profile = derive_runtime_capability_profile(**context)
        director = derive_creative_assistant_director_brief(
            request=request,
            route_decision=route,
            creative_translation=context["creative_translation"],
            creative_strategy=context["creative_strategy"],
            creative_techniques=context["creative_techniques"],
            creative_plan=context["creative_plan"],
            creative_constraints=context["creative_constraints"],
            runtime_capabilities=profile,
        )
        prompt_input = _prompt_input(request, route).model_copy(
            update={
                "creative_strategy": context["creative_strategy"],
                "creative_techniques": context["creative_techniques"],
                "creative_plan": context["creative_plan"],
                "creative_constraints": context["creative_constraints"],
                "runtime_capabilities": profile,
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

        self.assertIn("Runtime Capability Reasoner:", system)
        self.assertIn("Likely runtime candidates (non-binding):", system)
        self.assertTrue(
            any(
                "Runtime capability candidates:" in item
                for item in director.planning_focus
            ),
            director.planning_focus,
        )
        self.assertNotIn("Runtime Debug", system)


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
    return {
        "request": request,
        "route_decision": route,
        "creative_translation": prompt_input.creative_translation,
        "creative_strategy": strategy,
        "creative_techniques": techniques,
        "creative_plan": plan,
        "creative_constraints": constraints,
    }


def _candidate(profile, runtime):
    return next(
        candidate
        for candidate in profile.candidate_runtimes
        if candidate.runtime == runtime
    )


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
