import unittest

from creative_coding_assistant.contracts import (
    AssistantImageReference,
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
    creative_constraint_solution_prompt_lines,
    derive_creative_assistant_director_brief,
    derive_creative_constraint_solution,
    derive_creative_execution_plan,
)


class CreativeConstraintSolverTests(unittest.TestCase):
    def test_solver_structures_supported_runtime_tradeoffs(self) -> None:
        request = AssistantRequest(
            query=(
                "Create several luminous p5.js mandala variations with "
                "audio-reactive pulses for a mobile 60 fps installation."
            ),
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
            attachments=(
                AssistantImageReference(
                    id="ref-1",
                    name="palette.png",
                    mimeType="image/png",
                    sizeBytes=128,
                    dataUrl="data:image/png;base64,AAAA",
                ),
            ),
        )
        route = _route(CreativeCodingDomain.P5_JS)
        prompt_input = _prompt_input(request, route)
        plan = derive_creative_execution_plan(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            retrieval_chunk_count=4,
        )

        solution = derive_creative_constraint_solution(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_plan=plan,
            retrieval_chunk_count=4,
        )

        self.assertEqual(solution.role, "creative_constraint_solver")
        self.assertEqual(solution.runtime_fit, "supported")
        self.assertEqual(solution.recommended_runtime, "p5")
        self.assertEqual(solution.complexity_pressure, "high")
        self.assertIn(solution.performance_pressure, {"medium", "high"})
        self.assertTrue(
            any(item.axis == "runtime" for item in solution.active_constraints)
        )
        self.assertTrue(
            any(
                item.source_axis == "complexity" and item.target_axis == "performance"
                for item in solution.tradeoffs
            ),
            solution.tradeoffs,
        )
        self.assertTrue(creative_constraint_solution_prompt_lines(solution))

    def test_solver_flags_code_only_runtime_and_hitl_advisory(self) -> None:
        request = AssistantRequest(
            query="Generate a Tone.js generative rhythm sketch.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.TONE_JS,
        )
        route = _route(CreativeCodingDomain.TONE_JS)
        prompt_input = _prompt_input(request, route)
        plan = derive_creative_execution_plan(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
        )

        solution = derive_creative_constraint_solution(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_plan=plan,
        )

        self.assertEqual(solution.runtime_fit, "code_only")
        self.assertTrue(solution.hitl_advisable)
        self.assertIn("live preview runtime", solution.hitl_reason or "")
        self.assertIn(
            "Requested scope does not map to current live preview support.",
            solution.conflicts,
        )
        self.assertTrue(
            any(
                item.source_axis == "intent" and item.target_axis == "runtime"
                for item in solution.tradeoffs
            ),
            solution.tradeoffs,
        )

    def test_prompt_renderer_includes_constraint_solver_guidance(self) -> None:
        request = AssistantRequest(
            query="Generate a luminous p5.js particle field.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route = _route(CreativeCodingDomain.P5_JS)
        prompt_input = _prompt_input(request, route)
        plan = derive_creative_execution_plan(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
        )
        solution = derive_creative_constraint_solution(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_plan=plan,
        )
        prompt_input = prompt_input.model_copy(
            update={
                "creative_plan": plan,
                "creative_constraints": solution,
            }
        )

        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=route,
                prompt_input=prompt_input,
            )
        )

        system = rendered.sections[0].content
        self.assertIn("Creative Constraint Solver:", system)
        self.assertIn("Runtime fit: supported.", system)
        self.assertIn("Constraint guidance:", system)
        self.assertNotIn("function setup", system)

    def test_director_consumes_constraint_solver_signals(self) -> None:
        request = AssistantRequest(
            query="Generate a Tone.js generative rhythm sketch.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.TONE_JS,
        )
        route = _route(CreativeCodingDomain.TONE_JS)
        prompt_input = _prompt_input(request, route)
        plan = derive_creative_execution_plan(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
        )
        solution = derive_creative_constraint_solution(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_plan=plan,
        )

        director = derive_creative_assistant_director_brief(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
            creative_plan=plan,
            creative_constraints=solution,
        )

        self.assertIn(
            "Do not claim live preview readiness for this output.",
            director.planning_focus,
        )
        self.assertEqual(
            director.next_actions,
            ("Requested scope has no current live preview runtime support.",),
        )
        self.assertTrue(
            any("Constraint solver" in item for item in director.evidence),
            director.evidence,
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
