import unittest

from creative_coding_assistant.contracts import (
    AssistantImageReference,
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    CreativeExecutionPlan,
    JinjaPromptRenderer,
    RouteCapability,
    RouteDecision,
    RouteName,
    StructuredPromptInputBuilder,
    build_prompt_input_request,
    build_rendered_prompt_request,
    creative_execution_plan_prompt_lines,
    derive_creative_execution_plan,
)


class CreativePlanningTests(unittest.TestCase):
    def test_planner_derives_runtime_strategy_from_existing_translation(self) -> None:
        request = AssistantRequest(
            query="Create a luminous p5.js mandala with low-frequency audio pulse.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
            clarificationResponse="Visual sketch",
            attachments=(
                AssistantImageReference(
                    id="ref-1",
                    name="palette.png",
                    mimeType="image/png",
                    sizeBytes=68,
                    dataUrl=(
                        "data:image/png;base64,"
                        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
                        "AQUBAScY42YAAAAASUVORK5CYII="
                    ),
                ),
            ),
        )
        prompt_input = StructuredPromptInputBuilder().build(
            build_prompt_input_request(
                assistant_request=request,
                route_decision=_route_decision(CreativeCodingDomain.P5_JS),
                assembled_context=None,
            )
        )

        plan = derive_creative_execution_plan(
            request=request,
            route_decision=_route_decision(CreativeCodingDomain.P5_JS),
            creative_translation=prompt_input.creative_translation,
            retrieval_chunk_count=3,
        )

        self.assertEqual(plan.output_modality.value, "visual")
        self.assertEqual(plan.recommended_runtime, "p5")
        self.assertEqual(plan.recommended_renderer_id, "surface.p5")
        self.assertTrue(plan.runtime_available)
        self.assertGreaterEqual(plan.estimated_token_cost, 2500)
        self.assertIn(
            "Clarification answer: Visual sketch.",
            plan.evidence,
        )
        self.assertTrue(
            any("Reference fusion" in item for item in plan.evidence),
            plan.evidence,
        )
        self.assertTrue(
            any("Geometry guidance" in item for item in plan.evidence),
            plan.evidence,
        )
        self.assertFalse(
            any("sacred" in item.lower() for item in plan.evidence),
            plan.evidence,
        )
        self.assertTrue(
            any("audio-reactive" in item.lower() for item in plan.plan_steps),
            plan.plan_steps,
        )

    def test_planner_uses_legacy_fallback_without_translation(self) -> None:
        request = AssistantRequest(
            query="Make a compact visual study.",
            mode=AssistantMode.GENERATE,
        )

        plan = derive_creative_execution_plan(
            request=request,
            route_decision=None,
            creative_translation=None,
        )

        self.assertEqual(plan.output_modality.value, "visual")
        self.assertEqual(plan.export_readiness, "partial")
        self.assertFalse(plan.runtime_available)
        self.assertIsNone(plan.recommended_runtime)
        self.assertIn("No explicit runtime support", plan.runtime_support_summary)

    def test_plan_prompt_lines_do_not_generate_code(self) -> None:
        plan = CreativeExecutionPlan(
            outputModality="visual",
            generationStrategy="Generate one visual candidate.",
            recommendedRuntime="p5",
            recommendedRendererId="surface.p5",
            recommendedPreviewTarget="browser_sandbox",
            recommendedShaderStyle="glow",
            candidateCount=1,
            refinementBudget=1,
            expectedComplexity="medium",
            estimatedTokenCost=2600,
            exportReadiness="ready",
            runtimeAvailable=True,
            runtimeSupportSummary="p5.js browser preview is available.",
            planSteps=("Use the translated creative intent.",),
            constraints=("Keep code browser-safe.",),
            evidence=("Route selected: generate.",),
        )

        rendered = "\n".join(creative_execution_plan_prompt_lines(plan))

        self.assertIn("Output modality: visual.", rendered)
        self.assertIn("Recommended runtime: p5.", rendered)
        self.assertNotIn("function setup", rendered)
        self.assertNotIn("```", rendered)

    def test_renderer_includes_existing_plan_in_system_section(self) -> None:
        request = AssistantRequest(
            query="Generate a p5.js field.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route = _route_decision(CreativeCodingDomain.P5_JS)
        prompt_input = StructuredPromptInputBuilder().build(
            build_prompt_input_request(
                assistant_request=request,
                route_decision=route,
                assembled_context=None,
            )
        )
        plan = derive_creative_execution_plan(
            request=request,
            route_decision=route,
            creative_translation=prompt_input.creative_translation,
        )
        prompt_input = prompt_input.model_copy(update={"creative_plan": plan})

        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=route,
                prompt_input=prompt_input,
            )
        )

        self.assertIn("Creative Execution Plan:", rendered.sections[0].content)
        self.assertIn("Recommended runtime: p5.", rendered.sections[0].content)


def _route_decision(domain: CreativeCodingDomain) -> RouteDecision:
    return RouteDecision(
        route=RouteName.GENERATE,
        mode=AssistantMode.GENERATE,
        domain=domain,
        domains=(domain,),
        capabilities=(RouteCapability.TOOL_USE,),
    )
