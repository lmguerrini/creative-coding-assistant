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
    build_prompt_input_request,
    build_rendered_prompt_request,
    creative_quality_prediction_prompt_lines,
    derive_creative_assistant_director_brief,
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
    derive_runtime_capability_profile,
)


class CreativeQualityPredictorTests(unittest.TestCase):
    def test_predicts_high_readiness_for_specific_coherent_plan(self) -> None:
        stack = _stack(
            "Generate a p5.js ritual mandala with clear sacred geometry, "
            "gold and blue palette, slow pulse, tranquil awe, single visual "
            "candidate."
        )

        prediction = stack.quality_prediction
        strongest_dimensions = {
            item.dimension for item in prediction.strongest_quality_signals
        }

        self.assertEqual(prediction.role, "creative_quality_predictor")
        self.assertIn(
            prediction.predicted_quality_level,
            {"strong", "promising"},
        )
        self.assertGreaterEqual(prediction.readiness_score, 68)
        self.assertFalse(prediction.missing_information)
        self.assertFalse(prediction.hitl_questions)
        self.assertTrue(
            {"intent_clarity", "geometric_formal_clarity"}.intersection(
                strongest_dimensions
            ),
            strongest_dimensions,
        )
        self.assertIn(
            "pre-generation readiness",
            prediction.authority_boundary,
        )
        self.assertIn(
            "Predicted quality:",
            " ".join(creative_quality_prediction_prompt_lines(prediction)),
        )

    def test_identifies_ambiguous_readiness_and_missing_information(self) -> None:
        stack = _stack(
            "Make something profound and symbolic, maybe interactive or "
            "audio-reactive, but keep it simple and cinematic."
        )

        prediction = stack.quality_prediction
        weakest_dimensions = {
            item.dimension for item in prediction.weakest_quality_signals
        }

        self.assertIn(
            prediction.predicted_quality_level,
            {"ambiguous", "risky", "blocked"},
        )
        self.assertLess(prediction.readiness_score, 68)
        self.assertTrue(prediction.missing_information)
        self.assertTrue(prediction.hitl_questions)
        self.assertIn("intent_clarity", weakest_dimensions)
        self.assertTrue(
            any("palette" in item.lower() for item in prediction.missing_information),
            prediction.missing_information,
        )
        self.assertTrue(
            any("interaction" in item.lower() for item in prediction.hitl_questions),
            prediction.hitl_questions,
        )

    def test_identifies_quality_risks_and_likely_failure_modes(self) -> None:
        stack = _stack(
            "Generate a dense cinematic particle spectacle with complex "
            "interaction and audio, must stay 60 fps on mobile and "
            "browser-safe, symbolic fidelity non-negotiable but visual and "
            "geometric direction unspecified."
        )

        prediction = stack.quality_prediction

        self.assertLess(prediction.readiness_score, 68)
        self.assertTrue(prediction.quality_risks)
        self.assertTrue(prediction.likely_failure_modes)
        self.assertTrue(prediction.suggested_improvements)
        self.assertTrue(
            any("performance" in item.lower() for item in prediction.quality_risks),
            prediction.quality_risks,
        )
        self.assertTrue(
            any("symbolic" in item.lower() for item in prediction.missing_information),
            prediction.missing_information,
        )
        self.assertTrue(
            any(
                "performance" in item.lower() or "generic" in item.lower()
                for item in prediction.likely_failure_modes
            ),
            prediction.likely_failure_modes,
        )

    def test_integrates_with_prompt_director_and_reasoning_metadata(self) -> None:
        stack = _stack(
            "Generate a symbolic spiral with optional interaction. Keep the "
            "symbolic fidelity non-negotiable and sacrifice preview polish if "
            "needed."
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

        self.assertIn("Creative Quality Predictor:", system)
        self.assertIn("Quality predictor guidance:", system)
        self.assertTrue(
            any("Quality readiness:" in item for item in stack.director.planning_focus),
            stack.director.planning_focus,
        )
        self.assertIn(
            "quality_predictor",
            {item.source for item in stack.reasoning.evidence_chain},
        )
        self.assertNotIn("Artifact Selection Engine", system)


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
    director: object
    reasoning: object


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
        director=director,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    unittest.main()
