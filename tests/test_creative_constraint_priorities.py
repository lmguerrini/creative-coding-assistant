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
    creative_constraint_priorities_prompt_lines,
    derive_creative_constraint_priorities,
    derive_creative_constraint_solution,
    derive_creative_execution_plan,
    derive_creative_hierarchy_plan,
    derive_creative_intent_decomposition,
    derive_creative_strategy_profile,
    derive_creative_technique_profile,
    derive_creative_tradeoff_profile,
    derive_runtime_capability_profile,
)


class CreativeConstraintPrioritizerTests(unittest.TestCase):
    def test_prioritizes_constraints_beyond_solver_axis_copying(self) -> None:
        stack = _stack(
            "Create an initiatic death-and-rebirth ritual visual. Symbolic "
            "transformation and emotional arc are non-negotiable. Geometry "
            "should support the myth. Interaction can be optional and preview "
            "polish can be sacrificed if needed."
        )

        prioritization = stack.prioritization
        non_negotiable = _categories(prioritization.non_negotiable_constraints)
        relaxable = _categories(prioritization.relaxable_constraints)
        sacrificial = _categories(prioritization.sacrificial_constraints)
        solver_axes = {item.axis for item in stack.constraints.active_constraints}

        self.assertEqual(prioritization.role, "creative_constraint_prioritizer")
        self.assertIn("symbolic_fidelity", non_negotiable)
        self.assertIn("emotional_fidelity", non_negotiable)
        self.assertIn("interaction_complexity", relaxable)
        self.assertIn("previewability", sacrificial)
        self.assertNotIn("symbolic_fidelity", solver_axes)
        self.assertTrue(prioritization.priority_rationale)
        self.assertTrue(creative_constraint_priorities_prompt_lines(prioritization))
        self.assertIn(
            "does not auto-select runtimes",
            prioritization.authority_boundary,
        )

    def test_identifies_sacrificial_conflicts_and_hitl_questions(self) -> None:
        stack = _stack(
            "Generate a cinematic visual spectacle with dense complex layers "
            "and interactive controls, but it must stay 60 fps on mobile and "
            "browser-safe. Interaction can be dropped if needed."
        )

        prioritization = stack.prioritization
        non_negotiable = _categories(prioritization.non_negotiable_constraints)
        sacrificial = _categories(prioritization.sacrificial_constraints)

        self.assertIn("performance", non_negotiable)
        self.assertIn("runtime_safety", non_negotiable)
        self.assertIn("interaction_complexity", sacrificial)
        self.assertTrue(prioritization.conflict_relationships)
        self.assertTrue(prioritization.negotiation_notes)
        self.assertTrue(prioritization.hitl_questions)

    def test_prompt_rendering_includes_priority_guidance(self) -> None:
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
            }
        )

        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=stack.route,
                prompt_input=prompt_input,
            )
        )
        system = rendered.sections[0].content

        self.assertIn("Creative Constraint Prioritizer:", system)
        self.assertIn("Non-negotiable constraint:", system)
        self.assertIn("Constraint priority guidance:", system)
        self.assertNotIn("runtime auto-selection", system)


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
    )


def _categories(values: tuple[object, ...]) -> set[str]:
    return {str(item.category) for item in values}


if __name__ == "__main__":
    unittest.main()
