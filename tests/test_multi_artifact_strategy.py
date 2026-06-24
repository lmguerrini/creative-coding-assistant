import unittest
from dataclasses import dataclass

from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    build_rendered_prompt_request,
    derive_creative_assistant_director_brief,
    derive_creative_reasoning_result,
    derive_multi_artifact_strategy,
    multi_artifact_strategy_prompt_lines,
)
from test_artifact_capability_matrix import _stack as _capability_stack


class MultiArtifactStrategyTests(unittest.TestCase):
    def test_derives_multi_artifact_strategy_metadata(self) -> None:
        stack = _stack(
            "Generate a p5.js recursive mandala with primary code, dependency "
            "notes, runtime notes, and target capability caveats."
        )
        strategy = stack.multi_artifact_strategy

        self.assertEqual(strategy.role, "multi_artifact_strategy")
        self.assertTrue(strategy.artifact_strategy_summary)
        self.assertEqual(strategy.primary_artifact.artifact_id, "primary_artifact")
        self.assertEqual(strategy.primary_artifact.role, "primary")
        self.assertTrue(strategy.supporting_artifacts)
        self.assertEqual(strategy.artifact_sequence[0].artifact_id, "primary_artifact")
        self.assertEqual(strategy.artifact_sequence[0].action, "produce")
        self.assertTrue(strategy.artifact_priority)
        self.assertTrue(strategy.artifact_grouping)
        self.assertTrue(strategy.artifact_separation_strategy)
        self.assertTrue(strategy.artifact_combination_strategy)
        self.assertTrue(strategy.artifact_dependency_order)
        self.assertTrue(strategy.artifact_handoff_points)
        self.assertTrue(strategy.runtime_aware_artifact_strategy)
        self.assertTrue(strategy.capability_aware_artifact_strategy)
        self.assertIn(
            strategy.combination_mode,
            {
                "primary_with_supporting_sections",
                "separated_parallel_sections",
                "defer_combination",
            },
        )
        self.assertTrue(strategy.risk_areas)
        self.assertTrue(strategy.prompt_guidance)
        self.assertIn("does not generate artifacts", strategy.authority_boundary)

        prompt_lines = multi_artifact_strategy_prompt_lines(strategy)
        self.assertTrue(prompt_lines)
        self.assertTrue(any("Artifact sequence:" in item for item in prompt_lines))
        self.assertTrue(any("Combination mode:" in item for item in prompt_lines))

    def test_integrates_with_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack(
            "Generate a symbolic p5.js phoenix mandala with separated primary "
            "code, dependency guidance, runtime guidance, and capability notes."
        )
        base = stack.base.base
        prompt_input = base.prompt_input.model_copy(
            update={
                "creative_intent": base.intent,
                "creative_hierarchy": base.hierarchy,
                "creative_strategy": base.strategy,
                "creative_techniques": base.techniques,
                "creative_plan": base.plan,
                "creative_constraints": base.constraints,
                "creative_constraint_priorities": base.prioritization,
                "runtime_capabilities": base.runtime_capabilities,
                "creative_tradeoffs": base.tradeoffs,
                "artifact_plan": base.artifact_plan,
                "artifact_dependency_graph": base.artifact_dependency_graph,
                "runtime_compatibility": base.runtime_compatibility,
                "artifact_capability_matrix": stack.base.artifact_capability_matrix,
                "multi_artifact_strategy": stack.multi_artifact_strategy,
                "creative_director": stack.director,
                "creative_reasoning": stack.reasoning,
            }
        )

        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=base.route,
                prompt_input=prompt_input,
            )
        )
        system = rendered.sections[0].content

        self.assertIn("Multi-Artifact Strategy:", system)
        self.assertIn("Artifact sequence:", system)
        self.assertTrue(
            any(
                "Multi-artifact strategy:" in item
                for item in stack.director.planning_focus
            ),
            stack.director.planning_focus,
        )
        self.assertIn(
            "multi_artifact_strategy",
            {item.source for item in stack.reasoning.evidence_chain},
        )
        self.assertEqual(
            stack.multi_artifact_strategy.model_dump(mode="json")["role"],
            "multi_artifact_strategy",
        )
        self.assertNotIn("runtime auto-selection enabled", system.lower())
        self.assertNotIn("provider routing enabled", system.lower())


@dataclass(frozen=True)
class _DerivedStack:
    base: object
    multi_artifact_strategy: object
    director: object
    reasoning: object


def _stack(query: str) -> _DerivedStack:
    base = _capability_stack(query)
    runtime_base = base.base
    strategy = derive_multi_artifact_strategy(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        artifact_plan=runtime_base.artifact_plan,
        artifact_dependency_graph=runtime_base.artifact_dependency_graph,
        runtime_capabilities=runtime_base.runtime_capabilities,
        runtime_compatibility=runtime_base.runtime_compatibility,
        artifact_capability_matrix=base.artifact_capability_matrix,
        creative_plan=runtime_base.plan,
        creative_constraints=runtime_base.constraints,
        creative_tradeoffs=runtime_base.tradeoffs,
    )
    director = derive_creative_assistant_director_brief(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        creative_translation=runtime_base.prompt_input.creative_translation,
        creative_intent=runtime_base.intent,
        creative_hierarchy=runtime_base.hierarchy,
        creative_strategy=runtime_base.strategy,
        creative_techniques=runtime_base.techniques,
        creative_plan=runtime_base.plan,
        creative_constraints=runtime_base.constraints,
        creative_constraint_priorities=runtime_base.prioritization,
        runtime_capabilities=runtime_base.runtime_capabilities,
        creative_tradeoffs=runtime_base.tradeoffs,
        artifact_plan=runtime_base.artifact_plan,
        artifact_dependency_graph=runtime_base.artifact_dependency_graph,
        runtime_compatibility=runtime_base.runtime_compatibility,
        artifact_capability_matrix=base.artifact_capability_matrix,
        multi_artifact_strategy=strategy,
    )
    reasoning = derive_creative_reasoning_result(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        creative_translation=runtime_base.prompt_input.creative_translation,
        creative_intent=runtime_base.intent,
        creative_hierarchy=runtime_base.hierarchy,
        creative_plan=runtime_base.plan,
        creative_director=director,
        creative_constraints=runtime_base.constraints,
        creative_constraint_priorities=runtime_base.prioritization,
        creative_strategy=runtime_base.strategy,
        creative_techniques=runtime_base.techniques,
        runtime_capabilities=runtime_base.runtime_capabilities,
        creative_tradeoffs=runtime_base.tradeoffs,
        artifact_plan=runtime_base.artifact_plan,
        artifact_dependency_graph=runtime_base.artifact_dependency_graph,
        runtime_compatibility=runtime_base.runtime_compatibility,
        artifact_capability_matrix=base.artifact_capability_matrix,
        multi_artifact_strategy=strategy,
    )
    return _DerivedStack(
        base=base,
        multi_artifact_strategy=strategy,
        director=director,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    unittest.main()
