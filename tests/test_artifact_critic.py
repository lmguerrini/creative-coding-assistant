import unittest
from dataclasses import dataclass

from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    artifact_critic_prompt_lines,
    build_rendered_prompt_request,
    derive_artifact_critic_profile,
    derive_creative_assistant_director_brief,
    derive_creative_reasoning_result,
)
from test_multi_artifact_strategy import _stack as _strategy_stack


class ArtifactCriticTests(unittest.TestCase):
    def test_derives_artifact_critic_for_strong_plans(self) -> None:
        stack = _stack(
            "Generate a p5.js recursive mandala with bounded particles, "
            "dependency notes, runtime notes, and capability caveats."
        )
        critic = stack.artifact_critic

        self.assertEqual(critic.role, "artifact_critic")
        self.assertGreater(critic.critique_confidence, 0)
        self.assertTrue(critic.critique_summary)
        self.assertTrue(critic.strengths)
        self.assertIn(critic.risk_assessment, {"low", "medium", "high", "blocked"})
        self.assertTrue(critic.unsupported_assumptions)
        self.assertTrue(critic.prompt_guidance)
        self.assertIn("does not modify artifacts", critic.authority_boundary)
        self.assertTrue(artifact_critic_prompt_lines(critic))

    def test_flags_weak_artifact_plans(self) -> None:
        stack = _stack(
            "Generate a dense high-complexity p5.js particle system with "
            "performance-sensitive output caveats."
        )
        runtime_base = stack.base.base.base
        weak_plan = runtime_base.artifact_plan.model_copy(
            update={
                "implementation_risks": (
                    "Dense particle counts can pressure frame rate.",
                    "Complex scope can reduce inspectability.",
                ),
            }
        )
        critic = derive_artifact_critic_profile(
            request=runtime_base.request,
            route_decision=runtime_base.route,
            artifact_plan=weak_plan,
            artifact_dependency_graph=runtime_base.artifact_dependency_graph,
            runtime_compatibility=runtime_base.runtime_compatibility,
            artifact_capability_matrix=stack.base.base.artifact_capability_matrix,
            multi_artifact_strategy=stack.multi_artifact_strategy,
        )

        self.assertTrue(critic.scalability_concerns)
        self.assertTrue(critic.complexity_concerns)
        self.assertIn(critic.risk_assessment, {"medium", "high"})

    def test_critiques_multi_artifact_plans(self) -> None:
        stack = _stack(
            "Generate p5.js primary code with dependency notes, runtime notes, "
            "capability notes, and trade-off notes."
        )
        critic = stack.artifact_critic

        self.assertTrue(
            any("Multi-artifact strategy" in item for item in critic.strengths),
            critic.strengths,
        )
        self.assertTrue(critic.improvement_opportunities)

    def test_flags_conflicting_plans(self) -> None:
        stack = _stack(
            "Generate a p5.js visual system with conflicting dependency and "
            "runtime assumptions."
        )
        runtime_base = stack.base.base.base
        conflicting_graph = runtime_base.artifact_dependency_graph.model_copy(
            update={
                "dependency_conflicts": (
                    "Runtime-facing dependency conflicts with output structure.",
                ),
                "blocking_dependencies": (
                    "Required downstream consumer assumption is unresolved.",
                ),
            }
        )
        critic = derive_artifact_critic_profile(
            request=runtime_base.request,
            route_decision=runtime_base.route,
            artifact_plan=runtime_base.artifact_plan,
            artifact_dependency_graph=conflicting_graph,
            runtime_compatibility=runtime_base.runtime_compatibility,
            artifact_capability_matrix=stack.base.base.artifact_capability_matrix,
            multi_artifact_strategy=stack.multi_artifact_strategy,
        )

        self.assertTrue(critic.dependency_concerns)
        self.assertIn(critic.risk_assessment, {"medium", "high"})

    def test_handles_missing_information_cases(self) -> None:
        stack = _stack("Explain a possible creative coding artifact.")
        runtime_base = stack.base.base.base
        critic = derive_artifact_critic_profile(
            request=runtime_base.request,
            route_decision=None,
            artifact_plan=None,
            artifact_dependency_graph=None,
            runtime_compatibility=None,
            artifact_capability_matrix=None,
            multi_artifact_strategy=None,
        )

        self.assertEqual(critic.risk_assessment, "blocked")
        self.assertTrue(critic.missing_information)
        self.assertTrue(critic.hitl_questions)

    def test_integrates_with_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack(
            "Generate a symbolic p5.js phoenix mandala with artifact critic "
            "metadata before Director and Reasoning."
        )
        runtime_base = stack.base.base.base
        prompt_input = runtime_base.prompt_input.model_copy(
            update={
                "creative_intent": runtime_base.intent,
                "creative_hierarchy": runtime_base.hierarchy,
                "creative_strategy": runtime_base.strategy,
                "creative_techniques": runtime_base.techniques,
                "creative_plan": runtime_base.plan,
                "creative_constraints": runtime_base.constraints,
                "creative_constraint_priorities": runtime_base.prioritization,
                "runtime_capabilities": runtime_base.runtime_capabilities,
                "creative_tradeoffs": runtime_base.tradeoffs,
                "artifact_plan": runtime_base.artifact_plan,
                "artifact_dependency_graph": runtime_base.artifact_dependency_graph,
                "runtime_compatibility": runtime_base.runtime_compatibility,
                "artifact_capability_matrix": (
                    stack.base.base.artifact_capability_matrix
                ),
                "multi_artifact_strategy": stack.multi_artifact_strategy,
                "artifact_critic": stack.artifact_critic,
                "creative_director": stack.director,
                "creative_reasoning": stack.reasoning,
            }
        )

        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=runtime_base.route,
                prompt_input=prompt_input,
            )
        )
        system = rendered.sections[0].content

        self.assertIn("Artifact Critic:", system)
        self.assertIn("Critique risk assessment:", system)
        self.assertTrue(
            any("Artifact critic:" in item for item in stack.director.planning_focus),
            stack.director.planning_focus,
        )
        self.assertIn(
            "artifact_critic",
            {item.source for item in stack.reasoning.evidence_chain},
        )
        self.assertEqual(
            stack.artifact_critic.model_dump(mode="json")["role"],
            "artifact_critic",
        )
        self.assertNotIn("runtime auto-selection enabled", system.lower())
        self.assertNotIn("provider routing enabled", system.lower())


@dataclass(frozen=True)
class _DerivedStack:
    base: object
    multi_artifact_strategy: object
    artifact_critic: object
    director: object
    reasoning: object


def _stack(query: str) -> _DerivedStack:
    base = _strategy_stack(query)
    runtime_base = base.base.base
    critic = derive_artifact_critic_profile(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        artifact_plan=runtime_base.artifact_plan,
        artifact_dependency_graph=runtime_base.artifact_dependency_graph,
        runtime_compatibility=runtime_base.runtime_compatibility,
        artifact_capability_matrix=base.base.artifact_capability_matrix,
        multi_artifact_strategy=base.multi_artifact_strategy,
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
        artifact_capability_matrix=base.base.artifact_capability_matrix,
        multi_artifact_strategy=base.multi_artifact_strategy,
        artifact_critic=critic,
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
        artifact_capability_matrix=base.base.artifact_capability_matrix,
        multi_artifact_strategy=base.multi_artifact_strategy,
        artifact_critic=critic,
    )
    return _DerivedStack(
        base=base,
        multi_artifact_strategy=base.multi_artifact_strategy,
        artifact_critic=critic,
        director=director,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    unittest.main()
