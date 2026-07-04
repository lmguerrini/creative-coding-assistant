import unittest
from dataclasses import dataclass

from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    artifact_refiner_prompt_lines,
    build_rendered_prompt_request,
    derive_artifact_refiner_profile,
    derive_creative_assistant_director_brief,
    derive_creative_reasoning_result,
)
from test_artifact_critic import _stack as _critic_stack


class ArtifactRefinerTests(unittest.TestCase):
    def test_derives_artifact_refiner_for_strong_plans(self) -> None:
        stack = _stack(
            "Generate a p5.js recursive mandala with bounded particles, "
            "dependency notes, runtime notes, capability caveats, and critic "
            "metadata."
        )
        refiner = stack.artifact_refiner

        self.assertEqual(refiner.role, "artifact_refiner")
        self.assertGreater(refiner.refinement_confidence, 0)
        self.assertTrue(refiner.refinement_summary)
        self.assertTrue(refiner.recommended_improvements)
        self.assertTrue(refiner.priority_improvements)
        self.assertTrue(refiner.refinement_candidates)
        self.assertTrue(refiner.implementation_suggestions)
        self.assertTrue(refiner.prompt_guidance)
        self.assertIn("does not modify artifacts", refiner.authority_boundary)
        self.assertTrue(artifact_refiner_prompt_lines(refiner))

    def test_flags_weak_artifact_plans(self) -> None:
        stack = _stack(
            "Generate a dense high-complexity p5.js particle system with "
            "performance-sensitive output caveats."
        )
        weak_plan = stack.runtime_base.artifact_plan.model_copy(
            update={
                "implementation_risks": (
                    "Dense particle counts can pressure frame rate.",
                    "Complex scope can reduce inspectability.",
                ),
            }
        )
        weak_critic = stack.critic_stack.artifact_critic.model_copy(
            update={
                "risk_assessment": "high",
                "weaknesses": (
                    "Dense particle counts can pressure frame rate.",
                    "Complex scope can reduce inspectability.",
                ),
                "scalability_concerns": (
                    "Dense particle counts can pressure frame rate.",
                ),
                "complexity_concerns": ("Complex scope can reduce inspectability.",),
            }
        )
        refiner = derive_artifact_refiner_profile(
            request=stack.runtime_base.request,
            route_decision=stack.runtime_base.route,
            artifact_plan=weak_plan,
            artifact_dependency_graph=stack.runtime_base.artifact_dependency_graph,
            runtime_compatibility=stack.runtime_base.runtime_compatibility,
            artifact_capability_matrix=stack.artifact_capability_matrix,
            multi_artifact_strategy=stack.multi_artifact_strategy,
            artifact_critic=weak_critic,
        )

        self.assertTrue(refiner.scalability_improvements)
        self.assertTrue(refiner.complexity_reductions)
        self.assertTrue(
            any(
                "Critic risk" in item or "risk" in item
                for item in refiner.risk_reductions
            ),
            refiner.risk_reductions,
        )

    def test_uses_critique_driven_improvements(self) -> None:
        stack = _stack(
            "Generate p5.js primary code with dependency notes, runtime notes, "
            "capability notes, and trade-off notes."
        )
        refiner = stack.artifact_refiner

        self.assertTrue(
            any("critic" in item.lower() for item in refiner.recommended_improvements),
            refiner.recommended_improvements,
        )
        self.assertTrue(
            any("advisory" in item.lower() for item in refiner.risk_reductions),
            refiner.risk_reductions,
        )

    def test_flags_conflicting_improvement_paths(self) -> None:
        stack = _stack(
            "Generate a p5.js visual system with conflicting dependency and "
            "runtime assumptions."
        )
        conflicting_graph = stack.runtime_base.artifact_dependency_graph.model_copy(
            update={
                "dependency_conflicts": (
                    "Runtime-facing dependency conflicts with output structure.",
                ),
                "blocking_dependencies": (
                    "Required downstream consumer assumption is unresolved.",
                ),
            }
        )
        conflicting_critic = stack.critic_stack.artifact_critic.model_copy(
            update={
                "risk_assessment": "high",
                "dependency_concerns": (
                    "Runtime-facing dependency conflicts with output structure.",
                ),
                "runtime_concerns": (
                    "Unsupported runtimes should not be treated as viable targets.",
                ),
            }
        )
        refiner = derive_artifact_refiner_profile(
            request=stack.runtime_base.request,
            route_decision=stack.runtime_base.route,
            artifact_plan=stack.runtime_base.artifact_plan,
            artifact_dependency_graph=conflicting_graph,
            runtime_compatibility=stack.runtime_base.runtime_compatibility,
            artifact_capability_matrix=stack.artifact_capability_matrix,
            multi_artifact_strategy=stack.multi_artifact_strategy,
            artifact_critic=conflicting_critic,
        )

        self.assertTrue(refiner.dependency_improvements)
        self.assertTrue(refiner.runtime_improvements)
        self.assertGreaterEqual(len(refiner.alternative_refinement_paths), 2)

    def test_handles_missing_information_scenarios(self) -> None:
        stack = _stack("Explain a possible creative coding artifact.")
        refiner = derive_artifact_refiner_profile(
            request=stack.runtime_base.request,
            route_decision=None,
            artifact_plan=None,
            artifact_dependency_graph=None,
            runtime_compatibility=None,
            artifact_capability_matrix=None,
            multi_artifact_strategy=None,
            artifact_critic=None,
        )

        self.assertLess(refiner.refinement_confidence, 0.5)
        self.assertTrue(refiner.priority_improvements)
        self.assertTrue(refiner.hitl_questions)
        self.assertTrue(
            any("missing" in item.lower() for item in refiner.recommended_improvements),
            refiner.recommended_improvements,
        )

    def test_integrates_with_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack(
            "Generate a symbolic p5.js phoenix mandala with artifact refiner "
            "metadata before Director and Reasoning."
        )
        prompt_input = stack.runtime_base.prompt_input.model_copy(
            update={
                "creative_intent": stack.runtime_base.intent,
                "creative_hierarchy": stack.runtime_base.hierarchy,
                "creative_strategy": stack.runtime_base.strategy,
                "creative_techniques": stack.runtime_base.techniques,
                "creative_plan": stack.runtime_base.plan,
                "creative_constraints": stack.runtime_base.constraints,
                "creative_constraint_priorities": stack.runtime_base.prioritization,
                "runtime_capabilities": stack.runtime_base.runtime_capabilities,
                "creative_tradeoffs": stack.runtime_base.tradeoffs,
                "artifact_plan": stack.runtime_base.artifact_plan,
                "artifact_dependency_graph": (
                    stack.runtime_base.artifact_dependency_graph
                ),
                "runtime_compatibility": stack.runtime_base.runtime_compatibility,
                "artifact_capability_matrix": stack.artifact_capability_matrix,
                "multi_artifact_strategy": stack.multi_artifact_strategy,
                "artifact_critic": stack.critic_stack.artifact_critic,
                "artifact_refiner": stack.artifact_refiner,
                "creative_director": stack.director,
                "creative_reasoning": stack.reasoning,
            }
        )

        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=stack.runtime_base.route,
                prompt_input=prompt_input,
            )
        )
        system = rendered.sections[0].content

        self.assertIn("Artifact Refiner:", system)
        self.assertIn("Refinement confidence:", system)
        self.assertTrue(
            any("Artifact refiner:" in item for item in stack.director.planning_focus),
            stack.director.planning_focus,
        )
        self.assertIn(
            "artifact_refiner",
            {item.source for item in stack.reasoning.evidence_chain},
        )
        self.assertEqual(
            stack.artifact_refiner.model_dump(mode="json")["role"],
            "artifact_refiner",
        )
        self.assertNotIn("runtime auto-selection enabled", system.lower())
        self.assertNotIn("provider routing enabled", system.lower())


@dataclass(frozen=True)
class _DerivedStack:
    critic_stack: object
    runtime_base: object
    artifact_capability_matrix: object
    multi_artifact_strategy: object
    artifact_refiner: object
    director: object
    reasoning: object


def _stack(query: str) -> _DerivedStack:
    critic_stack = _critic_stack(query)
    runtime_base = critic_stack.base.base.base
    artifact_capability_matrix = critic_stack.base.base.artifact_capability_matrix
    multi_artifact_strategy = critic_stack.multi_artifact_strategy
    refiner = derive_artifact_refiner_profile(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        artifact_plan=runtime_base.artifact_plan,
        artifact_dependency_graph=runtime_base.artifact_dependency_graph,
        runtime_compatibility=runtime_base.runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=critic_stack.artifact_critic,
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
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=critic_stack.artifact_critic,
        artifact_refiner=refiner,
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
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=critic_stack.artifact_critic,
        artifact_refiner=refiner,
    )
    return _DerivedStack(
        critic_stack=critic_stack,
        runtime_base=runtime_base,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_refiner=refiner,
        director=director,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    unittest.main()
