import unittest
from dataclasses import dataclass

from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    artifact_merge_planner_prompt_lines,
    build_rendered_prompt_request,
    derive_artifact_merge_planner_profile,
    derive_creative_assistant_director_brief,
    derive_creative_reasoning_result,
)
from test_artifact_intelligence_synthesis import _stack as _synthesis_stack


class ArtifactMergePlannerTests(unittest.TestCase):
    def test_derives_single_artifact_no_merge_plan(self) -> None:
        stack = _stack("Generate a single p5.js sketch with metadata caveats.")
        merge = derive_artifact_merge_planner_profile(
            request=stack.runtime_base.request,
            route_decision=stack.runtime_base.route,
            artifact_plan=stack.runtime_base.artifact_plan,
            artifact_dependency_graph=stack.runtime_base.artifact_dependency_graph,
            runtime_compatibility=stack.runtime_base.runtime_compatibility,
            artifact_capability_matrix=stack.artifact_capability_matrix,
            multi_artifact_strategy=None,
            artifact_critic=stack.synthesis_stack.refiner_stack.critic_stack.artifact_critic,
            artifact_refiner=stack.synthesis_stack.refiner_stack.artifact_refiner,
            artifact_intelligence_synthesis=stack.artifact_intelligence_synthesis,
        )

        self.assertEqual(merge.role, "artifact_merge_planner")
        self.assertEqual(merge.merge_strategy, "single_artifact_no_merge")
        self.assertIn("Do not merge", merge.recommended_merge_path)
        self.assertTrue(merge.artifact_boundaries)
        self.assertTrue(merge.rejected_merge_paths)

    def test_derives_multi_artifact_merge_plan(self) -> None:
        stack = _stack(
            "Generate primary p5.js code with dependency notes, runtime notes, "
            "and capability caveats as supporting artifacts."
        )
        merge = stack.artifact_merge_planner

        self.assertEqual(merge.role, "artifact_merge_planner")
        self.assertGreater(merge.merge_confidence, 0)
        self.assertIn(
            merge.merge_strategy,
            {
                "primary_with_supporting_sections",
                "separated_advisory_sections",
                "defer_merge_preserve_separation",
            },
        )
        self.assertTrue(merge.artifact_join_points)
        self.assertTrue(merge.integration_order)
        self.assertTrue(artifact_merge_planner_prompt_lines(merge))

    def test_derives_clear_merge_paths(self) -> None:
        stack = _stack(
            "Generate p5.js primary code with clear supporting notes and no "
            "dependency conflicts."
        )
        clear_graph = stack.runtime_base.artifact_dependency_graph.model_copy(
            update={"dependency_conflicts": (), "blocking_dependencies": ()}
        )
        clear_critic = (
            stack.synthesis_stack.refiner_stack.critic_stack.artifact_critic.model_copy(
                update={"risk_assessment": "low", "weaknesses": ()}
            )
        )
        clear_synthesis = stack.artifact_intelligence_synthesis.model_copy(
            update={"implementation_risk": "low", "major_risks": ()}
        )
        merge = derive_artifact_merge_planner_profile(
            request=stack.runtime_base.request,
            route_decision=stack.runtime_base.route,
            artifact_plan=stack.runtime_base.artifact_plan,
            artifact_dependency_graph=clear_graph,
            runtime_compatibility=stack.runtime_base.runtime_compatibility,
            artifact_capability_matrix=stack.artifact_capability_matrix,
            multi_artifact_strategy=stack.multi_artifact_strategy,
            artifact_critic=clear_critic,
            artifact_refiner=stack.synthesis_stack.refiner_stack.artifact_refiner,
            artifact_intelligence_synthesis=clear_synthesis,
        )

        self.assertIn(
            merge.merge_strategy,
            {"primary_with_supporting_sections", "separated_advisory_sections"},
        )
        self.assertTrue(merge.recommended_merge_path)
        self.assertTrue(merge.alternative_merge_paths)

    def test_flags_conflicting_merge_paths(self) -> None:
        stack = _stack(
            "Generate a p5.js artifact with conflicting dependency merge paths."
        )
        conflicting_graph = stack.runtime_base.artifact_dependency_graph.model_copy(
            update={
                "dependency_conflicts": (
                    "Runtime notes conflict with primary output structure.",
                ),
                "blocking_dependencies": (
                    "Primary artifact depends on unresolved downstream consumer.",
                ),
            }
        )
        merge = derive_artifact_merge_planner_profile(
            request=stack.runtime_base.request,
            route_decision=stack.runtime_base.route,
            artifact_plan=stack.runtime_base.artifact_plan,
            artifact_dependency_graph=conflicting_graph,
            runtime_compatibility=stack.runtime_base.runtime_compatibility,
            artifact_capability_matrix=stack.artifact_capability_matrix,
            multi_artifact_strategy=stack.multi_artifact_strategy,
            artifact_critic=stack.synthesis_stack.refiner_stack.critic_stack.artifact_critic,
            artifact_refiner=stack.synthesis_stack.refiner_stack.artifact_refiner,
            artifact_intelligence_synthesis=stack.artifact_intelligence_synthesis,
        )

        self.assertEqual(merge.merge_strategy, "defer_merge_preserve_separation")
        self.assertTrue(merge.dependency_merge_risks)
        self.assertTrue(merge.artifact_separation_points)

    def test_preserves_separated_artifact_paths(self) -> None:
        stack = _stack(
            "Generate separate primary code, runtime notes, and capability notes."
        )
        separated_strategy = stack.multi_artifact_strategy.model_copy(
            update={"combination_mode": "separated_parallel_sections"}
        )
        merge = derive_artifact_merge_planner_profile(
            request=stack.runtime_base.request,
            route_decision=stack.runtime_base.route,
            artifact_plan=stack.runtime_base.artifact_plan,
            artifact_dependency_graph=stack.runtime_base.artifact_dependency_graph,
            runtime_compatibility=stack.runtime_base.runtime_compatibility,
            artifact_capability_matrix=stack.artifact_capability_matrix,
            multi_artifact_strategy=separated_strategy,
            artifact_critic=stack.synthesis_stack.refiner_stack.critic_stack.artifact_critic,
            artifact_refiner=stack.synthesis_stack.refiner_stack.artifact_refiner,
            artifact_intelligence_synthesis=stack.artifact_intelligence_synthesis,
        )

        self.assertTrue(merge.artifact_separation_points)
        self.assertTrue(
            any(
                "separate" in item.lower() for item in merge.artifact_separation_points
            ),
            merge.artifact_separation_points,
        )

    def test_flags_high_risk_integration_paths(self) -> None:
        stack = _stack("Generate a high-risk merged artifact composition.")
        high_risk_critic = (
            stack.synthesis_stack.refiner_stack.critic_stack.artifact_critic.model_copy(
                update={
                    "risk_assessment": "high",
                    "hitl_questions": (
                        "Should high-risk merge planning wait for user input?",
                    ),
                }
            )
        )
        high_risk_synthesis = stack.artifact_intelligence_synthesis.model_copy(
            update={"implementation_risk": "high"}
        )
        merge = derive_artifact_merge_planner_profile(
            request=stack.runtime_base.request,
            route_decision=stack.runtime_base.route,
            artifact_plan=stack.runtime_base.artifact_plan,
            artifact_dependency_graph=stack.runtime_base.artifact_dependency_graph,
            runtime_compatibility=stack.runtime_base.runtime_compatibility,
            artifact_capability_matrix=stack.artifact_capability_matrix,
            multi_artifact_strategy=stack.multi_artifact_strategy,
            artifact_critic=high_risk_critic,
            artifact_refiner=stack.synthesis_stack.refiner_stack.artifact_refiner,
            artifact_intelligence_synthesis=high_risk_synthesis,
        )

        self.assertEqual(merge.merge_strategy, "defer_merge_preserve_separation")
        self.assertTrue(merge.composition_risks)
        self.assertTrue(merge.hitl_questions)

    def test_integrates_with_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack(
            "Generate a symbolic p5.js phoenix mandala with artifact merge "
            "planning before Director and Reasoning."
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
                "artifact_critic": (
                    stack.synthesis_stack.refiner_stack.critic_stack.artifact_critic
                ),
                "artifact_refiner": (
                    stack.synthesis_stack.refiner_stack.artifact_refiner
                ),
                "artifact_intelligence_synthesis": (
                    stack.artifact_intelligence_synthesis
                ),
                "artifact_merge_planner": stack.artifact_merge_planner,
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

        self.assertIn("Artifact Merge Planner:", system)
        self.assertIn("Merge confidence:", system)
        self.assertTrue(
            any(
                "Artifact merge planner:" in item
                for item in stack.director.planning_focus
            ),
            stack.director.planning_focus,
        )
        self.assertIn(
            "artifact_merge_planner",
            {item.source for item in stack.reasoning.evidence_chain},
        )
        self.assertEqual(
            stack.artifact_merge_planner.model_dump(mode="json")["role"],
            "artifact_merge_planner",
        )
        self.assertNotIn("artifact merge executed", system.lower())


@dataclass(frozen=True)
class _DerivedStack:
    synthesis_stack: object
    runtime_base: object
    artifact_capability_matrix: object
    multi_artifact_strategy: object
    artifact_intelligence_synthesis: object
    artifact_merge_planner: object
    director: object
    reasoning: object


def _stack(query: str) -> _DerivedStack:
    synthesis_stack = _synthesis_stack(query)
    runtime_base = synthesis_stack.runtime_base
    artifact_capability_matrix = synthesis_stack.artifact_capability_matrix
    multi_artifact_strategy = synthesis_stack.multi_artifact_strategy
    synthesis = synthesis_stack.artifact_intelligence_synthesis
    merge_planner = derive_artifact_merge_planner_profile(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        artifact_plan=runtime_base.artifact_plan,
        artifact_dependency_graph=runtime_base.artifact_dependency_graph,
        runtime_compatibility=runtime_base.runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=synthesis_stack.refiner_stack.critic_stack.artifact_critic,
        artifact_refiner=synthesis_stack.refiner_stack.artifact_refiner,
        artifact_intelligence_synthesis=synthesis,
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
        artifact_critic=synthesis_stack.refiner_stack.critic_stack.artifact_critic,
        artifact_refiner=synthesis_stack.refiner_stack.artifact_refiner,
        artifact_intelligence_synthesis=synthesis,
        artifact_merge_planner=merge_planner,
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
        artifact_critic=synthesis_stack.refiner_stack.critic_stack.artifact_critic,
        artifact_refiner=synthesis_stack.refiner_stack.artifact_refiner,
        artifact_intelligence_synthesis=synthesis,
        artifact_merge_planner=merge_planner,
    )
    return _DerivedStack(
        synthesis_stack=synthesis_stack,
        runtime_base=runtime_base,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_intelligence_synthesis=synthesis,
        artifact_merge_planner=merge_planner,
        director=director,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    unittest.main()
