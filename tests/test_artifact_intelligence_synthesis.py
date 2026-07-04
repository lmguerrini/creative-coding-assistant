import unittest
from dataclasses import dataclass

from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    artifact_intelligence_synthesis_prompt_lines,
    build_rendered_prompt_request,
    derive_artifact_intelligence_synthesis_profile,
    derive_creative_assistant_director_brief,
    derive_creative_reasoning_result,
)
from test_artifact_refiner import _stack as _refiner_stack


class ArtifactIntelligenceSynthesisTests(unittest.TestCase):
    def test_derives_synthesis_for_strong_plans(self) -> None:
        stack = _stack(
            "Generate a p5.js recursive mandala with artifact planning, "
            "dependency graph, runtime compatibility, capability matrix, "
            "critic metadata, and refiner metadata."
        )
        synthesis = stack.artifact_intelligence_synthesis

        self.assertEqual(synthesis.role, "artifact_intelligence_synthesis")
        self.assertGreater(synthesis.synthesis_confidence, 0)
        self.assertTrue(synthesis.synthesis_summary)
        self.assertTrue(synthesis.recommended_artifact_path)
        self.assertTrue(synthesis.recommended_strategy_summary)
        self.assertTrue(synthesis.recommended_runtime_direction)
        self.assertTrue(synthesis.major_strengths)
        self.assertTrue(synthesis.prompt_guidance)
        self.assertIn("does not modify artifacts", synthesis.authority_boundary)
        self.assertTrue(artifact_intelligence_synthesis_prompt_lines(synthesis))

    def test_flags_weak_plans(self) -> None:
        stack = _stack(
            "Generate a dense high-complexity p5.js particle artifact with "
            "performance-sensitive caveats."
        )
        weak_critic = stack.refiner_stack.critic_stack.artifact_critic.model_copy(
            update={
                "risk_assessment": "high",
                "weaknesses": (
                    "Dense particle counts can pressure frame rate.",
                    "Complex scope can reduce inspectability.",
                ),
                "complexity_concerns": ("Complex scope can reduce inspectability.",),
                "hitl_questions": (
                    "Should generation wait for the high-risk artifact scope?",
                ),
            }
        )
        weak_refiner = stack.refiner_stack.artifact_refiner.model_copy(
            update={
                "priority_improvements": (
                    "Reduce dense particle scope before generation.",
                ),
                "complexity_reductions": (
                    "Reduce implementation scope before optional details.",
                ),
            }
        )
        synthesis = derive_artifact_intelligence_synthesis_profile(
            request=stack.runtime_base.request,
            route_decision=stack.runtime_base.route,
            artifact_plan=stack.runtime_base.artifact_plan,
            artifact_dependency_graph=stack.runtime_base.artifact_dependency_graph,
            runtime_compatibility=stack.runtime_base.runtime_compatibility,
            artifact_capability_matrix=stack.artifact_capability_matrix,
            multi_artifact_strategy=stack.multi_artifact_strategy,
            artifact_critic=weak_critic,
            artifact_refiner=weak_refiner,
        )

        self.assertIn(synthesis.implementation_readiness, {"needs_hitl", "blocked"})
        self.assertIn(synthesis.implementation_risk, {"high", "blocked"})
        self.assertTrue(synthesis.major_weaknesses)
        self.assertTrue(synthesis.hitl_questions)

    def test_flags_conflicting_plans(self) -> None:
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
        critic = stack.refiner_stack.critic_stack.artifact_critic
        conflicting_critic = critic.model_copy(
            update={
                "risk_assessment": "high",
                "dependency_concerns": (
                    "Runtime-facing dependency conflicts with output structure.",
                ),
            }
        )
        synthesis = derive_artifact_intelligence_synthesis_profile(
            request=stack.runtime_base.request,
            route_decision=stack.runtime_base.route,
            artifact_plan=stack.runtime_base.artifact_plan,
            artifact_dependency_graph=conflicting_graph,
            runtime_compatibility=stack.runtime_base.runtime_compatibility,
            artifact_capability_matrix=stack.artifact_capability_matrix,
            multi_artifact_strategy=stack.multi_artifact_strategy,
            artifact_critic=conflicting_critic,
            artifact_refiner=stack.refiner_stack.artifact_refiner,
        )

        self.assertIn("1 blocking", synthesis.dependency_overview)
        self.assertIn("1 conflicts", synthesis.dependency_overview)
        self.assertTrue(
            any("conflict" in item.lower() for item in synthesis.major_risks),
            synthesis.major_risks,
        )
        self.assertIn(synthesis.implementation_priority, {"critical", "high"})

    def test_flags_blocked_high_risk_plans(self) -> None:
        stack = _stack("Generate an artifact with blocked dependency and runtime risk.")
        blocked_critic = stack.refiner_stack.critic_stack.artifact_critic.model_copy(
            update={
                "risk_assessment": "blocked",
                "hitl_questions": (
                    "Should blocked artifact intelligence halt generation?",
                ),
            }
        )
        synthesis = derive_artifact_intelligence_synthesis_profile(
            request=stack.runtime_base.request,
            route_decision=stack.runtime_base.route,
            artifact_plan=stack.runtime_base.artifact_plan,
            artifact_dependency_graph=stack.runtime_base.artifact_dependency_graph,
            runtime_compatibility=stack.runtime_base.runtime_compatibility,
            artifact_capability_matrix=stack.artifact_capability_matrix,
            multi_artifact_strategy=stack.multi_artifact_strategy,
            artifact_critic=blocked_critic,
            artifact_refiner=stack.refiner_stack.artifact_refiner,
        )

        self.assertEqual(synthesis.implementation_readiness, "blocked")
        self.assertEqual(synthesis.implementation_risk, "blocked")
        self.assertEqual(synthesis.implementation_priority, "critical")
        self.assertTrue(synthesis.hitl_questions)

    def test_summarizes_multi_artifact_paths(self) -> None:
        stack = _stack(
            "Generate primary p5.js code with runtime notes, dependency notes, "
            "and capability caveats as separated supporting artifacts."
        )
        synthesis = stack.artifact_intelligence_synthesis

        self.assertIn("primary_artifact", synthesis.recommended_artifact_path)
        self.assertIn("supporting", synthesis.recommended_artifact_path)
        self.assertIn("ordered steps", synthesis.recommended_strategy_summary)

    def test_integrates_with_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack(
            "Generate a symbolic p5.js phoenix mandala with artifact intelligence "
            "synthesis before Director and Reasoning."
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
                "artifact_critic": stack.refiner_stack.critic_stack.artifact_critic,
                "artifact_refiner": stack.refiner_stack.artifact_refiner,
                "artifact_intelligence_synthesis": (
                    stack.artifact_intelligence_synthesis
                ),
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

        self.assertIn("Artifact Intelligence Synthesis:", system)
        self.assertIn("Synthesis confidence:", system)
        self.assertTrue(
            any(
                "Artifact intelligence synthesis:" in item
                for item in stack.director.planning_focus
            ),
            stack.director.planning_focus,
        )
        self.assertIn(
            "artifact_intelligence_synthesis",
            {item.source for item in stack.reasoning.evidence_chain},
        )
        self.assertEqual(
            stack.artifact_intelligence_synthesis.model_dump(mode="json")["role"],
            "artifact_intelligence_synthesis",
        )
        self.assertNotIn("runtime auto-selection enabled", system.lower())
        self.assertNotIn("provider routing enabled", system.lower())


@dataclass(frozen=True)
class _DerivedStack:
    refiner_stack: object
    runtime_base: object
    artifact_capability_matrix: object
    multi_artifact_strategy: object
    artifact_intelligence_synthesis: object
    director: object
    reasoning: object


def _stack(query: str) -> _DerivedStack:
    refiner_stack = _refiner_stack(query)
    runtime_base = refiner_stack.runtime_base
    artifact_capability_matrix = refiner_stack.artifact_capability_matrix
    multi_artifact_strategy = refiner_stack.multi_artifact_strategy
    synthesis = derive_artifact_intelligence_synthesis_profile(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        artifact_plan=runtime_base.artifact_plan,
        artifact_dependency_graph=runtime_base.artifact_dependency_graph,
        runtime_compatibility=runtime_base.runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=refiner_stack.critic_stack.artifact_critic,
        artifact_refiner=refiner_stack.artifact_refiner,
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
        artifact_critic=refiner_stack.critic_stack.artifact_critic,
        artifact_refiner=refiner_stack.artifact_refiner,
        artifact_intelligence_synthesis=synthesis,
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
        artifact_critic=refiner_stack.critic_stack.artifact_critic,
        artifact_refiner=refiner_stack.artifact_refiner,
        artifact_intelligence_synthesis=synthesis,
    )
    return _DerivedStack(
        refiner_stack=refiner_stack,
        runtime_base=runtime_base,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_intelligence_synthesis=synthesis,
        director=director,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    unittest.main()
