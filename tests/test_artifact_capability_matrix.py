import unittest
from dataclasses import dataclass

from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    artifact_capability_matrix_prompt_lines,
    build_rendered_prompt_request,
    derive_artifact_capability_matrix,
    derive_creative_assistant_director_brief,
    derive_creative_reasoning_result,
)
from test_runtime_compatibility import _stack as _runtime_stack


class ArtifactCapabilityMatrixTests(unittest.TestCase):
    def test_derives_artifact_capability_matrix_metadata(self) -> None:
        stack = _stack(
            "Generate a p5.js recursive mandala with bounded particles, "
            "clear setup/draw structure, and explicit capability trade-offs."
        )
        matrix = stack.artifact_capability_matrix

        self.assertEqual(matrix.role, "artifact_capability_matrix")
        self.assertTrue(matrix.capability_profiles)
        self.assertIn("p5_js", matrix.strongest_targets)
        self.assertTrue(matrix.weakest_targets)
        self.assertTrue(matrix.target_strengths)
        self.assertTrue(matrix.target_weaknesses)
        self.assertTrue(matrix.unsupported_or_risky_capabilities)
        self.assertTrue(matrix.capability_confidence)
        self.assertIn(matrix.artifact_fit, {"strong", "moderate", "weak"})
        self.assertIn(matrix.creative_fit, {"strong", "moderate", "weak"})
        self.assertIn(matrix.generative_fit, {"strong", "moderate", "weak"})
        self.assertTrue(matrix.prompt_guidance)
        self.assertIn("does not auto-select runtimes", matrix.authority_boundary)
        self.assertTrue(artifact_capability_matrix_prompt_lines(matrix))

        p5_profile = next(
            profile
            for profile in matrix.capability_profiles
            if profile.target == "p5_js"
        )
        self.assertEqual(p5_profile.artifact_fit, "strong")
        self.assertTrue(p5_profile.strengths)
        self.assertTrue(p5_profile.weaknesses)
        self.assertTrue(p5_profile.capability_reasons)

    def test_integrates_with_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack(
            "Generate a symbolic p5.js phoenix mandala with target capability "
            "matrix notes and dependency-aware runtime guidance."
        )
        prompt_input = stack.base.prompt_input.model_copy(
            update={
                "creative_intent": stack.base.intent,
                "creative_hierarchy": stack.base.hierarchy,
                "creative_strategy": stack.base.strategy,
                "creative_techniques": stack.base.techniques,
                "creative_plan": stack.base.plan,
                "creative_constraints": stack.base.constraints,
                "creative_constraint_priorities": stack.base.prioritization,
                "runtime_capabilities": stack.base.runtime_capabilities,
                "creative_tradeoffs": stack.base.tradeoffs,
                "artifact_plan": stack.base.artifact_plan,
                "artifact_dependency_graph": stack.base.artifact_dependency_graph,
                "runtime_compatibility": stack.base.runtime_compatibility,
                "artifact_capability_matrix": stack.artifact_capability_matrix,
                "creative_director": stack.director,
                "creative_reasoning": stack.reasoning,
            }
        )

        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=stack.base.route,
                prompt_input=prompt_input,
            )
        )
        system = rendered.sections[0].content

        self.assertIn("Artifact Capability Matrix:", system)
        self.assertIn("Capability profile:", system)
        self.assertTrue(
            any(
                "Artifact capability matrix:" in item
                for item in stack.director.planning_focus
            ),
            stack.director.planning_focus,
        )
        self.assertIn(
            "artifact_capability_matrix",
            {item.source for item in stack.reasoning.evidence_chain},
        )
        self.assertIn(
            "Capabilities as",
            stack.reasoning.recommended_creative_direction,
        )
        self.assertEqual(
            stack.artifact_capability_matrix.model_dump(mode="json")["role"],
            "artifact_capability_matrix",
        )
        self.assertNotIn("runtime auto-selection enabled", system.lower())
        self.assertNotIn("provider routing enabled", system.lower())


@dataclass(frozen=True)
class _DerivedStack:
    base: object
    artifact_capability_matrix: object
    director: object
    reasoning: object


def _stack(query: str) -> _DerivedStack:
    base = _runtime_stack(query)
    matrix = derive_artifact_capability_matrix(
        request=base.request,
        route_decision=base.route,
        artifact_plan=base.artifact_plan,
        artifact_dependency_graph=base.artifact_dependency_graph,
        runtime_capabilities=base.runtime_capabilities,
        runtime_compatibility=base.runtime_compatibility,
        creative_plan=base.plan,
        creative_constraints=base.constraints,
        creative_strategy=base.strategy,
        creative_techniques=base.techniques,
        creative_tradeoffs=base.tradeoffs,
    )
    director = derive_creative_assistant_director_brief(
        request=base.request,
        route_decision=base.route,
        creative_translation=base.prompt_input.creative_translation,
        creative_intent=base.intent,
        creative_hierarchy=base.hierarchy,
        creative_strategy=base.strategy,
        creative_techniques=base.techniques,
        creative_plan=base.plan,
        creative_constraints=base.constraints,
        creative_constraint_priorities=base.prioritization,
        runtime_capabilities=base.runtime_capabilities,
        creative_tradeoffs=base.tradeoffs,
        artifact_plan=base.artifact_plan,
        artifact_dependency_graph=base.artifact_dependency_graph,
        runtime_compatibility=base.runtime_compatibility,
        artifact_capability_matrix=matrix,
    )
    reasoning = derive_creative_reasoning_result(
        request=base.request,
        route_decision=base.route,
        creative_translation=base.prompt_input.creative_translation,
        creative_intent=base.intent,
        creative_hierarchy=base.hierarchy,
        creative_plan=base.plan,
        creative_director=director,
        creative_constraints=base.constraints,
        creative_constraint_priorities=base.prioritization,
        creative_strategy=base.strategy,
        creative_techniques=base.techniques,
        runtime_capabilities=base.runtime_capabilities,
        creative_tradeoffs=base.tradeoffs,
        artifact_plan=base.artifact_plan,
        artifact_dependency_graph=base.artifact_dependency_graph,
        runtime_compatibility=base.runtime_compatibility,
        artifact_capability_matrix=matrix,
    )
    return _DerivedStack(
        base=base,
        artifact_capability_matrix=matrix,
        director=director,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    unittest.main()
