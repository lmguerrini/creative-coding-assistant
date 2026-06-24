import unittest
from dataclasses import dataclass

from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    artifact_export_intelligence_prompt_lines,
    build_rendered_prompt_request,
    derive_artifact_export_intelligence_profile,
    derive_artifact_merge_planner_profile,
    derive_creative_assistant_director_brief,
    derive_creative_reasoning_result,
)
from test_artifact_merge_planner import _stack as _merge_stack


class ArtifactExportIntelligenceTests(unittest.TestCase):
    def test_derives_single_artifact_export_plan(self) -> None:
        stack = _stack("Generate a single p5.js sketch with export notes.")
        single_merge = derive_artifact_merge_planner_profile(
            request=stack.runtime_base.request,
            route_decision=stack.runtime_base.route,
            artifact_plan=stack.runtime_base.artifact_plan,
            artifact_dependency_graph=stack.runtime_base.artifact_dependency_graph,
            runtime_compatibility=stack.runtime_base.runtime_compatibility,
            artifact_capability_matrix=stack.artifact_capability_matrix,
            multi_artifact_strategy=None,
            artifact_critic=stack.merge_stack.synthesis_stack.refiner_stack.critic_stack.artifact_critic,
            artifact_refiner=stack.merge_stack.synthesis_stack.refiner_stack.artifact_refiner,
            artifact_intelligence_synthesis=stack.artifact_intelligence_synthesis,
        )
        export = derive_artifact_export_intelligence_profile(
            request=stack.runtime_base.request,
            route_decision=stack.runtime_base.route,
            artifact_plan=stack.runtime_base.artifact_plan,
            artifact_dependency_graph=stack.runtime_base.artifact_dependency_graph,
            runtime_compatibility=stack.runtime_base.runtime_compatibility,
            artifact_capability_matrix=stack.artifact_capability_matrix,
            multi_artifact_strategy=None,
            artifact_critic=stack.merge_stack.synthesis_stack.refiner_stack.critic_stack.artifact_critic,
            artifact_refiner=stack.merge_stack.synthesis_stack.refiner_stack.artifact_refiner,
            artifact_intelligence_synthesis=stack.artifact_intelligence_synthesis,
            artifact_merge_planner=single_merge,
        )

        self.assertEqual(export.role, "artifact_export_intelligence")
        self.assertIn("single_source_artifact", export.export_targets)
        self.assertTrue(export.preferred_export_target)
        self.assertTrue(export.export_format_recommendations)
        self.assertTrue(artifact_export_intelligence_prompt_lines(export))

    def test_derives_multi_artifact_export_plan(self) -> None:
        stack = _stack(
            "Generate primary p5.js code with runtime notes, capability notes, "
            "and documentation handoffs."
        )
        export = stack.artifact_export_intelligence

        self.assertEqual(export.role, "artifact_export_intelligence")
        self.assertIn("multi_artifact_package", export.export_targets)
        self.assertTrue(export.artifact_package_notes)
        self.assertTrue(export.documentation_requirements)
        self.assertGreaterEqual(export.export_confidence, 0)

    def test_derives_runtime_specific_export_notes(self) -> None:
        stack = _stack("Generate a p5.js sketch with runtime-specific export notes.")
        export = stack.artifact_export_intelligence

        self.assertTrue(
            any("p5_js" in item for item in export.runtime_export_notes),
            export.runtime_export_notes,
        )
        self.assertTrue(export.portability_notes)
        self.assertTrue(export.interoperability_notes)

    def test_rejects_export_paths_without_exporting(self) -> None:
        stack = _stack("Generate a p5.js sketch but do not export anything.")
        export = stack.artifact_export_intelligence

        self.assertTrue(export.rejected_export_paths)
        self.assertTrue(
            any("metadata-only" in item for item in export.rejected_export_paths),
            export.rejected_export_paths,
        )
        self.assertNotIn("exported file", " ".join(export.prompt_guidance).lower())

    def test_blocks_missing_information_cases(self) -> None:
        stack = _stack("Generate an artifact with export metadata gaps.")
        export = derive_artifact_export_intelligence_profile(
            request=stack.runtime_base.request,
            route_decision=stack.runtime_base.route,
            artifact_plan=None,
            artifact_dependency_graph=None,
            runtime_compatibility=None,
            artifact_capability_matrix=None,
            multi_artifact_strategy=None,
            artifact_critic=None,
            artifact_refiner=None,
            artifact_intelligence_synthesis=None,
            artifact_merge_planner=None,
        )

        self.assertEqual(export.export_readiness, "blocked_by_missing_metadata")
        self.assertIn(
            "defer_export_until_metadata_complete",
            export.preferred_export_target,
        )
        self.assertTrue(export.hitl_questions)
        self.assertTrue(export.export_risks)

    def test_derives_downstream_tool_handoffs(self) -> None:
        stack = _stack("Generate a p5.js sketch with downstream handoff notes.")
        export = stack.artifact_export_intelligence

        self.assertTrue(export.downstream_tool_handoffs)
        self.assertTrue(
            any(
                "workflow does not trigger export" in item
                for item in export.downstream_tool_handoffs
            ),
            export.downstream_tool_handoffs,
        )

    def test_integrates_with_prompt_director_reasoning_and_serialization(self) -> None:
        stack = _stack(
            "Generate a symbolic p5.js phoenix mandala with export intelligence "
            "before Director and Reasoning."
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
                    stack.merge_stack.synthesis_stack.refiner_stack.critic_stack.artifact_critic
                ),
                "artifact_refiner": (
                    stack.merge_stack.synthesis_stack.refiner_stack.artifact_refiner
                ),
                "artifact_intelligence_synthesis": (
                    stack.artifact_intelligence_synthesis
                ),
                "artifact_merge_planner": stack.artifact_merge_planner,
                "artifact_export_intelligence": stack.artifact_export_intelligence,
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

        self.assertIn("Artifact Export Intelligence:", system)
        self.assertIn("Export readiness:", system)
        self.assertTrue(
            any(
                "Artifact export intelligence:" in item
                for item in stack.director.planning_focus
            ),
            stack.director.planning_focus,
        )
        self.assertIn(
            "artifact_export_intelligence",
            {item.source for item in stack.reasoning.evidence_chain},
        )
        self.assertEqual(
            stack.artifact_export_intelligence.model_dump(mode="json")["role"],
            "artifact_export_intelligence",
        )
        self.assertNotIn("exported file", system.lower())


@dataclass(frozen=True)
class _DerivedStack:
    merge_stack: object
    runtime_base: object
    artifact_capability_matrix: object
    multi_artifact_strategy: object
    artifact_intelligence_synthesis: object
    artifact_merge_planner: object
    artifact_export_intelligence: object
    director: object
    reasoning: object


def _stack(query: str) -> _DerivedStack:
    merge_stack = _merge_stack(query)
    runtime_base = merge_stack.runtime_base
    artifact_capability_matrix = merge_stack.artifact_capability_matrix
    multi_artifact_strategy = merge_stack.multi_artifact_strategy
    synthesis = merge_stack.artifact_intelligence_synthesis
    merge_planner = merge_stack.artifact_merge_planner
    artifact_critic = (
        merge_stack.synthesis_stack.refiner_stack.critic_stack.artifact_critic
    )
    artifact_refiner = merge_stack.synthesis_stack.refiner_stack.artifact_refiner
    export_intelligence = derive_artifact_export_intelligence_profile(
        request=runtime_base.request,
        route_decision=runtime_base.route,
        artifact_plan=runtime_base.artifact_plan,
        artifact_dependency_graph=runtime_base.artifact_dependency_graph,
        runtime_compatibility=runtime_base.runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
        artifact_intelligence_synthesis=synthesis,
        artifact_merge_planner=merge_planner,
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
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
        artifact_intelligence_synthesis=synthesis,
        artifact_merge_planner=merge_planner,
        artifact_export_intelligence=export_intelligence,
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
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
        artifact_intelligence_synthesis=synthesis,
        artifact_merge_planner=merge_planner,
        artifact_export_intelligence=export_intelligence,
    )
    return _DerivedStack(
        merge_stack=merge_stack,
        runtime_base=runtime_base,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_intelligence_synthesis=synthesis,
        artifact_merge_planner=merge_planner,
        artifact_export_intelligence=export_intelligence,
        director=director,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    unittest.main()
