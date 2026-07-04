import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    CreativeComplexityAnalysis,
    analyze_creative_complexity,
    creative_complexity_factor_by_id,
    creative_complexity_factors_for_kind,
    derive_creative_constraint_solution,
    derive_creative_execution_plan,
    derive_creative_hierarchy_plan,
    derive_creative_intent_decomposition,
    derive_creative_strategy_profile,
    derive_creative_technique_profile,
    derive_creative_tradeoff_profile,
    derive_creative_translation,
    derive_runtime_capability_profile,
    route_request,
)

REQUIRED_FACTOR_FIELDS = {
    "factor_id",
    "factor_kind",
    "source_id",
    "level",
    "score",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "prompt_rewrite_implemented",
    "creative_output_mutation_implemented",
    "runtime_selection_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "artifact_execution_implemented",
    "preview_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "analysis_only",
}


class CreativeComplexityAnalyzerTests(unittest.TestCase):
    def test_default_analysis_reports_low_input_coverage(self) -> None:
        analysis = analyze_creative_complexity()

        self.assertEqual(analysis.role, "creative_complexity_analyzer")
        self.assertEqual(
            analysis.serialization_version,
            "creative_complexity_analysis.v1",
        )
        self.assertEqual(analysis.source_metadata_roles, ())
        self.assertEqual(analysis.factor_ids, ("factor::input_coverage",))
        self.assertEqual(analysis.creative_complexity_score, 1)
        self.assertEqual(analysis.creative_complexity_level, "low")
        self.assertFalse(analysis.hitl_advisable)
        self.assertTrue(analysis.creative_complexity_analysis_implemented)
        self.assertFalse(analysis.prompt_rewrite_implemented)
        self.assertFalse(analysis.creative_output_mutation_implemented)
        self.assertFalse(analysis.runtime_selection_implemented)
        self.assertFalse(analysis.provider_model_routing_implemented)
        self.assertFalse(analysis.workflow_control_implemented)
        self.assertFalse(analysis.retry_triggering_implemented)
        self.assertFalse(analysis.artifact_execution_implemented)
        self.assertFalse(analysis.preview_mutation_implemented)
        self.assertFalse(analysis.persistent_storage_write_implemented)
        self.assertFalse(analysis.generated_output_mutation_implemented)
        self.assertTrue(analysis.analysis_only)

    def test_full_analysis_covers_creative_pressure_factors(self) -> None:
        metadata = _rich_creative_metadata()
        analysis = analyze_creative_complexity(**metadata)

        self.assertEqual(
            analysis.factor_ids,
            (
                "factor::translation_surface",
                "factor::intent_density",
                "factor::hierarchy_pressure",
                "factor::plan_shape",
                "factor::technique_pressure",
                "factor::constraint_pressure",
                "factor::runtime_pressure",
                "factor::tradeoff_risk",
            ),
        )
        self.assertIn("creative_translation", analysis.source_metadata_roles)
        self.assertIn("creative_intent_decomposer", analysis.source_metadata_roles)
        self.assertIn("creative_hierarchy_planner", analysis.source_metadata_roles)
        self.assertIn("creative_technique_selector", analysis.source_metadata_roles)
        self.assertIn("runtime_capability_reasoner", analysis.source_metadata_roles)
        self.assertGreater(analysis.active_intent_dimension_count, 0)
        self.assertGreater(analysis.runtime_candidate_count, 0)
        self.assertGreaterEqual(analysis.tradeoff_risk_count, 0)
        self.assertEqual(
            analysis.creative_complexity_score,
            sum(factor.score for factor in analysis.factors),
        )
        self.assertIn(analysis.creative_complexity_level, {"medium", "high"})
        self.assertIn(
            "does not rewrite prompts",
            analysis.authority_boundary,
        )

        for factor in analysis.factors:
            self.assertEqual(
                set(factor.model_dump(mode="json")), REQUIRED_FACTOR_FIELDS
            )
            self.assertEqual(
                factor.serialization_version,
                "creative_complexity_factor.v1",
            )
            self.assertIn("creative_output_mutation", factor.blocked_runtime_behaviors)
            self.assertFalse(factor.prompt_rewrite_implemented)
            self.assertFalse(factor.creative_output_mutation_implemented)
            self.assertFalse(factor.runtime_selection_implemented)
            self.assertFalse(factor.provider_model_routing_implemented)
            self.assertFalse(factor.workflow_control_implemented)
            self.assertFalse(factor.retry_triggering_implemented)
            self.assertFalse(factor.artifact_execution_implemented)
            self.assertFalse(factor.preview_mutation_implemented)
            self.assertFalse(factor.generated_output_mutation_implemented)
            self.assertTrue(factor.analysis_only)

    def test_factor_helpers_return_stable_read_only_views(self) -> None:
        analysis = analyze_creative_complexity(**_rich_creative_metadata())
        intent = creative_complexity_factor_by_id("factor::intent_density", analysis)
        runtime_factors = creative_complexity_factors_for_kind(
            "runtime_pressure",
            analysis,
        )
        missing = creative_complexity_factor_by_id("missing", analysis)

        self.assertIsNone(missing)
        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.factor_kind, "intent_density")
        self.assertEqual(len(runtime_factors), 1)
        self.assertEqual(runtime_factors[0].factor_id, "factor::runtime_pressure")
        self.assertIs(
            intent,
            creative_complexity_factor_by_id("factor::intent_density", analysis),
        )

    def test_analysis_rejects_mismatched_factors_or_scores(self) -> None:
        analysis = analyze_creative_complexity(**_rich_creative_metadata())
        payload = analysis.model_dump(mode="json")
        payload["factor_ids"] = ("missing",) + tuple(payload["factor_ids"][1:])

        with self.assertRaisesRegex(ValueError, "factor_ids must match"):
            CreativeComplexityAnalysis(**payload)

        payload = analysis.model_dump(mode="json")
        payload["creative_complexity_score"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "creative_complexity_score must match",
        ):
            CreativeComplexityAnalysis(**payload)

    def test_analysis_does_not_declare_output_or_routing_mutation_terms(self) -> None:
        analysis = analyze_creative_complexity(**_rich_creative_metadata())
        combined_text = " ".join(
            (
                analysis.authority_boundary,
                *analysis.blocked_runtime_behaviors,
                *analysis.advisory_actions,
                *(
                    field
                    for factor in analysis.factors
                    for field in (
                        factor.factor_id,
                        factor.source_id,
                        *factor.evidence,
                        *factor.advisory_actions,
                        *factor.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "rewrite_prompt(",
            "mutate_creative_output(",
            "select_runtime(",
            "route_provider(",
            "control_workflow(",
            "trigger_retry(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


def _rich_creative_metadata() -> dict[str, object]:
    request = AssistantRequest(
        query=(
            "Generate intertwined sacred geometry, audio reactive motion, "
            "fractal recursion, luminous color shifts, and three alternate "
            "p5.js compositions with export-ready structure."
        ),
        mode=AssistantMode.GENERATE,
        domain=CreativeCodingDomain.P5_JS,
    )
    route_decision = route_request(request)
    creative_translation = derive_creative_translation(
        request.query,
        domains=(CreativeCodingDomain.P5_JS,),
    )
    creative_intent = derive_creative_intent_decomposition(
        request=request,
        route_decision=route_decision,
        creative_translation=creative_translation,
    )
    creative_hierarchy = derive_creative_hierarchy_plan(
        request=request,
        route_decision=route_decision,
        creative_intent=creative_intent,
        creative_translation=creative_translation,
    )
    creative_strategy = derive_creative_strategy_profile(
        request=request,
        route_decision=route_decision,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_translation=creative_translation,
    )
    creative_techniques = derive_creative_technique_profile(
        request=request,
        route_decision=route_decision,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_translation=creative_translation,
        creative_strategy=creative_strategy,
    )
    creative_plan = derive_creative_execution_plan(
        request=request,
        route_decision=route_decision,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_translation=creative_translation,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        retrieval_chunk_count=5,
    )
    creative_constraints = derive_creative_constraint_solution(
        request=request,
        route_decision=route_decision,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_translation=creative_translation,
        creative_plan=creative_plan,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        retrieval_chunk_count=5,
    )
    runtime_capabilities = derive_runtime_capability_profile(
        request=request,
        route_decision=route_decision,
        creative_translation=creative_translation,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
    )
    creative_tradeoffs = derive_creative_tradeoff_profile(
        request=request,
        route_decision=route_decision,
        creative_translation=creative_translation,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
        runtime_capabilities=runtime_capabilities,
    )
    return {
        "creative_translation": creative_translation,
        "creative_intent": creative_intent,
        "creative_hierarchy": creative_hierarchy,
        "creative_plan": creative_plan,
        "creative_techniques": creative_techniques,
        "creative_constraints": creative_constraints,
        "runtime_capabilities": runtime_capabilities,
        "creative_tradeoffs": creative_tradeoffs,
    }


if __name__ == "__main__":
    unittest.main()
