import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    WorkflowComplexityAnalysis,
    analyze_assistant_execution_graph,
    analyze_workflow_complexity,
    analyze_workflow_cost,
    derive_creative_execution_plan,
    route_request,
    workflow_complexity_factor_by_id,
    workflow_complexity_factors_for_kind,
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
    "creative_semantic_scoring_implemented",
    "workflow_pruning_implemented",
    "execution_path_selection_implemented",
    "strategy_selection_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "analysis_only",
}


class WorkflowComplexityAnalyzerTests(unittest.TestCase):
    def test_default_analysis_derives_structural_complexity(self) -> None:
        graph = analyze_assistant_execution_graph()
        costs = analyze_workflow_cost(execution_graph=graph)
        analysis = analyze_workflow_complexity(
            execution_graph=graph,
            cost_analysis=costs,
        )

        self.assertEqual(analysis.role, "workflow_complexity_analyzer")
        self.assertEqual(
            analysis.serialization_version,
            "workflow_complexity_analysis.v1",
        )
        self.assertEqual(
            analysis.source_graph_serialization_version,
            graph.serialization_version,
        )
        self.assertEqual(
            analysis.source_cost_serialization_version,
            costs.serialization_version,
        )
        self.assertEqual(analysis.node_count, graph.node_count)
        self.assertEqual(analysis.edge_count, graph.edge_count)
        self.assertEqual(analysis.branch_count, graph.branch_count)
        self.assertEqual(analysis.failure_edge_count, graph.failure_edge_count)
        self.assertEqual(
            analysis.critical_path_length,
            len(graph.critical_path_node_ids),
        )
        self.assertTrue(analysis.retry_cycle_present)
        self.assertEqual(analysis.cost_pressure, costs.estimated_cost_pressure)
        self.assertEqual(
            analysis.structural_complexity_score,
            sum(factor.score for factor in analysis.factors),
        )
        self.assertEqual(analysis.complexity_level, "high")
        self.assertIn(
            "does not evaluate creative semantics", analysis.authority_boundary
        )
        self.assertTrue(analysis.complexity_analysis_implemented)
        self.assertFalse(analysis.creative_semantic_scoring_implemented)
        self.assertFalse(analysis.workflow_pruning_implemented)
        self.assertFalse(analysis.execution_path_selection_implemented)
        self.assertFalse(analysis.strategy_selection_implemented)
        self.assertFalse(analysis.provider_model_routing_implemented)
        self.assertFalse(analysis.workflow_control_implemented)
        self.assertFalse(analysis.retry_triggering_implemented)
        self.assertFalse(analysis.prompt_mutation_implemented)
        self.assertFalse(analysis.persistent_storage_write_implemented)
        self.assertFalse(analysis.generated_output_mutation_implemented)
        self.assertTrue(analysis.analysis_only)

    def test_factors_cover_topology_branching_retry_failure_and_cost(self) -> None:
        analysis = analyze_workflow_complexity()

        self.assertEqual(
            analysis.factor_ids,
            (
                "factor::topology",
                "factor::branching",
                "factor::retry",
                "factor::failure_path",
                "factor::cost_pressure",
            ),
        )
        for factor in analysis.factors:
            self.assertEqual(
                set(factor.model_dump(mode="json")), REQUIRED_FACTOR_FIELDS
            )
            self.assertEqual(
                factor.serialization_version,
                "workflow_complexity_factor.v1",
            )
            self.assertIn("workflow_pruning", factor.blocked_runtime_behaviors)
            self.assertFalse(factor.creative_semantic_scoring_implemented)
            self.assertFalse(factor.workflow_pruning_implemented)
            self.assertFalse(factor.execution_path_selection_implemented)
            self.assertFalse(factor.strategy_selection_implemented)
            self.assertFalse(factor.provider_model_routing_implemented)
            self.assertFalse(factor.workflow_control_implemented)
            self.assertFalse(factor.retry_triggering_implemented)
            self.assertFalse(factor.prompt_mutation_implemented)
            self.assertFalse(factor.generated_output_mutation_implemented)
            self.assertTrue(factor.analysis_only)

        topology = workflow_complexity_factor_by_id("factor::topology", analysis)
        branching = workflow_complexity_factor_by_id("factor::branching", analysis)
        retry = workflow_complexity_factor_by_id("factor::retry", analysis)
        cost = workflow_complexity_factor_by_id("factor::cost_pressure", analysis)
        self.assertIsNotNone(topology)
        self.assertIsNotNone(branching)
        self.assertIsNotNone(retry)
        self.assertIsNotNone(cost)
        assert topology is not None
        assert branching is not None
        assert retry is not None
        assert cost is not None
        self.assertEqual(topology.factor_kind, "topology")
        self.assertEqual(branching.factor_kind, "branching")
        self.assertEqual(retry.factor_kind, "retry")
        self.assertEqual(cost.factor_kind, "cost_pressure")
        self.assertGreater(branching.score, retry.score)

    def test_creative_plan_adds_plan_shape_without_creative_semantic_scoring(
        self,
    ) -> None:
        request = AssistantRequest(
            query=(
                "Generate three intricate audio reactive p5.js variations with "
                "layered sacred geometry and export-ready structure."
            ),
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        decision = route_request(request)
        plan = derive_creative_execution_plan(
            request=request,
            route_decision=decision,
            retrieval_chunk_count=5,
        )
        analysis = analyze_workflow_complexity(creative_plan=plan)
        plan_factors = workflow_complexity_factors_for_kind("plan_shape", analysis)
        plan_factor = workflow_complexity_factor_by_id("factor::plan_shape", analysis)

        self.assertEqual(len(plan_factors), 1)
        self.assertIs(plan_factors[0], plan_factor)
        self.assertIsNotNone(plan_factor)
        assert plan_factor is not None
        self.assertEqual(plan_factor.source_id, "creative_execution_plan")
        self.assertEqual(analysis.plan_shape_complexity, plan_factor.level)
        self.assertIn(
            f"candidate_count:{plan.candidate_count}",
            plan_factor.evidence,
        )
        self.assertFalse(plan_factor.creative_semantic_scoring_implemented)
        self.assertIn("factor::plan_shape", analysis.factor_ids)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        analysis = analyze_workflow_complexity()
        missing = workflow_complexity_factor_by_id("missing", analysis)
        retry_factors = workflow_complexity_factors_for_kind("retry", analysis)
        plan_factors = workflow_complexity_factors_for_kind("plan_shape", analysis)

        self.assertIsNone(missing)
        self.assertEqual(len(retry_factors), 1)
        self.assertEqual(retry_factors[0].factor_id, "factor::retry")
        self.assertEqual(plan_factors, ())
        self.assertIs(
            retry_factors[0],
            workflow_complexity_factor_by_id("factor::retry", analysis),
        )

    def test_analysis_rejects_mismatched_factors_or_scores(self) -> None:
        analysis = analyze_workflow_complexity()
        payload = analysis.model_dump(mode="json")
        payload["factor_ids"] = ("missing",) + tuple(payload["factor_ids"][1:])

        with self.assertRaisesRegex(ValueError, "factor_ids must match"):
            WorkflowComplexityAnalysis(**payload)

        payload = analysis.model_dump(mode="json")
        payload["structural_complexity_score"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "structural_complexity_score must match",
        ):
            WorkflowComplexityAnalysis(**payload)

    def test_analysis_does_not_declare_runtime_control_terms(self) -> None:
        analysis = analyze_workflow_complexity()
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
            "score_creative_semantics(",
            "prune_workflow(",
            "select_execution_path(",
            "select_strategy(",
            "route_provider(",
            "trigger_retry(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
