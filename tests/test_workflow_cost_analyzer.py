import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    MAX_WORKFLOW_REFINEMENT_COUNT,
    WorkflowCostAnalysis,
    analyze_workflow_cost,
    derive_creative_execution_plan,
    route_request,
    workflow_cost_component_by_id,
    workflow_cost_components_for_kind,
)

REQUIRED_COMPONENT_FIELDS = {
    "component_id",
    "component_kind",
    "source_id",
    "relative_cost",
    "estimated_token_cost",
    "cost_weight",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "pricing_lookup_implemented",
    "budget_enforcement_implemented",
    "cost_based_routing_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "analysis_only",
}

EXPECTED_SOURCE_REGISTRIES = (
    "execution_graph_analysis",
    "agent_cost_tracking_foundation_registry",
    "artifact_intelligence_engine_contract_registry",
    "creative_execution_plan",
)


class WorkflowCostAnalyzerTests(unittest.TestCase):
    def test_default_analysis_derives_bounded_cost_envelope(self) -> None:
        analysis = analyze_workflow_cost()

        self.assertEqual(analysis.role, "workflow_cost_analyzer")
        self.assertEqual(analysis.serialization_version, "workflow_cost_analysis.v1")
        self.assertEqual(
            analysis.source_cost_registries,
            EXPECTED_SOURCE_REGISTRIES,
        )
        self.assertEqual(
            analysis.source_graph_serialization_version,
            "execution_graph_analysis.v1",
        )
        self.assertEqual(analysis.node_count, len(ASSISTANT_WORKFLOW_NODE_ORDER))
        self.assertEqual(analysis.component_count, len(analysis.components))
        self.assertEqual(analysis.cost_source_count, 2)
        self.assertGreater(analysis.critical_path_token_estimate, 0)
        self.assertGreater(analysis.retry_iteration_token_estimate, 0)
        self.assertEqual(
            analysis.retry_iteration_count,
            MAX_WORKFLOW_REFINEMENT_COUNT,
        )
        self.assertEqual(
            analysis.retry_token_reserve,
            analysis.retry_iteration_token_estimate * analysis.retry_iteration_count,
        )
        self.assertEqual(
            analysis.worst_case_token_estimate,
            analysis.critical_path_token_estimate
            + analysis.retry_token_reserve
            + analysis.failure_path_token_reserve,
        )
        self.assertIn(analysis.estimated_cost_pressure, {"low", "medium", "high"})
        self.assertIn("does not look up provider pricing", analysis.authority_boundary)
        self.assertTrue(analysis.cost_analysis_implemented)
        self.assertFalse(analysis.pricing_lookup_implemented)
        self.assertFalse(analysis.live_usage_metering_implemented)
        self.assertFalse(analysis.budget_enforcement_implemented)
        self.assertFalse(analysis.cost_based_routing_implemented)
        self.assertFalse(analysis.provider_model_routing_implemented)
        self.assertFalse(analysis.workflow_control_implemented)
        self.assertFalse(analysis.retry_triggering_implemented)
        self.assertFalse(analysis.prompt_mutation_implemented)
        self.assertFalse(analysis.persistent_storage_write_implemented)
        self.assertFalse(analysis.generated_output_mutation_implemented)
        self.assertTrue(analysis.analysis_only)

    def test_components_cover_nodes_reserves_and_cost_sources(self) -> None:
        analysis = analyze_workflow_cost()
        workflow_components = workflow_cost_components_for_kind(
            "workflow_node",
            analysis,
        )
        cost_sources = workflow_cost_components_for_kind("cost_source", analysis)
        retry_reserves = workflow_cost_components_for_kind("retry_reserve", analysis)
        failure_reserves = workflow_cost_components_for_kind(
            "failure_reserve",
            analysis,
        )

        self.assertEqual(len(workflow_components), len(ASSISTANT_WORKFLOW_NODE_ORDER))
        self.assertEqual(len(cost_sources), 2)
        self.assertEqual(len(retry_reserves), 1)
        self.assertEqual(len(failure_reserves), 1)

        for component in analysis.components:
            self.assertEqual(
                set(component.model_dump(mode="json")),
                REQUIRED_COMPONENT_FIELDS,
            )
            self.assertEqual(
                component.serialization_version,
                "workflow_cost_component.v1",
            )
            self.assertIn("budget_enforcement", component.blocked_runtime_behaviors)
            self.assertFalse(component.pricing_lookup_implemented)
            self.assertFalse(component.budget_enforcement_implemented)
            self.assertFalse(component.cost_based_routing_implemented)
            self.assertFalse(component.provider_model_routing_implemented)
            self.assertFalse(component.workflow_control_implemented)
            self.assertFalse(component.retry_triggering_implemented)
            self.assertFalse(component.prompt_mutation_implemented)
            self.assertFalse(component.generated_output_mutation_implemented)
            self.assertTrue(component.analysis_only)

        generation = workflow_cost_component_by_id(
            "workflow_node::generation",
            analysis,
        )
        retry = workflow_cost_component_by_id("reserve::retry_path", analysis)
        agent_costs = workflow_cost_component_by_id(
            "cost_source::agent_cost_tracking_foundation",
            analysis,
        )
        engine_costs = workflow_cost_component_by_id(
            "cost_source::artifact_engine_contracts",
            analysis,
        )

        self.assertIsNotNone(generation)
        self.assertIsNotNone(retry)
        self.assertIsNotNone(agent_costs)
        self.assertIsNotNone(engine_costs)
        assert generation is not None
        assert retry is not None
        assert agent_costs is not None
        assert engine_costs is not None
        self.assertEqual(generation.relative_cost, "medium")
        self.assertIn("provider", generation.advisory_actions[0])
        self.assertGreater(retry.estimated_token_cost, generation.estimated_token_cost)
        self.assertEqual(agent_costs.estimated_token_cost, 0)
        self.assertGreater(engine_costs.cost_weight, 0)

    def test_creative_plan_adjusts_generation_and_retry_estimates(self) -> None:
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
        analysis = analyze_workflow_cost(creative_plan=plan)
        generation = workflow_cost_component_by_id(
            "workflow_node::generation",
            analysis,
        )
        plan_component = workflow_cost_component_by_id(
            "creative_plan::estimated_token_cost",
            analysis,
        )

        self.assertIsNotNone(generation)
        self.assertIsNotNone(plan_component)
        assert generation is not None
        assert plan_component is not None
        self.assertEqual(generation.estimated_token_cost, plan.estimated_token_cost)
        self.assertEqual(
            plan_component.estimated_token_cost,
            plan.estimated_token_cost,
        )
        self.assertEqual(
            analysis.creative_plan_token_estimate,
            plan.estimated_token_cost,
        )
        self.assertEqual(
            analysis.retry_iteration_count,
            min(plan.refinement_budget, MAX_WORKFLOW_REFINEMENT_COUNT),
        )
        self.assertIn(
            "creative_plan::estimated_token_cost",
            analysis.component_ids,
        )

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        analysis = analyze_workflow_cost()
        retrieval = workflow_cost_component_by_id("workflow_node::retrieval", analysis)
        missing = workflow_cost_component_by_id("missing", analysis)

        self.assertIsNone(missing)
        self.assertIsNotNone(retrieval)
        assert retrieval is not None
        self.assertEqual(retrieval.source_id, "retrieval")
        self.assertEqual(retrieval.component_kind, "workflow_node")
        self.assertEqual(
            workflow_cost_component_by_id("workflow_node::retrieval", analysis),
            retrieval,
        )
        self.assertEqual(
            workflow_cost_components_for_kind("creative_plan", analysis), ()
        )

    def test_analysis_rejects_mismatched_components_or_totals(self) -> None:
        analysis = analyze_workflow_cost()
        payload = analysis.model_dump(mode="json")
        payload["component_ids"] = ("missing",) + tuple(payload["component_ids"][1:])

        with self.assertRaisesRegex(ValueError, "component_ids must match"):
            WorkflowCostAnalysis(**payload)

        payload = analysis.model_dump(mode="json")
        payload["critical_path_token_estimate"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "critical_path_token_estimate must match",
        ):
            WorkflowCostAnalysis(**payload)

    def test_analysis_does_not_declare_runtime_cost_control_terms(self) -> None:
        analysis = analyze_workflow_cost()
        combined_text = " ".join(
            (
                analysis.authority_boundary,
                *analysis.blocked_runtime_behaviors,
                *analysis.advisory_actions,
                *(
                    field
                    for component in analysis.components
                    for field in (
                        component.component_id,
                        component.source_id,
                        *component.evidence,
                        *component.advisory_actions,
                        *component.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "lookup_provider_price(",
            "enforce_budget(",
            "route_by_cost(",
            "select_model(",
            "trigger_retry(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
