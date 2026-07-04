import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    AgentDeterminismAuditRegistry,
    agent_contract_registry,
    agent_dependency_graph_registry,
    agent_determinism_audit_by_agent_id,
    agent_determinism_audit_registry,
    agent_determinism_audits_for_cacheability,
    agent_determinism_audits_for_routing_priority_band,
    agent_determinism_audits_for_scheduling_group,
    agent_routing_registry,
    engine_contract_consistency_registry,
    parallel_scheduling_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SOURCE_REGISTRIES = (
    "agent_contract_registry",
    "agent_dependency_graph_registry",
    "agent_routing_registry",
    "parallel_scheduling_registry",
    "engine_contract_consistency_registry",
)

REQUIRED_AUDIT_RECORD_FIELDS = {
    "agent_id",
    "role_id",
    "audit_stage",
    "contract_serialization_version",
    "contract_cacheability",
    "contract_required_input_order",
    "contract_optional_input_order",
    "contract_output_order",
    "dependency_node_id",
    "dependency_stage_id",
    "dependency_upstream_node_ids",
    "dependency_downstream_node_ids",
    "routing_priority_band",
    "route_candidates",
    "scheduling_group_id",
    "scheduling_group_agent_ids",
    "scheduling_blocking_group_ids",
    "scheduling_downstream_group_ids",
    "consistency_family_ids",
    "determinism_source_registries",
    "validated_determinism_surfaces",
    "passive_boundary_flags",
    "audit_findings",
    "missing_coverage_items",
    "contract_blocked_runtime_behaviors",
    "dependency_blocked_runtime_behaviors",
    "routing_blocked_runtime_behaviors",
    "scheduling_blocked_runtime_behaviors",
    "consistency_blocked_runtime_behaviors",
    "blocked_runtime_behaviors",
    "audit_status",
    "metadata_only_declared",
    "deterministic_cacheability_declared",
    "dependency_order_reference_present",
    "routing_order_reference_present",
    "scheduling_order_reference_present",
    "random_seed_generation_implemented",
    "nondeterministic_sampling_implemented",
    "active_agent_execution_implemented",
    "workflow_order_mutation_implemented",
    "route_selection_mutation_implemented",
    "parallel_execution_implemented",
    "provider_model_routing_implemented",
    "runtime_selection_implemented",
    "retry_triggering_implemented",
    "prompt_rendering_change_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class AgentDeterminismAuditTests(unittest.TestCase):
    def test_audit_registry_covers_determinism_sources(self) -> None:
        registry = agent_determinism_audit_registry()
        contracts = agent_contract_registry()
        dependency_graph = agent_dependency_graph_registry()
        routing = agent_routing_registry()
        scheduling = parallel_scheduling_registry()
        consistency = engine_contract_consistency_registry()

        self.assertEqual(registry.role, "agent_determinism_audit_registry")
        self.assertEqual(
            registry.serialization_version,
            "agent_determinism_audit_registry.v1",
        )
        self.assertEqual(
            registry.audit_stage,
            "v4_6_agent_determinism_hardening",
        )
        self.assertEqual(registry.agent_ids, contracts.agent_ids)
        self.assertEqual(
            registry.determinism_source_registries, EXPECTED_SOURCE_REGISTRIES
        )
        self.assertEqual(
            registry.cacheability_modes,
            tuple(
                dict.fromkeys(contract.cacheability for contract in contracts.contracts)
            ),
        )
        self.assertEqual(registry.dependency_stage_order, dependency_graph.stage_order)
        self.assertEqual(
            registry.dependency_topological_node_order,
            dependency_graph.topological_node_order,
        )
        self.assertEqual(
            registry.route_names,
            tuple(route.value for route in routing.route_names),
        )
        self.assertEqual(registry.scheduling_group_ids, scheduling.group_ids)
        self.assertEqual(registry.scheduling_agent_order, scheduling.agent_ids)
        self.assertEqual(registry.consistency_family_ids, consistency.family_ids)
        self.assertEqual(registry.audit_count, 12)
        self.assertTrue(registry.all_agents_covered)
        self.assertTrue(registry.stable_order_references_present)
        self.assertTrue(registry.no_missing_coverage)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.active_determinism_engine_implemented)
        self.assertFalse(registry.random_seed_generation_implemented)
        self.assertFalse(registry.nondeterministic_sampling_implemented)
        self.assertFalse(registry.active_agent_execution_implemented)
        self.assertFalse(registry.workflow_order_mutation_implemented)
        self.assertFalse(registry.route_selection_mutation_implemented)
        self.assertFalse(registry.parallel_execution_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.runtime_selection_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.prompt_rendering_change_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertIn("does not generate seeds", registry.authority_boundary)

    def test_audit_records_are_passive_and_order_aligned(self) -> None:
        registry = agent_determinism_audit_registry()
        known_nodes = set(registry.dependency_topological_node_order)
        known_routes = set(registry.route_names)
        known_groups = set(registry.scheduling_group_ids)

        for record in registry.audit_records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_AUDIT_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "agent_determinism_audit_record.v1",
            )
            self.assertEqual(record.audit_status, "pass")
            self.assertEqual(record.audit_stage, registry.audit_stage)
            self.assertTrue(record.contract_cacheability.startswith("deterministic_"))
            self.assertIn(record.dependency_node_id, known_nodes)
            self.assertTrue(set(record.route_candidates).issubset(known_routes))
            self.assertIn(record.scheduling_group_id, known_groups)
            self.assertEqual(
                record.consistency_family_ids, registry.consistency_family_ids
            )
            self.assertEqual(
                record.determinism_source_registries,
                registry.determinism_source_registries,
            )
            self.assertEqual(
                record.validated_determinism_surfaces,
                registry.validated_determinism_surfaces,
            )
            self.assertEqual(
                record.passive_boundary_flags, registry.passive_boundary_flags
            )
            self.assertFalse(record.missing_coverage_items)
            self.assertTrue(record.contract_output_order)
            self.assertIn("agent_invocation", record.contract_blocked_runtime_behaviors)
            self.assertIn(
                "workflow_node_order_change",
                record.dependency_blocked_runtime_behaviors,
            )
            self.assertIn(
                "active_dynamic_agent_routing",
                record.routing_blocked_runtime_behaviors,
            )
            self.assertIn(
                "parallel_task_execution",
                record.scheduling_blocked_runtime_behaviors,
            )
            self.assertIn(
                "prompt_rendering_change",
                record.consistency_blocked_runtime_behaviors,
            )
            self.assertTrue(record.metadata_only_declared)
            self.assertTrue(record.deterministic_cacheability_declared)
            self.assertTrue(record.dependency_order_reference_present)
            self.assertTrue(record.routing_order_reference_present)
            self.assertTrue(record.scheduling_order_reference_present)
            self.assertTrue(record.metadata_only)
            self.assertFalse(record.random_seed_generation_implemented)
            self.assertFalse(record.nondeterministic_sampling_implemented)
            self.assertFalse(record.active_agent_execution_implemented)
            self.assertFalse(record.workflow_order_mutation_implemented)
            self.assertFalse(record.route_selection_mutation_implemented)
            self.assertFalse(record.parallel_execution_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.runtime_selection_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.prompt_rendering_change_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)

    def test_audit_lookup_cacheability_band_and_group_filtering_are_stable(
        self,
    ) -> None:
        registry = agent_determinism_audit_registry()
        planner_audit = agent_determinism_audit_by_agent_id("planner_agent")
        missing_audit = agent_determinism_audit_by_agent_id("missing_agent")
        deterministic_audits = agent_determinism_audits_for_cacheability(
            "deterministic_with_upstream_metadata"
        )
        domain_band_audits = agent_determinism_audits_for_routing_priority_band(
            "domain_context"
        )
        foundational_group_audits = agent_determinism_audits_for_scheduling_group(
            "parallel_group::foundational_context"
        )
        missing_group_audits = agent_determinism_audits_for_scheduling_group(
            "missing_group"
        )

        self.assertIsNone(missing_audit)
        self.assertIsNotNone(planner_audit)
        assert planner_audit is not None
        self.assertEqual(planner_audit.dependency_node_id, "agent::planner_agent")
        self.assertEqual(
            planner_audit.scheduling_group_id,
            "parallel_group::foundational_context",
        )
        self.assertIn(RouteName.GENERATE.value, planner_audit.route_candidates)
        self.assertGreaterEqual(len(deterministic_audits), 1)
        self.assertEqual(
            tuple(record.agent_id for record in foundational_group_audits),
            ("planner_agent", "research_agent"),
        )
        self.assertEqual(
            tuple(record.agent_id for record in domain_band_audits),
            (
                "research_agent",
                "style_agent",
                "art_direction_agent",
                "narrative_symbolic_agent",
            ),
        )
        self.assertEqual(missing_group_audits, ())
        self.assertIs(
            planner_audit,
            agent_determinism_audit_by_agent_id("planner_agent", registry),
        )

    def test_audit_registry_rejects_mismatched_or_incomplete_records(self) -> None:
        registry = agent_determinism_audit_registry()
        first_record = registry.audit_records[0]
        duplicate_record = first_record.model_copy(
            update={"contract_cacheability": "deterministic_static_metadata"}
        )
        mismatched_flags_record = first_record.model_copy(
            update={
                "passive_boundary_flags": (
                    "other_flag",
                    "nondeterministic_sampling_blocked",
                    "agent_invocation_blocked",
                    "workflow_order_mutation_blocked",
                    "route_selection_mutation_blocked",
                    "parallel_execution_blocked",
                    "provider_model_routing_blocked",
                    "generated_output_mutation_blocked",
                )
            }
        )
        incomplete_record = first_record.model_copy(
            update={"missing_coverage_items": ("random_sampling_enabled",)}
        )

        with self.assertRaisesRegex(ValueError, "agent_ids must be unique"):
            AgentDeterminismAuditRegistry(
                audit_records=(first_record, duplicate_record)
                + registry.audit_records[2:],
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                determinism_source_registries=registry.determinism_source_registries,
                cacheability_modes=registry.cacheability_modes,
                dependency_stage_order=registry.dependency_stage_order,
                dependency_topological_node_order=(
                    registry.dependency_topological_node_order
                ),
                route_names=registry.route_names,
                scheduling_group_ids=registry.scheduling_group_ids,
                scheduling_agent_order=registry.scheduling_agent_order,
                consistency_family_ids=registry.consistency_family_ids,
                validated_determinism_surfaces=(
                    registry.validated_determinism_surfaces
                ),
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "passive_boundary_flags"):
            AgentDeterminismAuditRegistry(
                audit_records=(mismatched_flags_record,) + registry.audit_records[1:],
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                determinism_source_registries=registry.determinism_source_registries,
                cacheability_modes=registry.cacheability_modes,
                dependency_stage_order=registry.dependency_stage_order,
                dependency_topological_node_order=(
                    registry.dependency_topological_node_order
                ),
                route_names=registry.route_names,
                scheduling_group_ids=registry.scheduling_group_ids,
                scheduling_agent_order=registry.scheduling_agent_order,
                consistency_family_ids=registry.consistency_family_ids,
                validated_determinism_surfaces=(
                    registry.validated_determinism_surfaces
                ),
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            AgentDeterminismAuditRegistry(
                audit_records=(incomplete_record,) + registry.audit_records[1:],
                agent_ids=registry.agent_ids,
                audit_count=registry.audit_count,
                determinism_source_registries=registry.determinism_source_registries,
                cacheability_modes=registry.cacheability_modes,
                dependency_stage_order=registry.dependency_stage_order,
                dependency_topological_node_order=(
                    registry.dependency_topological_node_order
                ),
                route_names=registry.route_names,
                scheduling_group_ids=registry.scheduling_group_ids,
                scheduling_agent_order=registry.scheduling_agent_order,
                consistency_family_ids=registry.consistency_family_ids,
                validated_determinism_surfaces=(
                    registry.validated_determinism_surfaces
                ),
                passive_boundary_flags=registry.passive_boundary_flags,
            )

    def test_agent_determinism_audit_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate deterministic p5.js composition metadata.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        agent_determinism_audit_registry()
        agent_determinism_audit_by_agent_id("planner_agent")
        agent_determinism_audits_for_routing_priority_band("domain_context")
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)
        self.assertNotIn(
            "agent_determinism_audit_registry",
            next_decision.model_dump_json(),
        )

    def test_audit_metadata_does_not_declare_active_determinism_terms(self) -> None:
        registry = agent_determinism_audit_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for record in registry.audit_records
                    for field in (
                        record.agent_id,
                        *record.audit_findings,
                        *record.passive_boundary_flags,
                    )
                ),
            )
        )

        for forbidden_term in (
            "generate_runtime_seed",
            "sample_randomness",
            "execute_agent",
            "mutate_route_selection",
            "run_parallel_schedule",
            "route_provider",
            "change_prompt_rendering",
            "mutate_generated_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
