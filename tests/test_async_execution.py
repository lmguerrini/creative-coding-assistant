import unittest

from creative_coding_assistant.orchestration import (
    AsyncExecutionPlan,
    async_execution_candidate_by_id,
    async_execution_candidates_for_status,
    optimize_latency,
    plan_async_execution,
    plan_parallel_scheduler,
)

REQUIRED_ASYNC_CANDIDATE_FIELDS = {
    "candidate_id",
    "source_schedule_candidate_id",
    "source_latency_candidate_id",
    "stage_id",
    "agent_ids",
    "status",
    "async_execution_mode",
    "advisory_rank",
    "dependency_depth",
    "max_parallel_agents",
    "blocking_input_count",
    "advisory_async_slot_count",
    "advisory_async_readiness_score",
    "required_guards",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "async_execution_planning_implemented",
    "event_loop_task_creation_implemented",
    "async_runtime_execution_implemented",
    "parallel_execution_implemented",
    "workflow_timing_change_implemented",
    "workflow_graph_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "provider_model_routing_implemented",
    "cancellation_policy_enforcement_implemented",
    "timeout_policy_enforcement_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class AsyncExecutionTests(unittest.TestCase):
    def test_default_plan_derives_advisory_async_candidates(self) -> None:
        scheduler = plan_parallel_scheduler()
        latency = optimize_latency(parallel_scheduler=scheduler)
        plan = plan_async_execution(
            parallel_scheduler=scheduler,
            latency_optimization=latency,
        )

        self.assertEqual(plan.role, "async_execution_planner")
        self.assertEqual(plan.serialization_version, "async_execution_plan.v1")
        self.assertEqual(
            plan.source_parallel_scheduler_serialization_version,
            scheduler.serialization_version,
        )
        self.assertEqual(
            plan.source_latency_optimization_serialization_version,
            latency.serialization_version,
        )
        self.assertTrue(plan.failure_normalization_required)
        self.assertEqual(plan.candidate_count, 6)
        self.assertEqual(
            plan.candidate_ids,
            (
                "async_execution::foundational_context",
                "async_execution::domain_context",
                "async_execution::execution_context",
                "async_execution::quality_review",
                "async_execution::refinement_context",
                "async_execution::final_synthesis",
            ),
        )
        self.assertEqual(plan.async_ready_candidate_count, 4)
        self.assertEqual(plan.serial_guardrail_count, 2)
        self.assertEqual(plan.max_async_width, 3)
        self.assertEqual(plan.total_advisory_async_slots, 12)
        self.assertEqual(plan.highest_advisory_async_readiness_score, 200)
        self.assertEqual(plan.total_advisory_async_readiness_score, 600)
        self.assertEqual(plan.async_execution_pressure, "high")
        self.assertIn("does not create event-loop tasks", plan.authority_boundary)
        self.assertTrue(plan.async_execution_planning_implemented)
        self.assertFalse(plan.event_loop_task_creation_implemented)
        self.assertFalse(plan.async_runtime_execution_implemented)
        self.assertFalse(plan.parallel_execution_implemented)
        self.assertFalse(plan.workflow_timing_change_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.graph_compilation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.cancellation_policy_enforcement_implemented)
        self.assertFalse(plan.timeout_policy_enforcement_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.advisory_only)

    def test_candidates_require_async_guards_without_runtime_hooks(self) -> None:
        plan = plan_async_execution()
        domain = async_execution_candidate_by_id(
            "async_execution::domain_context",
            plan,
        )
        refinement = async_execution_candidate_by_id(
            "async_execution::refinement_context",
            plan,
        )

        self.assertIsNotNone(domain)
        self.assertIsNotNone(refinement)
        assert domain is not None
        assert refinement is not None
        self.assertEqual(domain.status, "async_ready_candidate")
        self.assertEqual(domain.async_execution_mode, "bounded_parallel_group")
        self.assertEqual(domain.max_parallel_agents, 3)
        self.assertEqual(domain.advisory_async_readiness_score, 200)
        self.assertIn(
            "requires_cancellation_policy_before_activation",
            domain.required_guards,
        )
        self.assertEqual(refinement.status, "serial_guardrail")
        self.assertEqual(refinement.async_execution_mode, "ordered_serial_stage")
        self.assertEqual(refinement.advisory_async_readiness_score, 0)

        for candidate in plan.candidates:
            self.assertEqual(
                set(candidate.model_dump(mode="json")),
                REQUIRED_ASYNC_CANDIDATE_FIELDS,
            )
            self.assertEqual(
                candidate.serialization_version,
                "async_execution_candidate.v1",
            )
            self.assertEqual(
                candidate.advisory_async_slot_count,
                candidate.max_parallel_agents,
            )
            self.assertIn(
                "event_loop_task_creation",
                candidate.blocked_runtime_behaviors,
            )
            self.assertTrue(candidate.async_execution_planning_implemented)
            self.assertFalse(candidate.event_loop_task_creation_implemented)
            self.assertFalse(candidate.async_runtime_execution_implemented)
            self.assertFalse(candidate.parallel_execution_implemented)
            self.assertFalse(candidate.workflow_timing_change_implemented)
            self.assertFalse(candidate.workflow_graph_mutation_implemented)
            self.assertFalse(candidate.graph_compilation_implemented)
            self.assertFalse(candidate.workflow_execution_implemented)
            self.assertFalse(candidate.agent_invocation_implemented)
            self.assertFalse(candidate.node_handler_invocation_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.cancellation_policy_enforcement_implemented)
            self.assertFalse(candidate.timeout_policy_enforcement_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.persistent_storage_write_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.advisory_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = plan_async_execution()
        async_candidates = async_execution_candidates_for_status(
            "async_ready_candidate",
            plan,
        )
        guardrails = async_execution_candidates_for_status("serial_guardrail", plan)
        missing = async_execution_candidate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in async_candidates),
            plan.async_ready_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in guardrails),
            plan.serial_guardrail_candidate_ids,
        )
        self.assertIs(
            async_candidates[0],
            async_execution_candidate_by_id(async_candidates[0].candidate_id, plan),
        )

    def test_plan_rejects_mismatched_candidates_or_scores(self) -> None:
        plan = plan_async_execution()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            AsyncExecutionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_advisory_async_readiness_score"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_advisory_async_readiness_score must match",
        ):
            AsyncExecutionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["async_execution_pressure"] = "low"

        with self.assertRaisesRegex(
            ValueError,
            "async_execution_pressure must match",
        ):
            AsyncExecutionPlan(**payload)

    def test_plan_does_not_declare_runtime_async_terms(self) -> None:
        plan = plan_async_execution()
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for candidate in plan.candidates
                    for field in (
                        candidate.candidate_id,
                        candidate.source_schedule_candidate_id,
                        candidate.source_latency_candidate_id,
                        candidate.async_execution_mode,
                        *candidate.required_guards,
                        *candidate.evidence,
                        *candidate.advisory_actions,
                        *candidate.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "asyncio.gather(",
            "create_task(",
            "ensure_future(",
            "execute_parallel(",
            "change_workflow_timing(",
            "compile_graph(",
            "execute_workflow(",
            "invoke_agent(",
            "invoke_node_handler(",
            "route_provider(",
            "enforce_cancellation(",
            "enforce_timeout(",
            "trigger_retry(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
