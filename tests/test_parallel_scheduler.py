import unittest

from creative_coding_assistant.orchestration import (
    ParallelSchedulerPlan,
    analyze_assistant_execution_graph,
    parallel_schedule_candidate_by_id,
    parallel_schedule_candidates_for_status,
    parallel_scheduling_registry,
    plan_parallel_scheduler,
)

REQUIRED_SCHEDULE_CANDIDATE_FIELDS = {
    "candidate_id",
    "source_group_id",
    "stage_id",
    "agent_ids",
    "status",
    "advisory_rank",
    "dependency_depth",
    "blocking_candidate_ids",
    "downstream_candidate_ids",
    "max_parallel_agents",
    "advisory_parallel_slot_count",
    "advisory_parallelism_score",
    "source_scheduling_hint",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "parallel_scheduler_planning_implemented",
    "parallel_execution_implemented",
    "async_execution_implemented",
    "workflow_timing_mutation_implemented",
    "workflow_graph_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "node_handler_invocation_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "planning_only",
}


class ParallelSchedulerTests(unittest.TestCase):
    def test_default_plan_derives_advisory_parallel_candidates(self) -> None:
        scheduling = parallel_scheduling_registry()
        graph = analyze_assistant_execution_graph()
        plan = plan_parallel_scheduler(
            scheduling_registry=scheduling,
            execution_graph=graph,
        )

        self.assertEqual(plan.role, "parallel_scheduler")
        self.assertEqual(plan.serialization_version, "parallel_scheduler_plan.v1")
        self.assertEqual(
            plan.source_parallel_scheduling_serialization_version,
            scheduling.serialization_version,
        )
        self.assertEqual(
            plan.source_execution_graph_serialization_version,
            graph.serialization_version,
        )
        self.assertEqual(plan.source_graph_node_count, graph.node_count)
        self.assertEqual(
            plan.source_graph_failure_entry_node_ids,
            graph.failure_entry_node_ids,
        )
        self.assertEqual(plan.candidate_count, 6)
        self.assertEqual(
            plan.candidate_ids,
            (
                "parallel_scheduler::foundational_context",
                "parallel_scheduler::domain_context",
                "parallel_scheduler::execution_context",
                "parallel_scheduler::quality_review",
                "parallel_scheduler::refinement_context",
                "parallel_scheduler::final_synthesis",
            ),
        )
        self.assertEqual(plan.parallel_candidate_count, 4)
        self.assertEqual(plan.serial_guardrail_count, 2)
        self.assertEqual(plan.max_concurrency_width, 3)
        self.assertEqual(plan.total_advisory_parallel_slots, 12)
        self.assertEqual(plan.total_advisory_parallelism_score, 600)
        self.assertEqual(plan.scheduler_pressure, "high")
        self.assertTrue(plan.failure_normalization_preserved)
        self.assertIn("does not run tasks in parallel", plan.authority_boundary)
        self.assertTrue(plan.parallel_scheduler_planning_implemented)
        self.assertFalse(plan.parallel_execution_implemented)
        self.assertFalse(plan.async_execution_implemented)
        self.assertFalse(plan.workflow_timing_mutation_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.graph_compilation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.node_handler_invocation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.retry_triggering_implemented)
        self.assertFalse(plan.prompt_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertTrue(plan.planning_only)

    def test_candidates_preserve_parallel_and_serial_boundaries(self) -> None:
        plan = plan_parallel_scheduler()
        domain = parallel_schedule_candidate_by_id(
            "parallel_scheduler::domain_context",
            plan,
        )
        refinement = parallel_schedule_candidate_by_id(
            "parallel_scheduler::refinement_context",
            plan,
        )

        self.assertIsNotNone(domain)
        self.assertIsNotNone(refinement)
        assert domain is not None
        assert refinement is not None
        self.assertEqual(domain.status, "parallel_candidate")
        self.assertEqual(domain.max_parallel_agents, 3)
        self.assertEqual(domain.advisory_parallelism_score, 200)
        self.assertEqual(
            domain.blocking_candidate_ids,
            ("parallel_scheduler::foundational_context",),
        )
        self.assertEqual(
            domain.downstream_candidate_ids,
            ("parallel_scheduler::execution_context",),
        )
        self.assertEqual(refinement.status, "serial_guardrail")
        self.assertEqual(refinement.advisory_parallelism_score, 0)

        for candidate in plan.candidates:
            self.assertEqual(
                set(candidate.model_dump(mode="json")),
                REQUIRED_SCHEDULE_CANDIDATE_FIELDS,
            )
            self.assertEqual(
                candidate.serialization_version,
                "parallel_schedule_candidate.v1",
            )
            self.assertEqual(
                candidate.advisory_parallel_slot_count,
                candidate.max_parallel_agents,
            )
            self.assertIn(
                "parallel_task_execution",
                candidate.blocked_runtime_behaviors,
            )
            self.assertTrue(candidate.parallel_scheduler_planning_implemented)
            self.assertFalse(candidate.parallel_execution_implemented)
            self.assertFalse(candidate.async_execution_implemented)
            self.assertFalse(candidate.workflow_timing_mutation_implemented)
            self.assertFalse(candidate.workflow_graph_mutation_implemented)
            self.assertFalse(candidate.graph_compilation_implemented)
            self.assertFalse(candidate.workflow_execution_implemented)
            self.assertFalse(candidate.node_handler_invocation_implemented)
            self.assertFalse(candidate.provider_model_routing_implemented)
            self.assertFalse(candidate.retry_triggering_implemented)
            self.assertFalse(candidate.prompt_mutation_implemented)
            self.assertFalse(candidate.persistent_storage_write_implemented)
            self.assertFalse(candidate.generated_output_mutation_implemented)
            self.assertTrue(candidate.planning_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        plan = plan_parallel_scheduler()
        parallel_candidates = parallel_schedule_candidates_for_status(
            "parallel_candidate",
            plan,
        )
        serial_guardrails = parallel_schedule_candidates_for_status(
            "serial_guardrail",
            plan,
        )
        missing = parallel_schedule_candidate_by_id("missing", plan)

        self.assertIsNone(missing)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in parallel_candidates),
            plan.parallel_candidate_ids,
        )
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in serial_guardrails),
            plan.serial_guardrail_candidate_ids,
        )
        self.assertIs(
            parallel_candidates[0],
            parallel_schedule_candidate_by_id(
                parallel_candidates[0].candidate_id,
                plan,
            ),
        )

    def test_plan_rejects_mismatched_candidates_or_scores(self) -> None:
        plan = plan_parallel_scheduler()
        payload = plan.model_dump(mode="json")
        payload["candidate_ids"] = ("missing",) + tuple(payload["candidate_ids"][1:])

        with self.assertRaisesRegex(ValueError, "candidate_ids must match"):
            ParallelSchedulerPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["total_advisory_parallelism_score"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "total_advisory_parallelism_score must match",
        ):
            ParallelSchedulerPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["scheduler_pressure"] = "low"

        with self.assertRaisesRegex(ValueError, "scheduler_pressure must match"):
            ParallelSchedulerPlan(**payload)

    def test_plan_does_not_declare_runtime_scheduler_terms(self) -> None:
        plan = plan_parallel_scheduler()
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
                        candidate.source_group_id,
                        candidate.source_scheduling_hint,
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
            "thread_pool(",
            "execute_parallel(",
            "compile_graph(",
            "execute_workflow(",
            "invoke_node_handler(",
            "route_provider(",
            "trigger_retry(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
