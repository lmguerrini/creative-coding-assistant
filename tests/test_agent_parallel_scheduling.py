import unittest

from creative_coding_assistant.orchestration import (
    ParallelSchedulingRegistry,
    agent_dependency_graph_registry,
    parallel_scheduling_group_by_id,
    parallel_scheduling_group_for_agent,
    parallel_scheduling_registry,
)

REQUIRED_GROUP_FIELDS = {
    "group_id",
    "stage_id",
    "agent_ids",
    "scheduling_hint",
    "max_parallel_agents",
    "blocking_group_ids",
    "downstream_group_ids",
    "safety_flags",
    "source_dependency_nodes",
    "blocked_runtime_behaviors",
    "parallel_execution_implemented",
    "async_behavior_changed",
    "workflow_timing_changed",
    "scheduler_runtime_hook_implemented",
    "serialization_version",
    "metadata_only",
}


class AgentParallelSchedulingTests(unittest.TestCase):
    def test_registry_derives_groups_from_dependency_stages(self) -> None:
        scheduling = parallel_scheduling_registry()
        dependency_graph = agent_dependency_graph_registry()

        self.assertEqual(scheduling.role, "parallel_scheduling_registry")
        self.assertEqual(
            scheduling.serialization_version,
            "parallel_scheduling_registry.v1",
        )
        self.assertEqual(scheduling.group_count, 6)
        self.assertEqual(
            tuple(group.stage_id for group in scheduling.groups),
            dependency_graph.stage_order,
        )
        self.assertEqual(
            scheduling.source_dependency_graph,
            "agent_dependency_graph_registry",
        )
        self.assertIn(
            "does not run tasks in parallel",
            scheduling.authority_boundary,
        )
        self.assertFalse(scheduling.parallel_execution_implemented)
        self.assertFalse(scheduling.async_behavior_changed)
        self.assertFalse(scheduling.workflow_timing_changed)
        self.assertFalse(scheduling.scheduler_runtime_hook_implemented)
        self.assertTrue(scheduling.metadata_only)

    def test_groups_expose_passive_scheduling_hints(self) -> None:
        scheduling = parallel_scheduling_registry()

        for group in scheduling.groups:
            dumped = group.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_GROUP_FIELDS)
            self.assertEqual(
                group.serialization_version,
                "parallel_scheduling_group.v1",
            )
            self.assertEqual(group.max_parallel_agents, len(group.agent_ids))
            self.assertTrue(group.source_dependency_nodes)
            self.assertIn("no_async_execution_hook", group.safety_flags)
            self.assertIn(
                "parallel_task_execution",
                group.blocked_runtime_behaviors,
            )
            self.assertFalse(group.parallel_execution_implemented)
            self.assertFalse(group.async_behavior_changed)
            self.assertFalse(group.workflow_timing_changed)
            self.assertFalse(group.scheduler_runtime_hook_implemented)
            self.assertTrue(group.metadata_only)

        foundational = parallel_scheduling_group_by_id(
            "parallel_group::foundational_context"
        )
        refinement = parallel_scheduling_group_by_id(
            "parallel_group::refinement_context"
        )
        assert foundational is not None
        assert refinement is not None
        self.assertEqual(
            foundational.scheduling_hint,
            "parallel_after_upstream_dependencies",
        )
        self.assertEqual(
            refinement.scheduling_hint,
            "single_agent_after_upstream_dependencies",
        )

    def test_blocking_relationships_are_ordered_metadata(self) -> None:
        scheduling = parallel_scheduling_registry()
        group_index = {
            group.group_id: index for index, group in enumerate(scheduling.groups)
        }

        for group in scheduling.groups:
            for blocker in group.blocking_group_ids:
                self.assertLess(group_index[blocker], group_index[group.group_id])
            for downstream in group.downstream_group_ids:
                self.assertGreater(group_index[downstream], group_index[group.group_id])

        domain_group = parallel_scheduling_group_by_id("parallel_group::domain_context")
        assert domain_group is not None
        self.assertEqual(
            domain_group.blocking_group_ids,
            ("parallel_group::foundational_context",),
        )
        self.assertEqual(
            domain_group.downstream_group_ids,
            ("parallel_group::execution_context",),
        )

    def test_lookup_maps_agents_to_scheduling_groups(self) -> None:
        runtime_group = parallel_scheduling_group_for_agent("runtime_agent")
        missing_group = parallel_scheduling_group_for_agent("missing_agent")

        self.assertIsNone(missing_group)
        self.assertIsNotNone(runtime_group)
        assert runtime_group is not None
        self.assertEqual(runtime_group.stage_id, "execution_context")
        self.assertIn("artifact_agent", runtime_group.agent_ids)
        self.assertIsNone(parallel_scheduling_group_by_id("missing_group"))

    def test_registry_rejects_cyclic_or_mismatched_groups(self) -> None:
        scheduling = parallel_scheduling_registry()
        second_group = scheduling.groups[1].model_copy(
            update={"blocking_group_ids": ("parallel_group::execution_context",)}
        )

        with self.assertRaisesRegex(
            ValueError, "blocking relationships must be acyclic"
        ):
            ParallelSchedulingRegistry(
                groups=(scheduling.groups[0], second_group) + scheduling.groups[2:],
                group_ids=scheduling.group_ids,
                agent_ids=scheduling.agent_ids,
                group_count=scheduling.group_count,
            )

        with self.assertRaisesRegex(ValueError, "agent_ids must match groups"):
            ParallelSchedulingRegistry(
                groups=scheduling.groups,
                group_ids=scheduling.group_ids,
                agent_ids=("other_agent",) + scheduling.agent_ids[1:],
                group_count=scheduling.group_count,
            )

    def test_scheduling_metadata_does_not_declare_execution_hooks(self) -> None:
        scheduling = parallel_scheduling_registry()
        combined_text = " ".join(
            (
                scheduling.authority_boundary,
                *scheduling.blocked_runtime_behaviors,
                *(
                    field
                    for group in scheduling.groups
                    for field in (
                        group.group_id,
                        group.scheduling_hint,
                        *group.safety_flags,
                        *group.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "asyncio.gather",
            "create_task",
            "thread_pool",
            "execute_parallel",
            "workflow_timer",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
