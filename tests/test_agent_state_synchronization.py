import unittest

from creative_coding_assistant.orchestration import (
    AgentStateSynchronizationRegistry,
    agent_lifecycle_registry,
    agent_state_conflict_surface_by_id,
    agent_state_sync_checkpoint_by_id,
    agent_state_sync_profile_by_agent_id,
    agent_state_synchronization_registry,
    shared_context_view_registry,
)

REQUIRED_PROFILE_FIELDS = {
    "agent_id",
    "sync_profile_id",
    "source_lifecycle_profile_id",
    "source_context_view_id",
    "visible_blackboard_channel_ids",
    "sync_checkpoint_ids",
    "consistency_constraint_ids",
    "stale_warning_ids",
    "conflict_surface_ids",
    "sync_boundary",
    "blocked_runtime_behaviors",
    "runtime_synchronization_implemented",
    "blackboard_mutation_implemented",
    "conflict_resolution_implemented",
    "storage_mutation_implemented",
    "serialization_version",
    "metadata_only",
}

REQUIRED_CHECKPOINT_FIELDS = {
    "checkpoint_id",
    "checkpoint_stage",
    "applicable_lifecycle_states",
    "consistency_constraint_ids",
    "checkpoint_boundary",
    "blocked_runtime_behaviors",
    "runtime_synchronization_implemented",
    "blackboard_mutation_implemented",
    "conflict_resolution_implemented",
    "storage_mutation_implemented",
    "serialization_version",
    "metadata_only",
}

REQUIRED_WARNING_FIELDS = {
    "warning_id",
    "category",
    "affected_lifecycle_states",
    "advisory_evidence_keys",
    "warning_boundary",
    "blocked_runtime_behaviors",
    "stale_state_detection_implemented",
    "runtime_synchronization_implemented",
    "blackboard_mutation_implemented",
    "serialization_version",
    "metadata_only",
}

REQUIRED_CONFLICT_SURFACE_FIELDS = {
    "conflict_surface_id",
    "category",
    "source_registry_ids",
    "conflict_boundary",
    "blocked_runtime_behaviors",
    "conflict_detection_implemented",
    "conflict_resolution_implemented",
    "blackboard_mutation_implemented",
    "serialization_version",
    "metadata_only",
}

EXPECTED_SOURCE_REGISTRIES = (
    "agent_lifecycle_registry",
    "shared_context_view_registry",
    "blackboard_memory_registry",
    "agent_dependency_graph_registry",
    "agent_state_synchronization_registry",
    "agent_escalation_signal_registry",
)


class AgentStateSynchronizationTests(unittest.TestCase):
    def test_registry_maps_lifecycle_profiles_to_context_views(self) -> None:
        registry = agent_state_synchronization_registry()
        lifecycle = agent_lifecycle_registry()
        context_views = shared_context_view_registry()

        self.assertEqual(registry.role, "agent_state_synchronization_registry")
        self.assertEqual(registry.serialization_version, "agent_state_sync_registry.v1")
        self.assertEqual(registry.agent_ids, lifecycle.agent_ids)
        self.assertEqual(registry.agent_ids, context_views.agent_ids)
        self.assertEqual(registry.profile_count, 12)
        self.assertEqual(len(registry.checkpoints), 5)
        self.assertEqual(len(registry.constraints), 4)
        self.assertEqual(len(registry.stale_warnings), 4)
        self.assertEqual(len(registry.conflict_surfaces), 4)
        self.assertEqual(
            registry.source_registries,
            EXPECTED_SOURCE_REGISTRIES,
        )
        self.assertIn("does not synchronize runtime state", registry.authority_boundary)
        self.assertFalse(registry.runtime_synchronization_implemented)
        self.assertFalse(registry.blackboard_mutation_implemented)
        self.assertFalse(registry.conflict_resolution_implemented)
        self.assertFalse(registry.storage_mutation_implemented)
        self.assertTrue(registry.metadata_only)

    def test_profiles_checkpoints_warnings_and_conflicts_are_passive(self) -> None:
        registry = agent_state_synchronization_registry()
        checkpoint_ids = set(registry.checkpoint_ids)
        constraint_ids = set(registry.constraint_ids)
        stale_warning_ids = set(registry.stale_warning_ids)
        conflict_surface_ids = set(registry.conflict_surface_ids)

        for profile in registry.profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertTrue(set(profile.sync_checkpoint_ids).issubset(checkpoint_ids))
            self.assertTrue(
                set(profile.consistency_constraint_ids).issubset(constraint_ids)
            )
            self.assertTrue(set(profile.stale_warning_ids).issubset(stale_warning_ids))
            self.assertTrue(
                set(profile.conflict_surface_ids).issubset(conflict_surface_ids)
            )
            self.assertIn(
                "blackboard_mutation",
                profile.blocked_runtime_behaviors,
            )
            self.assertFalse(profile.runtime_synchronization_implemented)
            self.assertFalse(profile.blackboard_mutation_implemented)
            self.assertFalse(profile.conflict_resolution_implemented)
            self.assertFalse(profile.storage_mutation_implemented)
            self.assertTrue(profile.metadata_only)

        for checkpoint in registry.checkpoints:
            dumped = checkpoint.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CHECKPOINT_FIELDS)
            self.assertTrue(checkpoint.applicable_lifecycle_states)
            self.assertTrue(
                set(checkpoint.consistency_constraint_ids).issubset(constraint_ids)
            )
            self.assertFalse(checkpoint.runtime_synchronization_implemented)
            self.assertFalse(checkpoint.blackboard_mutation_implemented)
            self.assertFalse(checkpoint.conflict_resolution_implemented)
            self.assertFalse(checkpoint.storage_mutation_implemented)
            self.assertTrue(checkpoint.metadata_only)

        for warning in registry.stale_warnings:
            dumped = warning.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_WARNING_FIELDS)
            self.assertFalse(warning.stale_state_detection_implemented)
            self.assertFalse(warning.runtime_synchronization_implemented)
            self.assertFalse(warning.blackboard_mutation_implemented)
            self.assertTrue(warning.metadata_only)

        for surface in registry.conflict_surfaces:
            dumped = surface.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CONFLICT_SURFACE_FIELDS)
            self.assertTrue(
                set(surface.source_registry_ids).issubset(registry.source_registries)
            )
            self.assertFalse(surface.conflict_detection_implemented)
            self.assertFalse(surface.conflict_resolution_implemented)
            self.assertFalse(surface.blackboard_mutation_implemented)
            self.assertTrue(surface.metadata_only)

    def test_state_sync_lookups_are_stable_and_non_mutating(self) -> None:
        planner = agent_state_sync_profile_by_agent_id("planner_agent")
        checkpoint = agent_state_sync_checkpoint_by_id(
            "state_sync_pre_activation_checkpoint"
        )
        conflict_surface = agent_state_conflict_surface_by_id(
            "lifecycle_context_conflict_surface"
        )

        self.assertIsNone(agent_state_sync_profile_by_agent_id("missing_agent"))
        self.assertIsNone(agent_state_sync_checkpoint_by_id("missing_checkpoint"))
        self.assertIsNone(agent_state_conflict_surface_by_id("missing_surface"))
        self.assertIsNotNone(planner)
        self.assertIsNotNone(checkpoint)
        self.assertIsNotNone(conflict_surface)
        assert planner is not None
        assert checkpoint is not None
        assert conflict_surface is not None
        self.assertEqual(planner.source_lifecycle_profile_id, "planner_agent_lifecycle_profile")
        self.assertEqual(planner.source_context_view_id, "planner_agent_shared_context_view")
        self.assertEqual(checkpoint.checkpoint_stage, "pre_activation")
        self.assertEqual(conflict_surface.category, "lifecycle_context")
        self.assertFalse(conflict_surface.conflict_resolution_implemented)

    def test_registry_rejects_unknown_sync_references(self) -> None:
        registry = agent_state_synchronization_registry()
        unknown_checkpoint_profile = registry.profiles[0].model_copy(
            update={"sync_checkpoint_ids": ("missing_checkpoint",)}
        )
        unknown_constraint_checkpoint = registry.checkpoints[0].model_copy(
            update={"consistency_constraint_ids": ("missing_constraint",)}
        )
        missing_source_registries = (
            registry.source_registries[:-1] + ("missing_registry",)
        )

        with self.assertRaisesRegex(ValueError, "profile checkpoints must be known"):
            AgentStateSynchronizationRegistry(
                profiles=(unknown_checkpoint_profile,) + registry.profiles[1:],
                checkpoints=registry.checkpoints,
                constraints=registry.constraints,
                stale_warnings=registry.stale_warnings,
                conflict_surfaces=registry.conflict_surfaces,
                agent_ids=registry.agent_ids,
                profile_ids=registry.profile_ids,
                checkpoint_ids=registry.checkpoint_ids,
                constraint_ids=registry.constraint_ids,
                stale_warning_ids=registry.stale_warning_ids,
                conflict_surface_ids=registry.conflict_surface_ids,
                profile_count=registry.profile_count,
                source_registries=registry.source_registries,
            )

        with self.assertRaisesRegex(ValueError, "checkpoint constraints must be known"):
            AgentStateSynchronizationRegistry(
                profiles=registry.profiles,
                checkpoints=(unknown_constraint_checkpoint,) + registry.checkpoints[1:],
                constraints=registry.constraints,
                stale_warnings=registry.stale_warnings,
                conflict_surfaces=registry.conflict_surfaces,
                agent_ids=registry.agent_ids,
                profile_ids=registry.profile_ids,
                checkpoint_ids=registry.checkpoint_ids,
                constraint_ids=registry.constraint_ids,
                stale_warning_ids=registry.stale_warning_ids,
                conflict_surface_ids=registry.conflict_surface_ids,
                profile_count=registry.profile_count,
                source_registries=registry.source_registries,
            )

        with self.assertRaisesRegex(
            ValueError,
            "source_registries must match conflict surface sources",
        ):
            AgentStateSynchronizationRegistry(
                profiles=registry.profiles,
                checkpoints=registry.checkpoints,
                constraints=registry.constraints,
                stale_warnings=registry.stale_warnings,
                conflict_surfaces=registry.conflict_surfaces,
                agent_ids=registry.agent_ids,
                profile_ids=registry.profile_ids,
                checkpoint_ids=registry.checkpoint_ids,
                constraint_ids=registry.constraint_ids,
                stale_warning_ids=registry.stale_warning_ids,
                conflict_surface_ids=registry.conflict_surface_ids,
                profile_count=registry.profile_count,
                source_registries=missing_source_registries,
            )

    def test_state_sync_does_not_mutate_context_or_storage_metadata(self) -> None:
        context_before = shared_context_view_registry().model_dump(mode="json")
        registry = agent_state_synchronization_registry()
        context_after = shared_context_view_registry().model_dump(mode="json")
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.profiles
                    for field in (
                        profile.sync_profile_id,
                        profile.sync_boundary,
                        *profile.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        self.assertEqual(context_before, context_after)
        for forbidden_term in (
            "synchronize_runtime_state",
            "mutate_blackboard",
            "write_storage",
            "resolve_conflict",
            "invoke_agent",
            "route_provider",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
