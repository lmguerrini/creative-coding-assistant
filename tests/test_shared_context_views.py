import unittest

from creative_coding_assistant.orchestration import (
    SharedContextViewRegistry,
    agent_memory_contract_registry,
    blackboard_memory_registry,
    shared_context_view_by_agent_id,
    shared_context_view_by_id,
    shared_context_view_registry,
)

REQUIRED_VIEW_FIELDS = {
    "agent_id",
    "view_id",
    "view_stage",
    "access_mode",
    "visible_memory_surfaces",
    "visible_blackboard_channel_ids",
    "hidden_blackboard_channel_ids",
    "visible_metadata_keys",
    "source_memory_contract_id",
    "source_blackboard_registry",
    "view_boundary",
    "blocked_runtime_behaviors",
    "unrestricted_global_state_exposed",
    "runtime_memory_implemented",
    "context_materialization_implemented",
    "context_mutation_implemented",
    "storage_backend_implemented",
    "serialization_version",
    "metadata_only",
}


class SharedContextViewTests(unittest.TestCase):
    def test_registry_covers_agent_memory_and_blackboard_contracts(self) -> None:
        memory_registry = agent_memory_contract_registry()
        blackboard_registry = blackboard_memory_registry()
        view_registry = shared_context_view_registry()

        self.assertEqual(view_registry.role, "shared_context_view_registry")
        self.assertEqual(
            view_registry.serialization_version,
            "shared_context_view_registry.v1",
        )
        self.assertEqual(view_registry.agent_ids, memory_registry.agent_ids)
        self.assertEqual(
            view_registry.blackboard_channel_ids, blackboard_registry.channel_ids
        )
        self.assertEqual(view_registry.view_count, 12)
        self.assertEqual(
            view_registry.source_registries,
            ("agent_memory_contract_registry", "blackboard_memory_registry"),
        )
        self.assertFalse(view_registry.unrestricted_global_state_exposed)
        self.assertFalse(view_registry.runtime_memory_implemented)
        self.assertFalse(view_registry.context_materialization_implemented)
        self.assertFalse(view_registry.context_mutation_implemented)
        self.assertFalse(view_registry.storage_backend_implemented)
        self.assertTrue(view_registry.metadata_only)
        self.assertIn(
            "does not expose unrestricted global state",
            view_registry.authority_boundary,
        )

    def test_each_view_is_scoped_and_hides_context(self) -> None:
        registry = shared_context_view_registry()
        all_channels = set(registry.blackboard_channel_ids)

        for view in registry.views:
            dumped = view.model_dump(mode="json")
            visible = set(view.visible_blackboard_channel_ids)
            hidden = set(view.hidden_blackboard_channel_ids)

            self.assertEqual(set(dumped), REQUIRED_VIEW_FIELDS)
            self.assertEqual(view.serialization_version, "shared_context_view.v1")
            self.assertEqual(view.access_mode, "scoped_metadata_view")
            self.assertTrue(view.visible_memory_surfaces)
            self.assertTrue(view.visible_blackboard_channel_ids)
            self.assertTrue(view.hidden_blackboard_channel_ids)
            self.assertLess(len(visible), len(all_channels))
            self.assertEqual(hidden, all_channels - visible)
            self.assertTrue(view.visible_metadata_keys)
            self.assertIn(
                "unrestricted_global_state_access",
                view.blocked_runtime_behaviors,
            )
            self.assertFalse(view.unrestricted_global_state_exposed)
            self.assertFalse(view.runtime_memory_implemented)
            self.assertFalse(view.context_materialization_implemented)
            self.assertFalse(view.context_mutation_implemented)
            self.assertFalse(view.storage_backend_implemented)
            self.assertTrue(view.metadata_only)

    def test_view_lookup_preserves_agent_specific_scope(self) -> None:
        planner_view = shared_context_view_by_agent_id("planner_agent")
        style_view = shared_context_view_by_id("style_agent_shared_context_view")

        self.assertIsNone(shared_context_view_by_agent_id("missing_agent"))
        self.assertIsNone(shared_context_view_by_id("missing_view"))
        self.assertIsNotNone(planner_view)
        self.assertIsNotNone(style_view)
        assert planner_view is not None
        assert style_view is not None
        self.assertEqual(planner_view.agent_id, "planner_agent")
        self.assertIn(
            "planner_agent_blackboard_channel",
            planner_view.visible_blackboard_channel_ids,
        )
        self.assertNotIn(
            "style_agent_blackboard_channel",
            planner_view.visible_blackboard_channel_ids,
        )
        self.assertIn(
            "style_agent_blackboard_channel",
            style_view.visible_blackboard_channel_ids,
        )
        self.assertIn("planning_context_packet", planner_view.visible_metadata_keys)
        self.assertIn("visual_style_constraints", style_view.visible_metadata_keys)

    def test_registry_rejects_unrestricted_or_mismatched_views(self) -> None:
        registry = shared_context_view_registry()
        first_view = registry.views[0]
        unrestricted_view = first_view.model_copy(
            update={
                "visible_blackboard_channel_ids": registry.blackboard_channel_ids,
                "hidden_blackboard_channel_ids": (),
            }
        )

        with self.assertRaisesRegex(ValueError, "must not expose every channel"):
            SharedContextViewRegistry(
                views=(unrestricted_view,) + registry.views[1:],
                agent_ids=registry.agent_ids,
                view_ids=registry.view_ids,
                blackboard_channel_ids=registry.blackboard_channel_ids,
                view_count=12,
                source_registries=registry.source_registries,
            )

        with self.assertRaisesRegex(ValueError, "agent_ids must match views"):
            SharedContextViewRegistry(
                views=registry.views,
                agent_ids=("other_agent",) + registry.agent_ids[1:],
                view_ids=registry.view_ids,
                blackboard_channel_ids=registry.blackboard_channel_ids,
                view_count=12,
                source_registries=registry.source_registries,
            )

    def test_view_registry_does_not_declare_runtime_context_mutation(self) -> None:
        registry = shared_context_view_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for view in registry.views
                    for field in (
                        view.view_id,
                        view.access_mode,
                        view.view_boundary,
                        *view.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "global_state_dump",
            "runtime_memory_write",
            "context_mutation_trigger",
            "storage_adapter",
            "execute_provider",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
