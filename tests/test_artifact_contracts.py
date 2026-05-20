import unittest
from datetime import UTC, datetime, timedelta

from creative_coding_assistant.artifacts import (
    ArtifactCategory,
    ArtifactContentLocator,
    ArtifactContentReference,
    ArtifactIdentity,
    ArtifactMetadata,
    ArtifactRecord,
    ArtifactStatus,
    ArtifactTimestamps,
    ArtifactType,
    ArtifactWorkflowLink,
    ArtifactWorkspace,
    attach_artifact_to_workspace,
    can_transition_artifact_status,
    get_allowed_artifact_status_transitions,
    transition_artifact_status,
    validate_artifact_status_transition,
)
from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)


class ArtifactContractsTests(unittest.TestCase):
    def test_artifact_schema_normalizes_ids_tags_and_primary_reference(self) -> None:
        artifact = _artifact(
            metadata=ArtifactMetadata(
                title="WebGPU Gradient",
                tags=(" Preview ", "webgpu", "preview"),
                domain=CreativeCodingDomain.WEBGPU_WGSL,
            ),
            content_references=(
                ArtifactContentReference(
                    reference_id=" preview-png ",
                    locator=ArtifactContentLocator.WORKSPACE_FILE,
                    workspace_path="sessions/run-1/gradient.png",
                    mime_type="image/png",
                    is_primary=True,
                ),
                ArtifactContentReference(
                    reference_id="manifest",
                    locator=ArtifactContentLocator.URI,
                    uri="https://example.com/artifacts/gradient.json",
                    mime_type="application/json",
                ),
            ),
        )

        self.assertEqual(artifact.artifact_id, "preview-card")
        self.assertEqual(artifact.workspace_id, "workspace-main")
        self.assertEqual(artifact.metadata.tags, ("preview", "webgpu"))
        self.assertEqual(
            artifact.primary_content_reference.workspace_path,
            "sessions/run-1/gradient.png",
        )

    def test_content_reference_rejects_unsafe_workspace_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "safe relative paths"):
            ArtifactContentReference(
                reference_id="preview",
                locator=ArtifactContentLocator.WORKSPACE_FILE,
                workspace_path="../outside.png",
            )

    def test_status_transition_helpers_validate_lifecycle(self) -> None:
        self.assertTrue(
            can_transition_artifact_status(
                ArtifactStatus.DRAFT,
                ArtifactStatus.READY,
            )
        )
        self.assertEqual(
            get_allowed_artifact_status_transitions(ArtifactStatus.READY),
            (
                ArtifactStatus.SUPERSEDED,
                ArtifactStatus.FAILED,
                ArtifactStatus.ARCHIVED,
            ),
        )

        with self.assertRaisesRegex(ValueError, "ready -> draft"):
            validate_artifact_status_transition(
                ArtifactStatus.READY,
                ArtifactStatus.DRAFT,
            )

    def test_transition_artifact_status_updates_timestamps(self) -> None:
        created_at = datetime(2026, 5, 20, 8, 0, tzinfo=UTC)
        artifact = _artifact(
            timestamps=ArtifactTimestamps(
                created_at=created_at,
                updated_at=created_at,
            )
        )

        transitioned = transition_artifact_status(
            artifact,
            ArtifactStatus.READY,
            changed_at=created_at + timedelta(minutes=15),
        )

        self.assertEqual(transitioned.status, ArtifactStatus.READY)
        self.assertEqual(
            transitioned.timestamps.updated_at,
            created_at + timedelta(minutes=15),
        )
        self.assertEqual(
            transitioned.timestamps.status_changed_at,
            created_at + timedelta(minutes=15),
        )

    def test_workspace_groups_artifacts_and_workflow_runs(self) -> None:
        created_at = datetime(2026, 5, 20, 8, 0, tzinfo=UTC)
        request = AssistantRequest(
            query="Generate a WebGPU gradient study.",
            conversation_id="conversation-42",
            project_id="project-lab",
            domains=(CreativeCodingDomain.WEBGPU_WGSL,),
            mode=AssistantMode.GENERATE,
        )
        workspace = ArtifactWorkspace.from_request(
            request,
            workspace_id="workspace-main",
            session_id="session-7",
            created_at=created_at,
        )
        artifact = _artifact(
            timestamps=ArtifactTimestamps(
                created_at=created_at,
                updated_at=created_at + timedelta(minutes=5),
            ),
            workflow_link=ArtifactWorkflowLink.from_request(
                request,
                workflow_run_id="workflow-17",
                step="generation",
            ),
        )

        updated_workspace = attach_artifact_to_workspace(workspace, artifact)

        self.assertEqual(updated_workspace.conversation_id, "conversation-42")
        self.assertEqual(updated_workspace.project_id, "project-lab")
        self.assertEqual(updated_workspace.artifact_ids, ("preview-card",))
        self.assertEqual(updated_workspace.workflow_run_ids, ("workflow-17",))
        self.assertEqual(updated_workspace.artifact_count, 1)
        self.assertEqual(
            updated_workspace.updated_at,
            created_at + timedelta(minutes=5),
        )

    def test_workspace_rejects_foreign_artifact(self) -> None:
        workspace = ArtifactWorkspace(
            workspace_id="workspace-main",
            session_id="session-7",
            created_at=datetime(2026, 5, 20, 8, 0, tzinfo=UTC),
            updated_at=datetime(2026, 5, 20, 8, 0, tzinfo=UTC),
        )
        artifact = _artifact(
            identity=ArtifactIdentity(
                artifact_id="preview-card",
                workspace_id="workspace-other",
            )
        )

        with self.assertRaisesRegex(ValueError, "must match the target workspace"):
            attach_artifact_to_workspace(workspace, artifact)

    def test_workflow_link_from_request_carries_mode_and_domains(self) -> None:
        request = AssistantRequest(
            query="Preview the shader.",
            conversation_id="conversation-42",
            domains=(
                CreativeCodingDomain.GLSL,
                CreativeCodingDomain.WEBGPU_WGSL,
            ),
            mode=AssistantMode.PREVIEW,
        )

        link = ArtifactWorkflowLink.from_request(
            request,
            workflow_run_id="workflow-42",
            step="preview_render",
        )

        self.assertEqual(link.assistant_mode, AssistantMode.PREVIEW)
        self.assertEqual(link.conversation_id, "conversation-42")
        self.assertEqual(
            link.domains,
            (
                CreativeCodingDomain.GLSL,
                CreativeCodingDomain.WEBGPU_WGSL,
            ),
        )
        self.assertEqual(link.step, "preview_render")


def _artifact(
    *,
    identity: ArtifactIdentity | None = None,
    metadata: ArtifactMetadata | None = None,
    content_references: tuple[ArtifactContentReference, ...] = (),
    timestamps: ArtifactTimestamps | None = None,
    workflow_link: ArtifactWorkflowLink | None = None,
) -> ArtifactRecord:
    created_at = datetime(2026, 5, 20, 8, 0, tzinfo=UTC)
    return ArtifactRecord(
        identity=identity
        or ArtifactIdentity(
            artifact_id=" Preview-Card ",
            workspace_id=" Workspace-Main ",
        ),
        category=ArtifactCategory.PREVIEW,
        artifact_type=ArtifactType.IMAGE,
        metadata=metadata or ArtifactMetadata(summary="Primary preview output."),
        content_references=content_references,
        timestamps=timestamps
        or ArtifactTimestamps(
            created_at=created_at,
            updated_at=created_at,
        ),
        workflow_link=workflow_link,
    )


if __name__ == "__main__":
    unittest.main()
