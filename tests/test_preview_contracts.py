import unittest
from datetime import UTC, datetime, timedelta

from creative_coding_assistant.artifacts import (
    ArtifactCategory,
    ArtifactContentLocator,
    ArtifactContentReference,
    ArtifactIdentity,
    ArtifactMetadata,
    ArtifactRecord,
    ArtifactTimestamps,
    ArtifactType,
    ArtifactWorkflowLink,
)
from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEventType,
)
from creative_coding_assistant.orchestration import StreamEventBuilder
from creative_coding_assistant.preview import (
    NoOpPreviewRenderer,
    PreviewRendererCapability,
    PreviewRequest,
    PreviewResult,
    PreviewStatus,
    PreviewTarget,
    can_transition_preview_status,
    get_allowed_preview_status_transitions,
    is_previewable_artifact,
    resolve_preview_target_for_artifact,
    validate_preview_status_transition,
)


class PreviewContractsTests(unittest.TestCase):
    def test_preview_request_from_artifact_maps_web_code_to_browser_target(
        self,
    ) -> None:
        artifact = _artifact()
        request = PreviewRequest.from_artifact(
            artifact,
            request_id="preview-request-1",
            preview_id="preview-run-1",
            requested_at=_timestamp(0),
        )

        self.assertEqual(request.preview_id, "preview-run-1")
        self.assertEqual(request.artifact_id, artifact.artifact_id)
        self.assertEqual(request.identity.workspace_id, artifact.workspace_id)
        self.assertEqual(request.target, PreviewTarget.BROWSER_SANDBOX)
        self.assertEqual(
            request.artifact_link.content_reference_id,
            "sketch-entry",
        )
        self.assertEqual(request.workflow_link, artifact.workflow_link)

    def test_preview_detection_maps_assets_and_rejects_unpreviewable_code(self) -> None:
        image_artifact = _artifact(
            artifact_type=ArtifactType.IMAGE,
            metadata=ArtifactMetadata(domain=CreativeCodingDomain.THREE_JS),
            content_references=(
                ArtifactContentReference(
                    reference_id="preview-image",
                    locator=ArtifactContentLocator.WORKSPACE_FILE,
                    workspace_path="outputs/frame.png",
                    mime_type="image/png",
                    is_primary=True,
                ),
            ),
        )
        unsupported_artifact = _artifact(
            metadata=ArtifactMetadata(language="python"),
        )

        self.assertEqual(
            resolve_preview_target_for_artifact(image_artifact),
            PreviewTarget.IMAGE_ASSET,
        )
        self.assertTrue(is_previewable_artifact(image_artifact))
        self.assertIsNone(resolve_preview_target_for_artifact(unsupported_artifact))
        self.assertFalse(is_previewable_artifact(unsupported_artifact))

    def test_preview_status_transition_helpers_validate_lifecycle(self) -> None:
        self.assertTrue(
            can_transition_preview_status(
                PreviewStatus.PENDING,
                PreviewStatus.RUNNING,
            )
        )
        self.assertEqual(
            get_allowed_preview_status_transitions(PreviewStatus.RUNNING),
            (
                PreviewStatus.SUCCEEDED,
                PreviewStatus.FAILED,
            ),
        )

        with self.assertRaisesRegex(ValueError, "succeeded -> skipped"):
            validate_preview_status_transition(
                PreviewStatus.SUCCEEDED,
                PreviewStatus.SKIPPED,
            )

    def test_preview_result_rejects_non_terminal_status(self) -> None:
        request = _preview_request()

        with self.assertRaisesRegex(ValueError, "terminal preview status"):
            PreviewResult(
                request=request,
                status=PreviewStatus.RUNNING,
            )

    def test_renderer_capability_matches_request(self) -> None:
        request = _preview_request()
        capability = PreviewRendererCapability(
            renderer_id="renderer.browser",
            display_name="Browser Renderer",
            supported_targets=(PreviewTarget.BROWSER_SANDBOX,),
            supported_artifact_types=(ArtifactType.CODE,),
            supported_domains=(CreativeCodingDomain.WEBGPU_WGSL,),
            tags=("Preview", "browser"),
        )
        mismatched_capability = PreviewRendererCapability(
            renderer_id="renderer.image",
            display_name="Image Renderer",
            supported_targets=(PreviewTarget.IMAGE_ASSET,),
        )

        self.assertTrue(capability.supports_request(request))
        self.assertEqual(capability.tags, ("preview", "browser"))
        self.assertFalse(mismatched_capability.supports_request(request))

    def test_noop_renderer_returns_skipped_preview_result(self) -> None:
        renderer = NoOpPreviewRenderer()
        request = _preview_request()

        result = renderer.render(request)

        self.assertEqual(result.status, PreviewStatus.SKIPPED)
        self.assertEqual(result.preview_id, request.preview_id)
        self.assertEqual(
            result.provenance.renderer_id,
            renderer.capability.renderer_id,
        )
        self.assertIn("foundation", result.summary)

    def test_stream_event_builder_serializes_preview_results(self) -> None:
        request = _preview_request()
        result = PreviewResult.succeeded(
            request=request,
            preview_artifact_id="preview-artifact-1",
            summary="Preview artifact is ready.",
            completed_at=_timestamp(5),
        )
        builder = StreamEventBuilder()

        event = builder.preview_artifact(result, phase="preview")

        self.assertEqual(event.sequence, 0)
        self.assertEqual(event.event_type, StreamEventType.PREVIEW_ARTIFACT)
        self.assertEqual(event.payload["status"], "succeeded")
        self.assertEqual(event.payload["preview_id"], "preview-run-1")
        self.assertEqual(event.payload["artifact_id"], "source-sketch")
        self.assertEqual(event.payload["phase"], "preview")
        self.assertEqual(
            event.payload["result"]["preview_artifact_id"],
            "preview-artifact-1",
        )


def _preview_request() -> PreviewRequest:
    return PreviewRequest.from_artifact(
        _artifact(),
        request_id="preview-request-1",
        preview_id="preview-run-1",
        requested_at=_timestamp(0),
    )


def _artifact(
    *,
    artifact_type: ArtifactType = ArtifactType.CODE,
    metadata: ArtifactMetadata | None = None,
    content_references: tuple[ArtifactContentReference, ...] | None = None,
) -> ArtifactRecord:
    resolved_metadata = metadata or ArtifactMetadata(
        domain=CreativeCodingDomain.WEBGPU_WGSL,
        language="TypeScript",
        tags=("Preview", "webgpu"),
    )
    request_domains = (
        (resolved_metadata.domain,)
        if resolved_metadata.domain is not None
        else (CreativeCodingDomain.WEBGPU_WGSL,)
    )
    request = AssistantRequest(
        query="Preview the WebGPU sketch.",
        conversation_id="conversation-99",
        project_id="project-preview",
        domains=request_domains,
        mode=AssistantMode.PREVIEW,
    )
    return ArtifactRecord(
        identity=ArtifactIdentity(
            artifact_id="source-sketch",
            workspace_id="workspace-preview",
        ),
        category=ArtifactCategory.GENERATED,
        artifact_type=artifact_type,
        metadata=resolved_metadata,
        content_references=content_references
        or (
            ArtifactContentReference(
                reference_id="sketch-entry",
                locator=ArtifactContentLocator.WORKSPACE_FILE,
                workspace_path="outputs/sketch.ts",
                mime_type="text/typescript",
                is_primary=True,
            ),
        ),
        timestamps=ArtifactTimestamps(
            created_at=_timestamp(0),
            updated_at=_timestamp(1),
        ),
        workflow_link=ArtifactWorkflowLink.from_request(
            request,
            workflow_run_id="workflow-preview-1",
            step="generation",
        ),
    )


def _timestamp(minutes: int) -> datetime:
    return datetime(2026, 5, 20, 9, 0, tzinfo=UTC) + timedelta(minutes=minutes)


if __name__ == "__main__":
    unittest.main()
