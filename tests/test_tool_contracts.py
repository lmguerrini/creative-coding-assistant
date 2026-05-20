import unittest

from creative_coding_assistant.orchestration import RouteCapability, StreamEventBuilder
from creative_coding_assistant.tools import (
    DuplicateToolRegistrationError,
    ToolIdentity,
    ToolMetadata,
    ToolNotRegisteredError,
    ToolRegistry,
    ToolRequest,
    ToolResult,
    ToolStatus,
    can_transition_tool_status,
    get_allowed_tool_status_transitions,
    validate_tool_status_transition,
)


class ToolContractsTests(unittest.TestCase):
    def test_tool_identity_normalizes_lowercase_name(self) -> None:
        identity = ToolIdentity(name="  Preview.Render  ")

        self.assertEqual(identity.name, "preview.render")

    def test_tool_metadata_defaults_to_tool_use_capability(self) -> None:
        metadata = _tool_metadata()

        self.assertEqual(
            metadata.required_capabilities,
            (RouteCapability.TOOL_USE,),
        )
        self.assertEqual(metadata.tags, ("preview", "render"))

    def test_tool_result_rejects_non_terminal_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "terminal tool status"):
            ToolResult(
                request=_tool_request(),
                status=ToolStatus.RUNNING,
            )

    def test_failed_tool_result_requires_error_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "error payload"):
            ToolResult(
                request=_tool_request(),
                status=ToolStatus.FAILED,
            )

    def test_status_transition_helpers_validate_lifecycle(self) -> None:
        self.assertTrue(
            can_transition_tool_status(
                ToolStatus.PENDING,
                ToolStatus.RUNNING,
            )
        )
        self.assertEqual(
            get_allowed_tool_status_transitions(ToolStatus.RUNNING),
            (ToolStatus.SUCCEEDED, ToolStatus.FAILED),
        )

        with self.assertRaisesRegex(ValueError, "succeeded -> skipped"):
            validate_tool_status_transition(
                ToolStatus.SUCCEEDED,
                ToolStatus.SKIPPED,
            )

    def test_registry_registers_and_invokes_tool(self) -> None:
        request = _tool_request()
        registry = ToolRegistry((_EchoTool(),))

        result = registry.invoke(request)

        self.assertEqual(result.status, ToolStatus.SUCCEEDED)
        self.assertEqual(result.output["echo"], request.arguments["prompt"])
        self.assertTrue(registry.is_registered("preview.render"))
        self.assertEqual(registry.list_metadata()[0].name, "preview.render")

    def test_registry_rejects_duplicate_registration(self) -> None:
        registry = ToolRegistry((_EchoTool(),))

        with self.assertRaises(DuplicateToolRegistrationError):
            registry.register(_EchoTool())

    def test_registry_rejects_unknown_tools(self) -> None:
        registry = ToolRegistry()

        with self.assertRaises(ToolNotRegisteredError):
            registry.get("missing_tool")

    def test_stream_event_builder_serializes_tool_contracts(self) -> None:
        request = _tool_request()
        result = ToolResult.succeeded(
            request=request,
            output={"artifact_id": "preview-1"},
        )
        builder = StreamEventBuilder()

        start_event = builder.tool_start(request, phase="preview")
        result_event = builder.tool_result(result, phase="preview")

        self.assertEqual(start_event.sequence, 0)
        self.assertEqual(start_event.event_type.value, "tool_start")
        self.assertEqual(start_event.payload["status"], "running")
        self.assertEqual(start_event.payload["tool_name"], "preview.render")
        self.assertEqual(start_event.payload["phase"], "preview")
        self.assertEqual(
            start_event.payload["request"]["tool"]["required_capabilities"],
            ["tool_use"],
        )
        self.assertEqual(result_event.sequence, 1)
        self.assertEqual(result_event.event_type.value, "tool_result")
        self.assertEqual(result_event.payload["status"], "succeeded")
        self.assertEqual(
            result_event.payload["result"]["output"]["artifact_id"],
            "preview-1",
        )


class _EchoTool:
    @property
    def metadata(self) -> ToolMetadata:
        return _tool_metadata()

    def invoke(self, request: ToolRequest) -> ToolResult:
        return ToolResult.succeeded(
            request=request,
            output={"echo": request.arguments["prompt"]},
        )


def _tool_metadata() -> ToolMetadata:
    return ToolMetadata(
        identity=ToolIdentity(name="preview.render"),
        display_name="Preview Render",
        description="Render a preview artifact for one sketch request.",
        tags=("Preview", "render", "render"),
    )


def _tool_request() -> ToolRequest:
    return ToolRequest(
        request_id="tool-request-1",
        tool=_tool_metadata(),
        arguments={"prompt": "Render a calm WebGPU gradient preview."},
    )


if __name__ == "__main__":
    unittest.main()
