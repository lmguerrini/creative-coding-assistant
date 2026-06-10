import io
import json
import unittest

from creative_coding_assistant.api import (
    AssistantStreamingApplication,
    AssistantStreamRequest,
    iter_assistant_stream_ndjson,
    serialize_stream_event,
)
from creative_coding_assistant.contracts import AssistantRequest, StreamEvent
from creative_coding_assistant.orchestration import StreamEventBuilder


class NextjsStreamingBridgeTests(unittest.TestCase):
    def test_stream_request_accepts_frontend_aliases(self) -> None:
        request = AssistantStreamRequest.model_validate(
            {
                "query": "Generate a WebGPU field.",
                "conversationId": "browser-session",
                "projectId": "workspace-a",
                "domain": "webgpu_wgsl",
                "mode": "generate",
                "artifactRefinement": {
                    "artifactId": "source-sketch",
                    "title": "aurora-field.p5.js",
                    "language": "p5.js",
                    "content": "function draw() { background(0); }",
                    "instruction": "Make this more organic.",
                    "domain": "p5_js",
                    "runtime": "p5",
                    "rendererId": "surface.p5",
                    "previewEligible": True,
                    "qualityScore": 0.91,
                    "qualityRank": 1,
                    "critiqueRationale": "Strong visual candidate.",
                    "refinementGuidance": "Soften the motion.",
                    "creativeTranslation": {
                        "outputModality": "visual",
                        "creativeIntent": "Create a calm particle field.",
                        "symbolicReferences": [],
                        "geometricReferences": [],
                        "musicalReferences": [],
                        "moodAtmosphere": ["calm"],
                        "movementLanguage": ["drift"],
                        "colorMaterialDirection": [],
                        "runtimeRecommendations": ["p5.js"],
                        "structureDirection": [],
                        "generationConstraints": [],
                        "refinementTargets": ["Preserve atmosphere: calm"],
                        "sacredGeometry": {
                            "concepts": ["mandala"],
                            "geometricStructure": ["Build nested rings."],
                            "symmetryType": ["Use radial symmetry."],
                            "movementBehavior": [],
                            "visualComposition": [],
                            "colorMaterialDirection": [],
                            "runtimeRecommendations": ["p5.js"],
                            "audioImplications": [],
                            "generationConstraints": [
                                "Do not add unsupported symbolic claims."
                            ],
                        },
                    },
                },
                "attachments": [
                    {
                        "type": "image",
                        "id": "image-reference-1",
                        "name": "palette.png",
                        "mimeType": "image/png",
                        "sizeBytes": 128,
                        "dataUrl": "data:image/png;base64,cGFsZXR0ZQ==",
                    }
                ],
            }
        )

        assistant_request = request.to_assistant_request()

        self.assertIsInstance(assistant_request, AssistantRequest)
        self.assertEqual(assistant_request.conversation_id, "browser-session")
        self.assertEqual(
            assistant_request.artifact_refinement.creative_translation[
                "creativeIntent"
            ],
            "Create a calm particle field.",
        )
        self.assertEqual(
            assistant_request.artifact_refinement.creative_translation[
                "sacredGeometry"
            ]["concepts"],
            ["mandala"],
        )
        self.assertEqual(assistant_request.project_id, "workspace-a")
        self.assertEqual(assistant_request.domain.value, "webgpu_wgsl")
        self.assertEqual(len(assistant_request.attachments), 1)
        self.assertEqual(assistant_request.attachments[0].name, "palette.png")
        self.assertEqual(assistant_request.attachments[0].mime_type, "image/png")
        self.assertIsNotNone(assistant_request.artifact_refinement)
        assert assistant_request.artifact_refinement is not None
        self.assertEqual(
            assistant_request.artifact_refinement.artifact_id,
            "source-sketch",
        )
        self.assertEqual(
            assistant_request.artifact_refinement.instruction,
            "Make this more organic.",
        )
        self.assertEqual(
            assistant_request.artifact_refinement.domain,
            "p5_js",
        )

    def test_stream_request_rejects_too_many_image_references(self) -> None:
        with self.assertRaisesRegex(ValueError, "Attach up to 4 image references"):
            AssistantStreamRequest.model_validate(
                {
                    "query": "Generate a WebGPU field.",
                    "attachments": [
                        {
                            "type": "image",
                            "id": f"image-reference-{index}",
                            "name": f"palette-{index}.png",
                            "mimeType": "image/png",
                            "sizeBytes": 128,
                            "dataUrl": "data:image/png;base64,cGFsZXR0ZQ==",
                        }
                        for index in range(5)
                    ],
                }
            )

    def test_stream_event_serialization_preserves_backend_shape(self) -> None:
        event = StreamEventBuilder().status(
            code="request_received",
            message="Request accepted.",
        )

        line = serialize_stream_event(event)
        parsed = json.loads(line)

        self.assertTrue(line.endswith("\n"))
        self.assertEqual(parsed["event_type"], "status")
        self.assertEqual(parsed["sequence"], 0)
        self.assertEqual(
            parsed["payload"]["code"],
            "request_received",
        )
        self.assertEqual(parsed["payload"]["message"], "Request accepted.")
        self.assertIn("emitted_at", parsed["payload"])

    def test_iter_stream_ndjson_emits_service_events(self) -> None:
        service = _FakeService(
            (
                StreamEventBuilder().status(
                    code="request_received",
                    message="Request accepted.",
                ),
            )
        )
        request = AssistantStreamRequest(query="Explain particles.")

        lines = tuple(
            iter_assistant_stream_ndjson(request=request, service=service)
        )

        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["payload"]["code"], "request_received")
        self.assertEqual(service.requests[0].query, "Explain particles.")

    def test_iter_stream_ndjson_emits_error_event_when_service_fails(self) -> None:
        service = _FailingService()
        request = AssistantStreamRequest(query="Generate.")

        lines = tuple(
            iter_assistant_stream_ndjson(request=request, service=service)
        )

        self.assertEqual(len(lines), 1)
        event = json.loads(lines[0])
        self.assertEqual(event["event_type"], "error")
        self.assertEqual(event["sequence"], 0)
        self.assertEqual(event["payload"]["code"], "assistant_stream_failed")
        self.assertTrue(event["payload"]["recoverable"])
        self.assertEqual(event["payload"]["subsystem"], "assistant_stream")

    def test_wsgi_endpoint_streams_ndjson(self) -> None:
        service = _FakeService(
            (
                StreamEventBuilder().status(
                    code="request_received",
                    message="Request accepted.",
                ),
                StreamEventBuilder().final(answer="Done."),
            )
        )
        app = AssistantStreamingApplication(service=service)
        payload = json.dumps({"query": "Generate a sketch."}).encode("utf-8")
        status_headers: dict[str, object] = {}

        body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/assistant/stream",
                    "REQUEST_METHOD": "POST",
                    "CONTENT_LENGTH": str(len(payload)),
                    "wsgi.input": io.BytesIO(payload),
                },
                _capture_start_response(status_headers),
            )
        )

        self.assertEqual(status_headers["status"], "200 OK")
        self.assertIn(
            ("Content-Type", "application/x-ndjson; charset=utf-8"),
            status_headers["headers"],
        )
        events = [json.loads(line) for line in body.decode("utf-8").splitlines()]
        self.assertEqual([event["event_type"] for event in events], ["status", "final"])
        self.assertEqual(events[-1]["payload"]["answer"], "Done.")

    def test_wsgi_endpoint_rejects_invalid_request(self) -> None:
        app = AssistantStreamingApplication(service=_FakeService(()))
        payload = json.dumps({"query": ""}).encode("utf-8")
        status_headers: dict[str, object] = {}

        body = b"".join(
            app(
                {
                    "PATH_INFO": "/api/assistant/stream",
                    "REQUEST_METHOD": "POST",
                    "CONTENT_LENGTH": str(len(payload)),
                    "wsgi.input": io.BytesIO(payload),
                },
                _capture_start_response(status_headers),
            )
        )

        self.assertEqual(status_headers["status"], "400 Bad Request")
        self.assertEqual(json.loads(body)["error"], "invalid_request")


def _capture_start_response(target: dict[str, object]):
    def start_response(
        status: str,
        headers: list[tuple[str, str]],
        exc_info: object | None = None,
    ) -> None:
        del exc_info
        target["status"] = status
        target["headers"] = headers

    return start_response


class _FakeService:
    def __init__(self, events: tuple[StreamEvent, ...]) -> None:
        self._events = events
        self.requests: list[AssistantRequest] = []

    def stream(self, request: AssistantRequest):
        self.requests.append(request)
        return iter(self._events)


class _FailingService:
    def stream(self, request: AssistantRequest):
        del request
        raise RuntimeError("boom")


if __name__ == "__main__":
    unittest.main()
