import json
import unittest
from types import SimpleNamespace

from creative_coding_assistant.api.streaming import AssistantStreamRequest
from creative_coding_assistant.contracts import (
    MAX_IMAGE_REFERENCE_BYTES,
    AssistantImageReference,
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.llm import OpenAIGenerationProvider
from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    LlmGenerationAdapter,
    RouteCapability,
    RouteDecision,
    RouteName,
    StructuredPromptInputBuilder,
    build_prompt_input_request,
    build_provider_generation_request,
    build_rendered_prompt_request,
)
from creative_coding_assistant.security import provider_input_summary

_IMAGE_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42Y"
    "AAAAASUVORK5CYII="
)


class MultimodalProviderInputTests(unittest.TestCase):
    def test_image_bytes_reach_openai_as_real_input_image_beside_prompt(self) -> None:
        browser_request = AssistantStreamRequest.model_validate(
            {
                "query": "Use the attached image as palette guidance.",
                "domain": "p5_js",
                "mode": "generate",
                "attachments": [
                    {
                        "type": "image",
                        "id": "image-reference-1",
                        "name": "palette.png",
                        "mimeType": "image/png",
                        "sizeBytes": 68,
                        "dataUrl": _IMAGE_DATA_URL,
                    }
                ],
            }
        )
        generation_input = _generation_input(
            attachments=browser_request.to_assistant_request().attachments
        )

        self.assertEqual(len(generation_input.image_inputs), 1)
        self.assertEqual(
            generation_input.image_inputs[0].data_url.get_secret_value(),
            _IMAGE_DATA_URL,
        )
        self.assertNotIn(_IMAGE_DATA_URL, repr(generation_input))
        self.assertNotIn(_IMAGE_DATA_URL, generation_input.model_dump_json())
        user_message = next(
            message
            for message in generation_input.messages
            if message.name.value == "user"
        )
        self.assertIn("visual input attached", user_message.content)
        self.assertNotIn(_IMAGE_DATA_URL, user_message.content)

        client = _FakeOpenAIClient()
        events = tuple(
            OpenAIGenerationProvider(client=client, model="gpt-5-mini").stream(
                generation_input
            )
        )

        self.assertEqual(len(events), 1)
        provider_user_message = next(
            item for item in client.last_kwargs["input"] if item["role"] == "user"
        )
        self.assertEqual(
            [content["type"] for content in provider_user_message["content"]],
            ["input_text", "input_image"],
        )
        self.assertIn(
            "Use the attached image as palette guidance.",
            provider_user_message["content"][0]["text"],
        )
        self.assertEqual(
            provider_user_message["content"][1],
            {
                "type": "input_image",
                "image_url": _IMAGE_DATA_URL,
                "detail": "auto",
            },
        )

        diagnostic_summary = provider_input_summary(generation_input)
        self.assertEqual(diagnostic_summary["image_input_count"], 1)
        self.assertEqual(
            diagnostic_summary["image_inputs"],
            [
                {
                    "id": "image-reference-1",
                    "name": "palette.png",
                    "mime_type": "image/png",
                    "size_bytes": 68,
                }
            ],
        )
        self.assertNotIn(_IMAGE_DATA_URL, json.dumps(diagnostic_summary))

    def test_text_only_and_metadata_only_requests_do_not_send_image_blocks(
        self,
    ) -> None:
        for attachments, expected_prompt_marker in (
            ((), None),
            (
                (
                    AssistantImageReference(
                        id="metadata-reference",
                        name="palette.png",
                        mimeType="image/png",
                        sizeBytes=7,
                    ),
                ),
                "metadata only; no pixels attached",
            ),
        ):
            with self.subTest(attachment_count=len(attachments)):
                generation_input = _generation_input(attachments=attachments)
                self.assertEqual(generation_input.image_inputs, ())
                user_message = next(
                    message
                    for message in generation_input.messages
                    if message.name.value == "user"
                )
                if expected_prompt_marker is None:
                    self.assertNotIn("Image References:", user_message.content)
                else:
                    self.assertIn(expected_prompt_marker, user_message.content)

                client = _FakeOpenAIClient()
                tuple(
                    OpenAIGenerationProvider(
                        client=client,
                        model="gpt-5-mini",
                    ).stream(generation_input)
                )
                provider_user_message = next(
                    item
                    for item in client.last_kwargs["input"]
                    if item["role"] == "user"
                )
                self.assertEqual(
                    [
                        content["type"]
                        for content in provider_user_message["content"]
                    ],
                    ["input_text"],
                )

    def test_backend_rejects_image_payloads_outside_validated_boundaries(self) -> None:
        with self.assertRaisesRegex(ValueError, "size must match"):
            AssistantImageReference(
                id="mismatched-image",
                name="palette.png",
                mimeType="image/png",
                sizeBytes=8,
                dataUrl=_IMAGE_DATA_URL,
            )

        with self.assertRaisesRegex(ValueError, "bytes must match"):
            AssistantImageReference(
                id="forged-image",
                name="palette.png",
                mimeType="image/png",
                sizeBytes=7,
                dataUrl="data:image/png;base64,cGFsZXR0ZQ==",
            )

        with self.assertRaises(ValueError):
            AssistantImageReference(
                id="oversized-image",
                name="large.png",
                mimeType="image/png",
                sizeBytes=MAX_IMAGE_REFERENCE_BYTES + 1,
            )


def _generation_input(
    *,
    attachments: tuple[AssistantImageReference, ...],
):
    route_decision = RouteDecision(
        route=RouteName.GENERATE,
        mode=AssistantMode.GENERATE,
        domain=CreativeCodingDomain.P5_JS,
        domains=(CreativeCodingDomain.P5_JS,),
        capabilities=(RouteCapability.TOOL_USE,),
    )
    assistant_request = AssistantRequest(
        query="Use the attached image as palette guidance.",
        domain=CreativeCodingDomain.P5_JS,
        mode=AssistantMode.GENERATE,
        attachments=attachments,
    )
    prompt_input = StructuredPromptInputBuilder().build(
        build_prompt_input_request(
            assistant_request=assistant_request,
            route_decision=route_decision,
            assembled_context=None,
        )
    )
    rendered_prompt = JinjaPromptRenderer().render(
        build_rendered_prompt_request(
            route_decision=route_decision,
            prompt_input=prompt_input,
        )
    )
    return LlmGenerationAdapter().prepare_generation(
        build_provider_generation_request(
            route_decision=route_decision,
            rendered_prompt=rendered_prompt,
            stream=False,
        )
    )


class _FakeResponsesApi:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.last_kwargs = dict(kwargs)
        return SimpleNamespace(
            id="resp_multimodal",
            model="gpt-5-mini",
            output_text="Generated from the visual reference.",
            status="completed",
        )


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeResponsesApi()

    @property
    def last_kwargs(self) -> dict[str, object]:
        assert self.responses.last_kwargs is not None
        return self.responses.last_kwargs


if __name__ == "__main__":
    unittest.main()
