import unittest

from creative_coding_assistant import __version__
from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEvent,
    StreamEventType,
)
from creative_coding_assistant.core import load_settings
from creative_coding_assistant.domains import get_supported_domain_values
from creative_coding_assistant.vectorstore import collection_names


class BootstrapImportTests(unittest.TestCase):
    def test_package_imports(self) -> None:
        self.assertEqual(__version__, "0.1.0")

    def test_settings_defaults_point_to_root_runtime_dirs(self) -> None:
        settings = load_settings()

        self.assertEqual(settings.chroma_persist_dir.as_posix(), "data/chroma")
        self.assertEqual(settings.artifact_dir.as_posix(), "data/artifacts")

    def test_request_contract_validates_empty_query(self) -> None:
        with self.assertRaises(ValueError):
            AssistantRequest(query=" ")

    def test_request_contract_accepts_domain_and_mode(self) -> None:
        request = AssistantRequest(
            query="Create a Three.js particle sketch.",
            domain=CreativeCodingDomain.THREE_JS,
            mode=AssistantMode.GENERATE,
        )

        self.assertEqual(request.domain, CreativeCodingDomain.THREE_JS)
        self.assertEqual(request.domains, (CreativeCodingDomain.THREE_JS,))
        self.assertEqual(request.mode, AssistantMode.GENERATE)

    def test_request_contract_accepts_empty_domains(self) -> None:
        request = AssistantRequest(
            query="Help me choose the right domain.",
            domains=(),
        )

        self.assertIsNone(request.domain)
        self.assertEqual(request.domains, ())

    def test_request_contract_syncs_single_domain_list_to_legacy_field(self) -> None:
        request = AssistantRequest(
            query="Explain this shader setup.",
            domains=(CreativeCodingDomain.GLSL,),
        )

        self.assertEqual(request.domain, CreativeCodingDomain.GLSL)
        self.assertEqual(request.domains, (CreativeCodingDomain.GLSL,))

    def test_request_contract_accepts_multiple_domains(self) -> None:
        request = AssistantRequest(
            query="How do R3F and GLSL work together?",
            domains=(
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
        )

        self.assertIsNone(request.domain)
        self.assertEqual(
            request.domains,
            (
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
        )

    def test_request_contract_rejects_misaligned_domain_and_domains(self) -> None:
        with self.assertRaisesRegex(ValueError, "domain must be included in domains"):
            AssistantRequest(
                query="Explain this setup.",
                domain=CreativeCodingDomain.THREE_JS,
                domains=(CreativeCodingDomain.GLSL,),
            )

    def test_stream_event_contract_validates_sequence(self) -> None:
        event = StreamEvent(
            event_type=StreamEventType.STATUS,
            sequence=0,
            payload={"message": "ready"},
        )

        self.assertEqual(event.payload["message"], "ready")

    def test_chroma_collection_names_are_separated_by_concern(self) -> None:
        self.assertEqual(
            collection_names(),
            (
                "kb_official_docs",
                "conversation_turns",
                "conversation_summaries",
                "project_memory",
                "eval_traces",
                "preview_artifacts_index",
            ),
        )

    def test_active_domain_values_match_v1_scope(self) -> None:
        self.assertEqual(
            get_supported_domain_values(),
            (
                "three_js",
                "react_three_fiber",
                "p5_js",
                "glsl",
            ),
        )


if __name__ == "__main__":
    unittest.main()
