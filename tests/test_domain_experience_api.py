import io
import json
import tempfile
import unittest
from pathlib import Path

from creative_coding_assistant.api.domain_experience import (
    DOMAIN_EXPERIENCE_CONTRACT_VERSION,
    DomainExperienceApplication,
    build_domain_experience_payload,
)
from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.core.config import Settings
from creative_coding_assistant.domains import (
    DomainDeliveryKind,
    domain_experience_records,
    get_domain_experience,
)


class DomainExperienceRegistryTests(unittest.TestCase):
    def test_every_registered_domain_has_one_public_contract(self) -> None:
        records = domain_experience_records()

        self.assertEqual(
            tuple(record.domain for record in records), tuple(CreativeCodingDomain)
        )
        self.assertTrue(all(record.intent_triggers for record in records))
        self.assertTrue(all(record.filename_extensions for record in records))
        self.assertTrue(all(record.public_claim_boundary for record in records))

    def test_only_the_validated_generation_contracts_claim_live_preview(self) -> None:
        live_domains = {
            record.domain for record in domain_experience_records() if record.live_preview
        }

        self.assertEqual(
            live_domains,
            {
                CreativeCodingDomain.P5_JS,
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.GLSL,
            },
        )
        self.assertEqual(
            get_domain_experience(CreativeCodingDomain.HYDRA).delivery_kind,
            DomainDeliveryKind.CODE_EXPORT,
        )
        self.assertFalse(get_domain_experience("hydra").live_preview)
        self.assertEqual(
            get_domain_experience("react_three_fiber").delivery_kind,
            DomainDeliveryKind.CODE_EXPORT,
        )
        self.assertEqual(
            get_domain_experience("touchdesigner").delivery_kind,
            DomainDeliveryKind.EXTERNAL_HANDOFF,
        )


class DomainExperienceApiTests(unittest.TestCase):
    def test_payload_distinguishes_registered_from_indexed_knowledge(self) -> None:
        payload = build_domain_experience_payload(
            chroma_sqlite_path=Path("/private/tmp/not-a-cca-kb/chroma.sqlite3")
        )

        knowledge = payload["knowledgeBase"]
        assert isinstance(knowledge, dict)
        self.assertEqual(payload["schemaVersion"], DOMAIN_EXPERIENCE_CONTRACT_VERSION)
        self.assertEqual(knowledge["status"], "not_initialized")
        self.assertGreater(knowledge["registeredSourceCount"], 0)
        self.assertEqual(knowledge["registeredSourceCount"], 57)
        self.assertEqual(knowledge["registeredDomainCount"], 43)
        self.assertEqual(knowledge["indexedSourceCount"], 0)
        self.assertEqual(knowledge["indexedChunkCount"], 0)
        self.assertIn("registered sources are not the same", knowledge["detail"])
        self.assertEqual(knowledge["freshnessStatus"], "not_reported")
        self.assertIn("freshness is not reported", knowledge["freshnessDetail"])
        self.assertEqual(knowledge["updateStatus"], "explicit_selected_source_actions")
        self.assertIn("explicit confirmation", knowledge["updateHint"])
        creative_knowledge = payload["creativeKnowledge"]
        assert isinstance(creative_knowledge, dict)
        self.assertEqual(creative_knowledge["status"], "available")
        self.assertEqual(creative_knowledge["recordCount"], 7)
        self.assertEqual(len(creative_knowledge["records"]), 7)
        self.assertIn("without fetching external sources", creative_knowledge["authorityBoundary"])
        first_record = creative_knowledge["records"][0]
        self.assertIn("title", first_record)
        self.assertIn("confidence", first_record)
        self.assertNotIn("secret-user-request", json.dumps(creative_knowledge).lower())

    def test_wsgi_endpoint_is_read_only_and_public_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = Settings(chroma_persist_dir=Path(temporary_directory) / "chroma")
            app = DomainExperienceApplication(settings_factory=lambda: settings)
            captured: dict[str, object] = {}

            body = b"".join(
                app(
                    {
                        "PATH_INFO": "/api/domain-experience",
                        "REQUEST_METHOD": "GET",
                        "wsgi.input": io.BytesIO(),
                    },
                    _capture_start_response(captured),
                )
            )

            payload = json.loads(body)
            self.assertEqual(captured["status"], "200 OK")
            self.assertEqual(payload["schemaVersion"], DOMAIN_EXPERIENCE_CONTRACT_VERSION)
            self.assertEqual(len(payload["domains"]), len(CreativeCodingDomain))
            self.assertNotIn("secret-user-request", json.dumps(payload).lower())
            self.assertNotIn("private-memory-value", json.dumps(payload).lower())
            self.assertFalse((Path(temporary_directory) / "chroma").exists())


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


if __name__ == "__main__":
    unittest.main()
