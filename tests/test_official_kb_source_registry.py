import unittest

from pydantic import ValidationError

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag import (
    OFFICIAL_HOSTS_BY_DOMAIN,
    OfficialSource,
    OfficialSourceType,
    SourceApprovalStatus,
    approved_official_sources,
    approved_sources_for_domain,
    get_official_source,
    official_source_domains,
)


class OfficialKnowledgeBaseSourceRegistryTests(unittest.TestCase):
    def test_registry_covers_all_v1_domains(self) -> None:
        self.assertEqual(
            official_source_domains(),
            (
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.P5_JS,
                CreativeCodingDomain.GLSL,
            ),
        )

    def test_sources_are_explicitly_approved(self) -> None:
        sources = approved_official_sources()

        self.assertGreater(len(sources), 0)
        for source in sources:
            self.assertEqual(source.approval_status, SourceApprovalStatus.APPROVED)
            self.assertEqual(source.url.startswith("https://"), True)
            self.assertGreaterEqual(source.priority, 1)
            self.assertGreater(len(source.allowed_path_prefixes), 0)

    def test_source_ids_are_unique(self) -> None:
        source_ids = [source.source_id for source in approved_official_sources()]

        self.assertEqual(len(source_ids), len(set(source_ids)))

    def test_sources_for_domain_are_priority_sorted(self) -> None:
        sources = approved_sources_for_domain(CreativeCodingDomain.THREE_JS)

        self.assertEqual(
            [source.source_id for source in sources],
            ["three_docs", "three_manual", "three_examples"],
        )

    def test_source_lookup_returns_registered_source(self) -> None:
        source = get_official_source("p5_reference")

        self.assertEqual(source.domain, CreativeCodingDomain.P5_JS)
        self.assertEqual(source.source_type, OfficialSourceType.API_REFERENCE)

    def test_source_lookup_rejects_unknown_source(self) -> None:
        with self.assertRaises(ValueError):
            get_official_source("unknown_source")

    def test_registry_restricts_hosts_by_domain(self) -> None:
        for source in approved_official_sources():
            hosts = OFFICIAL_HOSTS_BY_DOMAIN[source.domain]

            self.assertIn(source.url.split("/")[2], hosts)

    def test_source_model_rejects_unapproved_host(self) -> None:
        with self.assertRaises(ValidationError):
            OfficialSource(
                source_id="bad_three_source",
                domain=CreativeCodingDomain.THREE_JS,
                title="Bad Three Source",
                publisher="Not Official",
                url="https://example.com/docs/",
                source_type=OfficialSourceType.API_REFERENCE,
                priority=1,
                allowed_path_prefixes=("/docs/",),
            )

    def test_source_model_rejects_out_of_scope_path(self) -> None:
        with self.assertRaises(ValidationError):
            OfficialSource(
                source_id="bad_three_path",
                domain=CreativeCodingDomain.THREE_JS,
                title="Bad Three Path",
                publisher="three.js",
                url="https://threejs.org/examples/",
                source_type=OfficialSourceType.EXAMPLES,
                priority=1,
                allowed_path_prefixes=("/docs/",),
            )

    def test_glsl_sources_use_khronos_registry(self) -> None:
        sources = approved_sources_for_domain(CreativeCodingDomain.GLSL)

        self.assertEqual(
            {source.url.split("/")[2] for source in sources},
            {"registry.khronos.org"},
        )


if __name__ == "__main__":
    unittest.main()
