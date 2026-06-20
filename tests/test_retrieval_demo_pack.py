import unittest

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.eval import (
    CCA_OPERATIONAL_KB_SCOPE,
    FUTURE_HOLOMIND_BOUNDARY,
    build_capstone_retrieval_demo_pack,
    build_capstone_retrieval_demo_requests,
    capstone_retrieval_demo_source_ids,
)


class RetrievalDemoPackTests(unittest.TestCase):
    def test_demo_pack_frames_operational_kb_boundary(self) -> None:
        pack = build_capstone_retrieval_demo_pack()

        self.assertEqual(pack.operational_scope, CCA_OPERATIONAL_KB_SCOPE)
        self.assertEqual(pack.holomind_boundary, FUTURE_HOLOMIND_BOUNDARY)
        self.assertEqual(pack.pack_id, "capstone_kb_expansion_retrieval_demo_pack")
        self.assertGreaterEqual(len(pack.scenarios), 6)

    def test_demo_pack_uses_registered_source_ids_and_nonempty_domains(self) -> None:
        pack = build_capstone_retrieval_demo_pack()

        for scenario in pack.scenarios:
            with self.subTest(demo_id=scenario.demo_id):
                self.assertGreater(len(scenario.domains), 0)
                self.assertGreater(len(scenario.expected_source_ids), 0)
                for source_id in scenario.expected_source_ids:
                    self.assertIn(source_id, capstone_retrieval_demo_source_ids())

    def test_demo_sync_source_ids_are_unique_and_ordered_by_first_use(self) -> None:
        self.assertEqual(
            capstone_retrieval_demo_source_ids(),
            (
                "hydra_docs",
                "p5_reference",
                "p5_sound_analysis_reference",
                "web_audio_analyser_node",
                "web_audio_visualization_guide",
                "tone_js_analysis_reference",
                "three_manual_effects",
                "glsl_mdn_webgl_examples",
                "shadertoy_howto",
                "tone_js_docs",
                "three_manual",
                "p5_sound_reference",
            ),
        )

    def test_demo_requests_preserve_query_and_domain_filters(self) -> None:
        requests = build_capstone_retrieval_demo_requests(limit=4)

        self.assertTrue(all(request.limit == 4 for request in requests))
        self.assertEqual(
            requests[0].filters.domains,
            (CreativeCodingDomain.HYDRA, CreativeCodingDomain.P5_JS),
        )
        self.assertEqual(
            requests[-1].query,
            (
                "Turn a concentric mandala motif into a practical browser visual "
                "system with motion, rhythm, and runtime choices."
            ),
        )
        self.assertEqual(
            requests[-1].filters.domains,
            (
                CreativeCodingDomain.P5_JS,
                CreativeCodingDomain.GLSL,
                CreativeCodingDomain.TONE_JS,
                CreativeCodingDomain.THREE_JS,
            ),
        )


if __name__ == "__main__":
    unittest.main()
