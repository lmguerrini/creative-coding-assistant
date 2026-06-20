import unittest

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag import (
    OfficialSourceType,
    approved_sources_for_domain,
    get_official_source,
)


class CapstoneKnowledgeBaseSourcePackTests(unittest.TestCase):
    def test_threejs_domain_includes_effects_and_render_pipeline_manual(self) -> None:
        source = get_official_source("three_manual_effects")

        self.assertEqual(source.domain, CreativeCodingDomain.THREE_JS)
        self.assertEqual(source.source_type, OfficialSourceType.GUIDE)
        self.assertEqual(
            source.additional_urls,
            (
                "https://threejs.org/manual/en/shadows.html",
                "https://threejs.org/manual/en/rendertargets.html",
            ),
        )
        self.assertEqual(
            tuple(
                item.source_id
                for item in approved_sources_for_domain(CreativeCodingDomain.THREE_JS)
            ),
            (
                "three_docs",
                "three_manual",
                "three_manual_effects",
                "three_examples",
            ),
        )

    def test_web_audio_domain_adds_analysis_and_debugging_guidance(self) -> None:
        analyser = get_official_source("web_audio_analyser_node")
        visualization = get_official_source("web_audio_visualization_guide")

        self.assertEqual(analyser.domain, CreativeCodingDomain.WEB_AUDIO_API)
        self.assertEqual(analyser.source_type, OfficialSourceType.API_REFERENCE)
        self.assertEqual(
            analyser.url,
            "https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode",
        )

        self.assertEqual(
            visualization.additional_urls,
            (
                "https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Using_Web_Audio_API",
                "https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Best_practices",
            ),
        )
        self.assertEqual(
            tuple(
                item.source_id
                for item in approved_sources_for_domain(
                    CreativeCodingDomain.WEB_AUDIO_API
                )
            ),
            (
                "web_audio_mdn_api",
                "web_audio_analyser_node",
                "web_audio_visualization_guide",
            ),
        )

    def test_audio_reactive_reference_pack_expands_tone_and_p5_sound(self) -> None:
        tone = get_official_source("tone_js_analysis_reference")
        p5_sound = get_official_source("p5_sound_analysis_reference")

        self.assertEqual(tone.domain, CreativeCodingDomain.TONE_JS)
        self.assertEqual(tone.source_type, OfficialSourceType.API_REFERENCE)
        self.assertEqual(
            tone.additional_urls,
            (
                "https://tonejs.github.io/docs/15.1.22/classes/Meter.html",
                "https://tonejs.github.io/docs/15.1.22/classes/Loop.html",
                "https://tonejs.github.io/docs/15.1.22/classes/Player.html",
            ),
        )

        self.assertEqual(p5_sound.domain, CreativeCodingDomain.P5_SOUND)
        self.assertEqual(p5_sound.source_type, OfficialSourceType.API_REFERENCE)
        self.assertEqual(
            p5_sound.additional_urls,
            ("https://p5js.org/reference/p5.sound/p5.FFT/",),
        )

    def test_p5_core_reference_includes_browser_audio_start_guidance(self) -> None:
        source = get_official_source("p5_reference")

        self.assertIn(
            "https://p5js.org/reference/p5/userStartAudio/",
            source.additional_urls,
        )


if __name__ == "__main__":
    unittest.main()
