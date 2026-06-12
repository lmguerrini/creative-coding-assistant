import unittest

from creative_coding_assistant.orchestration.audio_reactive import (
    AudioReactiveSource,
    AudioReactiveVisualTarget,
    audio_reactive_prompt_lines,
    derive_audio_reactive_guidance,
)


class AudioReactiveGuidanceTests(unittest.TestCase):
    def test_derives_bounded_audio_to_visual_relationships(self) -> None:
        guidance = derive_audio_reactive_guidance(
            (
                "Create an audio-reactive field where amplitude drives light, "
                "bass expands the camera, mids shift color, highs add particles, "
                "rhythm rotates patterns, and the envelope reveals geometry."
            ),
            output_modality="audiovisual",
            runtime_recommendations=("Tone.js", "Three.js"),
        )

        self.assertIsNotNone(guidance)
        assert guidance is not None
        self.assertEqual(
            tuple(mapping.source for mapping in guidance.mappings),
            (
                AudioReactiveSource.AMPLITUDE,
                AudioReactiveSource.BASS,
                AudioReactiveSource.MIDS,
                AudioReactiveSource.HIGHS,
                AudioReactiveSource.RHYTHM,
                AudioReactiveSource.ENVELOPE,
            ),
        )
        bass = guidance.mappings[1]
        self.assertEqual(
            bass.targets,
            (
                AudioReactiveVisualTarget.PULSE,
                AudioReactiveVisualTarget.EXPANSION,
                AudioReactiveVisualTarget.CAMERA_MOVEMENT,
            ),
        )
        self.assertEqual(guidance.audio_runtime, "Tone.js")
        self.assertEqual(guidance.visual_runtime, "Three.js")
        self.assertEqual(guidance.activation, "explicit_user_gesture")

    def test_uses_drone_intensity_for_slow_atmospheric_targets(self) -> None:
        guidance = derive_audio_reactive_guidance(
            "Build an audiovisual drone with slow aura and fog movement.",
            output_modality="audiovisual",
            runtime_recommendations=("Tone.js", "GLSL"),
        )

        self.assertIsNotNone(guidance)
        assert guidance is not None
        drone = next(
            mapping
            for mapping in guidance.mappings
            if mapping.source is AudioReactiveSource.DRONE_INTENSITY
        )
        self.assertEqual(
            drone.targets,
            (
                AudioReactiveVisualTarget.FOG,
                AudioReactiveVisualTarget.AURA,
                AudioReactiveVisualTarget.FIELD_DENSITY,
            ),
        )

    def test_gates_visual_only_and_audio_only_requests(self) -> None:
        for modality in ("visual", "audio", None):
            with self.subTest(modality=modality):
                self.assertIsNone(
                    derive_audio_reactive_guidance(
                        "Create a pulsing field.",
                        output_modality=modality,
                    )
                )

    def test_uses_bounded_tone_and_dynamic_parameter_hints(self) -> None:
        guidance = derive_audio_reactive_guidance(
            "Create an audiovisual field.",
            output_modality="audiovisual",
            runtime_recommendations=("Tone.js", "p5.js"),
            tone_metadata=(
                "const fft = new Tone.FFT(); "
                "Tone.Transport.bpm.value = 96; "
                "const synth = new Tone.MembraneSynth();"
            ),
            dynamic_parameter_guidance=(
                "Rotation speed: 1.2 x\nBloom intensity: 1.4 x"
            ),
        )

        self.assertIsNotNone(guidance)
        assert guidance is not None
        sources = tuple(mapping.source for mapping in guidance.mappings)
        self.assertIn(AudioReactiveSource.RHYTHM, sources)
        self.assertIn(AudioReactiveSource.ENVELOPE, sources)
        self.assertIn(AudioReactiveSource.BASS, sources)
        self.assertIn(AudioReactiveSource.HIGHS, sources)
        self.assertIn("Tone.js metadata", guidance.mappings[0].evidence)
        self.assertIn("dynamic parameters", guidance.mappings[0].evidence)

    def test_prompt_guidance_preserves_explicit_audio_activation(self) -> None:
        guidance = derive_audio_reactive_guidance(
            "Create an audio-reactive visual using amplitude.",
            output_modality="audiovisual",
            runtime_recommendations=("Tone.js", "Hydra"),
        )

        self.assertIsNotNone(guidance)
        assert guidance is not None
        lines = audio_reactive_prompt_lines(guidance)
        self.assertTrue(any("amplitude" in line and "scale" in line for line in lines))
        self.assertTrue(
            any("silent until explicit user activation" in line for line in lines)
        )
        self.assertFalse(any("autoplay" in line.lower() for line in lines))

    def test_explicit_parameter_disable_removes_existing_mapping(self) -> None:
        base = derive_audio_reactive_guidance(
            "Create an audio-reactive visual.",
            output_modality="audiovisual",
        )

        guidance = derive_audio_reactive_guidance(
            "Refine the selected artifact.",
            output_modality="audiovisual",
            dynamic_parameter_guidance="Audio reactivity: disabled",
            base_guidance=base,
        )

        self.assertIsNone(guidance)


if __name__ == "__main__":
    unittest.main()
