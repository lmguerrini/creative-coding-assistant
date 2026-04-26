import unittest

from creative_coding_assistant.clients import (
    TraceVisibilityLevel,
    default_trace_visibility,
    resolve_session_trace_visibility,
    trace_sections_for_level,
    trace_visibility_summary,
)


class StreamlitTraceVisibilityTests(unittest.TestCase):
    def test_default_trace_visibility_is_standard(self) -> None:
        self.assertEqual(
            default_trace_visibility(),
            TraceVisibilityLevel.STANDARD,
        )

    def test_resolve_session_trace_visibility_defaults_and_falls_back_safely(
        self,
    ) -> None:
        self.assertEqual(
            resolve_session_trace_visibility(None),
            TraceVisibilityLevel.STANDARD,
        )
        self.assertEqual(
            resolve_session_trace_visibility("minimal"),
            TraceVisibilityLevel.MINIMAL,
        )
        self.assertEqual(
            resolve_session_trace_visibility("full"),
            TraceVisibilityLevel.FULL,
        )
        self.assertEqual(
            resolve_session_trace_visibility("invalid_level"),
            TraceVisibilityLevel.STANDARD,
        )

    def test_trace_sections_for_each_density_level(self) -> None:
        self.assertEqual(
            trace_sections_for_level(TraceVisibilityLevel.MINIMAL),
            ("retrieval",),
        )
        self.assertEqual(
            trace_sections_for_level(TraceVisibilityLevel.STANDARD),
            ("memory", "retrieval", "context"),
        )
        self.assertEqual(
            trace_sections_for_level(TraceVisibilityLevel.FULL),
            (
                "memory",
                "retrieval",
                "context",
                "prompt_input",
                "rendered_prompt",
                "generation_input",
            ),
        )

    def test_trace_visibility_summary_is_readable(self) -> None:
        self.assertEqual(
            trace_visibility_summary(TraceVisibilityLevel.MINIMAL),
            "Trace view: minimal",
        )
        self.assertEqual(
            trace_visibility_summary(TraceVisibilityLevel.STANDARD),
            "Trace view: standard",
        )
        self.assertEqual(
            trace_visibility_summary(TraceVisibilityLevel.FULL),
            "Trace view: full",
        )


if __name__ == "__main__":
    unittest.main()
