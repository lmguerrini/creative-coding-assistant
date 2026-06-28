import unittest

from creative_coding_assistant.orchestration import (
    PromptCompressionInputSection,
    PromptCompressionResult,
    PromptCompressionSection,
    compress_prompt_sections,
    compress_prompt_text,
    prompt_compression_section_by_id,
    prompt_compression_sections_for_status,
)

REQUIRED_PROMPT_COMPRESSION_SECTION_FIELDS = {
    "section_id",
    "role",
    "name",
    "original_text",
    "compressed_text",
    "original_token_estimate",
    "compressed_token_estimate",
    "saved_tokens",
    "compression_status",
    "compression_pressure",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "prompt_compression_implemented",
    "source_prompt_mutation_implemented",
    "context_routing_implemented",
    "retrieval_compression_implemented",
    "memory_summarization_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "compression_only",
}


class PromptCompressionTests(unittest.TestCase):
    def test_short_prompt_remains_unchanged_with_boundary_flags(self) -> None:
        result = compress_prompt_text(
            "Make a calm p5.js sketch with a blue background.",
            target_token_budget=200,
        )
        section = result.sections[0]

        self.assertEqual(result.role, "prompt_compressor")
        self.assertEqual(
            result.serialization_version,
            "prompt_compression_result.v1",
        )
        self.assertEqual(result.section_ids, ("prompt::text",))
        self.assertEqual(result.saved_total_tokens, 0)
        self.assertTrue(result.within_budget)
        self.assertEqual(result.compression_pressure, "low")
        self.assertIn("preserves the source prompt response", result.authority_boundary)
        self.assertTrue(result.prompt_compression_implemented)
        self.assertFalse(result.source_prompt_mutation_implemented)
        self.assertFalse(result.context_routing_implemented)
        self.assertFalse(result.retrieval_compression_implemented)
        self.assertFalse(result.memory_summarization_implemented)
        self.assertFalse(result.provider_model_routing_implemented)
        self.assertFalse(result.provider_execution_implemented)
        self.assertFalse(result.persistent_storage_write_implemented)
        self.assertFalse(result.generated_output_mutation_implemented)
        self.assertTrue(result.compression_only)
        self.assertEqual(section.compression_status, "unchanged")
        self.assertEqual(section.original_text, section.compressed_text)

    def test_long_prompt_sections_are_compressed_as_separate_artifact(self) -> None:
        result = compress_prompt_sections(
            (
                PromptCompressionInputSection(
                    section_id="section::system",
                    role="system",
                    name="system",
                    content=_repeated("System guidance detail.", 120),
                ),
                PromptCompressionInputSection(
                    section_id="section::user",
                    role="user",
                    name="user",
                    content=_repeated("User request detail.", 160),
                ),
            ),
            target_token_budget=120,
        )

        self.assertGreater(result.original_total_tokens, result.compressed_total_tokens)
        self.assertGreater(result.saved_total_tokens, 0)
        self.assertLessEqual(result.compressed_total_tokens, result.target_token_budget)
        self.assertTrue(result.within_budget)
        self.assertEqual(result.compression_pressure, "medium")
        self.assertIn("[system:system]", result.compressed_prompt_text)
        self.assertIn("[user:user]", result.compressed_prompt_text)
        self.assertIn(
            "Use compressed prompt artifact only when explicitly selected.",
            result.advisory_actions,
        )

        for section in result.sections:
            self.assertEqual(
                set(section.model_dump(mode="json")),
                REQUIRED_PROMPT_COMPRESSION_SECTION_FIELDS,
            )
            self.assertEqual(
                section.serialization_version,
                "prompt_compression_section.v1",
            )
            self.assertEqual(section.compression_status, "compressed")
            self.assertLess(
                section.compressed_token_estimate,
                section.original_token_estimate,
            )
            self.assertEqual(
                section.saved_tokens,
                section.original_token_estimate - section.compressed_token_estimate,
            )
            self.assertIn("source_prompt_mutation", section.blocked_runtime_behaviors)
            self.assertTrue(section.prompt_compression_implemented)
            self.assertFalse(section.source_prompt_mutation_implemented)
            self.assertFalse(section.context_routing_implemented)
            self.assertFalse(section.retrieval_compression_implemented)
            self.assertFalse(section.memory_summarization_implemented)
            self.assertFalse(section.provider_model_routing_implemented)
            self.assertFalse(section.provider_execution_implemented)
            self.assertFalse(section.persistent_storage_write_implemented)
            self.assertFalse(section.generated_output_mutation_implemented)
            self.assertTrue(section.compression_only)

    def test_lookup_helpers_are_stable_and_read_only(self) -> None:
        result = compress_prompt_sections(
            (
                PromptCompressionInputSection(
                    section_id="section::system",
                    role="system",
                    name="system",
                    content=_repeated("System guidance detail.", 100),
                ),
                PromptCompressionInputSection(
                    section_id="section::user",
                    role="user",
                    name="user",
                    content=_repeated("User request detail.", 140),
                ),
            ),
            target_token_budget=100,
        )

        system = prompt_compression_section_by_id("section::system", result)
        compressed = prompt_compression_sections_for_status("compressed", result)
        missing = prompt_compression_section_by_id("missing", result)

        self.assertIsNone(missing)
        self.assertIsNotNone(system)
        assert system is not None
        self.assertEqual(system.name, "system")
        self.assertEqual(len(compressed), 2)
        self.assertIs(system, compressed[0])

    def test_result_rejects_mismatched_sections_or_totals(self) -> None:
        result = compress_prompt_text(_repeated("Prompt detail.", 120), target_token_budget=50)
        payload = result.model_dump(mode="json")
        payload["section_ids"] = ("missing",)

        with self.assertRaisesRegex(ValueError, "section_ids must match"):
            PromptCompressionResult(**payload)

        payload = result.model_dump(mode="json")
        payload["compressed_total_tokens"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "compressed_total_tokens must match",
        ):
            PromptCompressionResult(**payload)

        section_payload = result.sections[0].model_dump(mode="json")
        section_payload["saved_tokens"] += 1

        with self.assertRaisesRegex(ValueError, "saved_tokens must match"):
            PromptCompressionSection(**section_payload)

    def test_result_does_not_declare_provider_or_output_mutation_terms(self) -> None:
        result = compress_prompt_text(_repeated("Prompt detail.", 120), target_token_budget=50)
        combined_text = " ".join(
            (
                result.authority_boundary,
                *result.blocked_runtime_behaviors,
                *result.advisory_actions,
                *(
                    field
                    for section in result.sections
                    for field in (
                        section.section_id,
                        section.role,
                        section.name,
                        *section.evidence,
                        *section.advisory_actions,
                        *section.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "mutate_source_prompt(",
            "route_context(",
            "compress_retrieval(",
            "summarize_memory(",
            "select_provider(",
            "route_provider(",
            "execute_provider(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


def _repeated(sentence: str, count: int) -> str:
    return " ".join([sentence] * count)


if __name__ == "__main__":
    unittest.main()
