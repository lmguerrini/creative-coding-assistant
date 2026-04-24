import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    UnsupportedGenerationProviderError,
    build_generation_provider,
)


class ProviderSelectionShellTests(unittest.TestCase):
    def test_factory_builds_openai_provider_from_settings(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        provider = build_generation_provider(settings)

        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")
        self.assertTrue(provider._settings.has_openai_api_key)
        self.assertEqual(provider._settings.get_openai_api_key(), "sk-test-secret")

    def test_factory_raises_for_unsupported_provider(self) -> None:
        settings = Settings.model_construct(
            default_generation_provider="anthropic",
            openai_model="gpt-5-mini",
            openai_api_key=None,
        )

        with self.assertRaisesRegex(
            UnsupportedGenerationProviderError,
            "Unsupported generation provider: anthropic",
        ):
            build_generation_provider(settings)

    def test_factory_error_does_not_include_api_key(self) -> None:
        settings = Settings.model_construct(
            default_generation_provider="anthropic",
            openai_model="gpt-5-mini",
            openai_api_key="sk-secret-never-log",
        )

        with self.assertRaises(UnsupportedGenerationProviderError) as context:
            build_generation_provider(settings)

        self.assertNotIn("sk-secret-never-log", str(context.exception))


if __name__ == "__main__":
    unittest.main()
