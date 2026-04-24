import unittest

from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    GenerationProvider,
    OpenAIGenerationProvider,
)
from creative_coding_assistant.orchestration import (
    AssistantService,
    LlmGenerationAdapter,
)


class ServiceProviderBootstrapTests(unittest.TestCase):
    def test_service_does_not_build_provider_without_generation_gateway(self) -> None:
        service = AssistantService(
            settings=Settings(
                default_generation_provider=GenerationProviderName.OPENAI,
                openai_model="gpt-5-mini",
                openai_api_key="sk-test-secret",
            )
        )

        self.assertIsNone(service._generation_provider)

    def test_service_preserves_explicit_generation_provider(self) -> None:
        explicit_provider = _ExplicitGenerationProvider()
        service = AssistantService(
            settings=Settings(
                default_generation_provider=GenerationProviderName.OPENAI,
                openai_model="gpt-5-mini",
                openai_api_key="sk-test-secret",
            ),
            generation_gateway=LlmGenerationAdapter(),
            generation_provider=explicit_provider,
        )

        self.assertIs(service._generation_provider, explicit_provider)

    def test_service_bootstraps_openai_provider_from_settings(self) -> None:
        service = AssistantService(
            settings=Settings(
                default_generation_provider=GenerationProviderName.OPENAI,
                openai_model="gpt-5",
                openai_api_key="sk-test-secret",
            ),
            generation_gateway=LlmGenerationAdapter(),
        )

        self.assertIsInstance(service._generation_provider, OpenAIGenerationProvider)
        provider = service._generation_provider
        assert provider is not None
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")
        self.assertTrue(provider._settings.has_openai_api_key)
        self.assertEqual(provider._settings.get_openai_api_key(), "sk-test-secret")


class _ExplicitGenerationProvider(GenerationProvider):
    def stream(self, request: object):
        del request
        return iter(())


if __name__ == "__main__":
    unittest.main()
