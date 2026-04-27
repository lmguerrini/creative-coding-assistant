import unittest
from unittest.mock import patch

from creative_coding_assistant.app import build_assistant_service
from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import GenerationProvider, OpenAIGenerationProvider


class ServiceBootstrapCompositionTests(unittest.TestCase):
    def test_build_service_skips_retrieval_without_embedding_config(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5-mini",
            openai_api_key=None,
        )
        fake_client = _FakeChromaClient()

        with patch(
            "creative_coding_assistant.app.bootstrap.create_chroma_client",
            return_value=fake_client,
        ) as create_client:
            with patch(
                "creative_coding_assistant.llm.openai_adapter._build_openai_client",
                side_effect=AssertionError(
                    "Provider API client should not be constructed during bootstrap."
                ),
            ) as build_openai_client:
                with patch(
                    "creative_coding_assistant.rag.embeddings.openai._build_openai_client",
                    side_effect=AssertionError(
                        "Embedder API client should not be constructed "
                        "during bootstrap."
                    ),
                ) as build_embedder_client:
                    service = build_assistant_service(settings=settings)

        create_client.assert_called_once_with(settings=settings)
        build_openai_client.assert_not_called()
        build_embedder_client.assert_not_called()
        self.assertIsNotNone(service._memory_gateway)
        self.assertIsNone(service._memory_recorder)
        self.assertIsNone(service._retrieval_gateway)
        self.assertIsNotNone(service._context_assembler)
        self.assertIsNotNone(service._prompt_input_builder)
        self.assertIsNotNone(service._prompt_renderer)
        self.assertIsNotNone(service._generation_gateway)
        self.assertIsInstance(service._generation_provider, OpenAIGenerationProvider)
        self.assertGreaterEqual(len(fake_client.collection_names), 6)

    def test_build_service_composes_retrieval_from_settings_when_configured(
        self,
    ) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5-mini",
            openai_api_key="sk-test-secret",
        )
        fake_client = _FakeChromaClient()

        with patch(
            "creative_coding_assistant.app.bootstrap.create_chroma_client",
            return_value=fake_client,
        ):
            with patch(
                "creative_coding_assistant.rag.embeddings.openai._build_openai_client",
                side_effect=AssertionError(
                    "Embedder API client should not be constructed during bootstrap."
                ),
            ) as build_embedder_client:
                service = build_assistant_service(settings=settings)

        self.assertIsNotNone(service._retrieval_gateway)
        self.assertIsNotNone(service._memory_recorder)
        build_embedder_client.assert_not_called()

    def test_build_service_preserves_explicit_query_embedder(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5-mini",
            openai_api_key="sk-test-secret",
        )
        fake_client = _FakeChromaClient()
        embedder = _FakeQueryEmbedder()

        with patch(
            "creative_coding_assistant.app.bootstrap.create_chroma_client",
            return_value=fake_client,
        ):
            service = build_assistant_service(
                settings=settings,
                query_embedder=embedder,
            )

        self.assertIsNotNone(service._retrieval_gateway)
        self.assertFalse(embedder.called)

    def test_build_service_preserves_explicit_generation_provider(self) -> None:
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5-mini",
            openai_api_key="sk-test-secret",
        )
        fake_client = _FakeChromaClient()
        explicit_provider = _ExplicitGenerationProvider()

        with patch(
            "creative_coding_assistant.app.bootstrap.create_chroma_client",
            return_value=fake_client,
        ):
            with patch(
                "creative_coding_assistant.orchestration.service.build_generation_provider",
                side_effect=AssertionError(
                    "Factory should not run when an explicit provider is injected."
                ),
            ) as build_provider:
                service = build_assistant_service(
                    settings=settings,
                    generation_provider=explicit_provider,
                )

        build_provider.assert_not_called()
        self.assertIs(service._generation_provider, explicit_provider)


class _FakeCollection:
    def upsert(self, **kwargs: object) -> None:
        del kwargs

    def get(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        return {"ids": [], "documents": [], "metadatas": []}

    def query(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    def count(self) -> int:
        return 0


class _FakeChromaClient:
    def __init__(self) -> None:
        self.collection_names: list[str] = []

    def get_or_create_collection(
        self,
        *,
        name: str,
        metadata: dict[str, object],
    ) -> _FakeCollection:
        del metadata
        self.collection_names.append(name)
        return _FakeCollection()


class _FakeQueryEmbedder:
    def __init__(self) -> None:
        self.called = False

    def embed_query(self, text: str) -> list[float]:
        del text
        self.called = True
        return [1.0, 0.0, 0.0]


class _ExplicitGenerationProvider(GenerationProvider):
    def stream(self, request: object):
        del request
        return iter(())


if __name__ == "__main__":
    unittest.main()
