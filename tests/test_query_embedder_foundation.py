import unittest
from unittest.mock import patch

from creative_coding_assistant.core import Settings
from creative_coding_assistant.rag.retrieval import (
    OpenAIQueryEmbedder,
    build_query_embedder,
)


class QueryEmbedderFoundationTests(unittest.TestCase):
    def test_settings_expose_openai_embedding_defaults(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.openai_embedding_model, "text-embedding-3-small")
        self.assertFalse(settings.has_openai_embedding_config)

    def test_settings_expose_openai_embedding_config_when_key_exists(self) -> None:
        settings = Settings(openai_api_key="sk-test-secret")

        self.assertTrue(settings.has_openai_embedding_config)

    def test_build_query_embedder_returns_none_without_embedding_config(self) -> None:
        settings = Settings(openai_api_key=None)

        embedder = build_query_embedder(settings)

        self.assertIsNone(embedder)

    def test_build_query_embedder_uses_openai_when_configured(self) -> None:
        settings = Settings(
            openai_api_key="sk-test-secret",
            openai_embedding_model="text-embedding-3-large",
        )

        embedder = build_query_embedder(settings)

        self.assertIsInstance(embedder, OpenAIQueryEmbedder)

    def test_openai_query_embedder_uses_settings_backed_model(self) -> None:
        settings = Settings(
            openai_api_key="sk-test-secret",
            openai_embedding_model="text-embedding-3-large",
        )
        client = _FakeOpenAIClient(embedding=[0.1, 0.2, 0.3])
        embedder = OpenAIQueryEmbedder(settings=settings, client=client)

        embedding = embedder.embed_query("camera guidance")

        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        self.assertEqual(
            client.calls,
            [
                {
                    "model": "text-embedding-3-large",
                    "input": ["camera guidance"],
                }
            ],
        )

    def test_openai_query_embedder_rejects_empty_query_text(self) -> None:
        embedder = OpenAIQueryEmbedder(
            settings=Settings(openai_api_key="sk-test-secret"),
            client=_FakeOpenAIClient(embedding=[0.1]),
        )

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            embedder.embed_query(" ")


class _FakeEmbeddingsApi:
    def __init__(
        self,
        *,
        embedding: list[float],
        calls: list[dict[str, object]],
    ) -> None:
        self._embedding = embedding
        self._calls = calls

    def create(self, *, model: str, input: str) -> dict[str, object]:
        self._calls.append({"model": model, "input": input})
        return {
            "data": [
                {
                    "embedding": self._embedding,
                }
            ]
        }


class _FakeOpenAIClient:
    def __init__(self, *, embedding: list[float]) -> None:
        self.calls: list[dict[str, object]] = []
        self.embeddings = _FakeEmbeddingsApi(embedding=embedding, calls=self.calls)


if __name__ == "__main__":
    unittest.main()
