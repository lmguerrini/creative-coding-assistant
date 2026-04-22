import tempfile
import unittest
from pathlib import Path

from creative_coding_assistant.contracts import AssistantMode, CreativeCodingDomain
from creative_coding_assistant.vectorstore import (
    ChromaCollection,
    ChromaRecordMetadata,
    ChromaRepository,
    VectorRecord,
    VectorRecordKind,
    collection_definitions,
    collection_names,
    create_chroma_client,
    ensure_project_collections,
    get_collection_definition,
    make_record_id,
)


class ChromaFoundationTests(unittest.TestCase):
    def test_collection_definitions_cover_every_collection(self) -> None:
        definitions = collection_definitions()

        self.assertEqual(
            tuple(definition.name.value for definition in definitions),
            collection_names(),
        )
        self.assertTrue(all(definition.responsibility for definition in definitions))

    def test_get_collection_definition_returns_responsibility(self) -> None:
        definition = get_collection_definition(ChromaCollection.PROJECT_MEMORY)

        self.assertEqual(definition.name, ChromaCollection.PROJECT_MEMORY)
        self.assertIn("project", definition.responsibility.lower())

    def test_make_record_id_is_stable_and_scoped(self) -> None:
        first_id = make_record_id(
            collection=ChromaCollection.KB_OFFICIAL_DOCS,
            record_kind=VectorRecordKind.OFFICIAL_DOC_CHUNK,
            parts=("three-js", "https://threejs.org/docs", "0"),
        )
        second_id = make_record_id(
            collection=ChromaCollection.KB_OFFICIAL_DOCS,
            record_kind=VectorRecordKind.OFFICIAL_DOC_CHUNK,
            parts=("three-js", "https://threejs.org/docs", "0"),
        )

        self.assertEqual(first_id, second_id)
        self.assertTrue(first_id.startswith("kb_official_docs:official_doc_chunk:v1:"))

    def test_metadata_flattens_to_chroma_safe_values(self) -> None:
        metadata = ChromaRecordMetadata(
            collection=ChromaCollection.CONVERSATION_TURNS,
            record_kind=VectorRecordKind.CONVERSATION_TURN,
            source_id="turn-1",
            domain=CreativeCodingDomain.P5_JS,
            mode=AssistantMode.DEBUG,
            conversation_id="conversation-1",
            project_id="project-1",
            extras={"turn_index": 1, "used_context": True},
        )

        self.assertEqual(
            metadata.to_chroma(),
            {
                "collection": "conversation_turns",
                "record_kind": "conversation_turn",
                "schema_version": 1,
                "source_id": "turn-1",
                "domain": "p5_js",
                "mode": "debug",
                "conversation_id": "conversation-1",
                "project_id": "project-1",
                "turn_index": 1,
                "used_context": True,
            },
        )

    def test_metadata_rejects_reserved_extra_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved keys"):
            ChromaRecordMetadata(
                collection=ChromaCollection.PROJECT_MEMORY,
                record_kind=VectorRecordKind.PROJECT_MEMORY,
                source_id="memory-1",
                extras={"collection": "other"},
            )

    def test_vector_record_requires_explicit_embedding(self) -> None:
        with self.assertRaises(ValueError):
            VectorRecord(
                id="record-1",
                document="Stored text",
                metadata=ChromaRecordMetadata(
                    collection=ChromaCollection.PROJECT_MEMORY,
                    record_kind=VectorRecordKind.PROJECT_MEMORY,
                    source_id="memory-1",
                ),
                embedding=[],
            )

    def test_repository_upserts_and_reads_from_persistent_chroma(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            persist_dir = Path(temp_dir) / "chroma"
            client = create_chroma_client(path=persist_dir)
            ensure_project_collections(client)
            definition = get_collection_definition(ChromaCollection.PROJECT_MEMORY)
            repository = ChromaRepository(client=client, definition=definition)
            record_id = make_record_id(
                collection=ChromaCollection.PROJECT_MEMORY,
                record_kind=VectorRecordKind.PROJECT_MEMORY,
                parts=("project-1", "style-preference"),
            )
            record = VectorRecord(
                id=record_id,
                document="Use restrained colors and visible source citations.",
                metadata=ChromaRecordMetadata(
                    collection=ChromaCollection.PROJECT_MEMORY,
                    record_kind=VectorRecordKind.PROJECT_MEMORY,
                    source_id="style-preference",
                    project_id="project-1",
                    domain=CreativeCodingDomain.THREE_JS,
                    extras={"memory_kind": "style"},
                ),
                embedding=[0.1, 0.2, 0.3],
            )

            repository.upsert([record])
            stored = repository.get(record_id)

            self.assertEqual(repository.count(), 1)
            self.assertIsNotNone(stored)
            self.assertEqual(stored.document, record.document)
            self.assertEqual(stored.metadata["collection"], "project_memory")
            self.assertEqual(stored.metadata["memory_kind"], "style")


if __name__ == "__main__":
    unittest.main()
