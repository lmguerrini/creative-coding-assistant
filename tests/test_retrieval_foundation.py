import hashlib
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag.retrieval import (
    KnowledgeBaseRetrievalFilter,
    KnowledgeBaseRetrievalRequest,
    KnowledgeBaseRetriever,
)
from creative_coding_assistant.rag.retrieval.models import KnowledgeBaseSearchResult
from creative_coding_assistant.rag.retrieval.postprocess import (
    select_retrieval_results,
)
from creative_coding_assistant.rag.sources import OfficialSourceType
from creative_coding_assistant.rag.sync import OfficialKnowledgeBaseIndexer
from creative_coding_assistant.rag.sync.models import OfficialSourceChunk
from creative_coding_assistant.vectorstore import (
    ChromaCollection,
    ChromaRecordMetadata,
    ChromaRepository,
    VectorRecord,
    VectorRecordKind,
    create_chroma_client,
    get_collection_definition,
)


class RetrievalFoundationTests(unittest.TestCase):
    def test_retriever_returns_semantic_matches_from_indexed_kb_chunks(self) -> None:
        with _kb_client() as client:
            _seed_kb_records(client)
            retriever = KnowledgeBaseRetriever(
                client=client,
                embedder=_FakeQueryEmbedder({"camera guidance": [1.0, 0.0, 0.0]}),
            )

            response = retriever.search(
                KnowledgeBaseRetrievalRequest(query="camera guidance", limit=2)
            )

            self.assertEqual(len(response.results), 2)
            self.assertEqual(response.results[0].source_id, "three_docs")
            self.assertEqual(response.results[0].domain, CreativeCodingDomain.THREE_JS)
            self.assertEqual(
                response.results[0].source_type,
                OfficialSourceType.API_REFERENCE,
            )
            self.assertIn("camera", response.results[0].text.lower())
            self.assertGreaterEqual(
                response.results[0].score,
                response.results[1].score,
            )

    def test_retriever_applies_domain_filter(self) -> None:
        with _kb_client() as client:
            _seed_kb_records(client)
            retriever = KnowledgeBaseRetriever(
                client=client,
                embedder=_FakeQueryEmbedder({"creative code": [1.0, 0.0, 0.0]}),
            )

            response = retriever.search(
                KnowledgeBaseRetrievalRequest(
                    query="creative code",
                    filters=KnowledgeBaseRetrievalFilter(
                        domain=CreativeCodingDomain.P5_JS
                    ),
                )
            )

            self.assertEqual(len(response.results), 1)
            self.assertEqual(response.results[0].domain, CreativeCodingDomain.P5_JS)
            self.assertEqual(response.results[0].source_id, "p5_reference")

    def test_retriever_treats_empty_domains_as_unconstrained(self) -> None:
        with _kb_client() as client:
            _seed_kb_records(client)
            retriever = KnowledgeBaseRetriever(
                client=client,
                embedder=_FakeQueryEmbedder({"creative code": [1.0, 0.0, 0.0]}),
            )

            response = retriever.search(
                KnowledgeBaseRetrievalRequest(
                    query="creative code",
                    limit=3,
                    filters=KnowledgeBaseRetrievalFilter(domains=()),
                )
            )

            self.assertEqual(len(response.results), 3)
            self.assertEqual(
                {result.domain for result in response.results},
                {
                    CreativeCodingDomain.THREE_JS,
                    CreativeCodingDomain.P5_JS,
                    CreativeCodingDomain.GLSL,
                },
            )

    def test_retriever_applies_multi_domain_filter(self) -> None:
        with _kb_client() as client:
            _seed_kb_records(client)
            retriever = KnowledgeBaseRetriever(
                client=client,
                embedder=_FakeQueryEmbedder({"creative code": [1.0, 0.0, 0.0]}),
            )

            response = retriever.search(
                KnowledgeBaseRetrievalRequest(
                    query="creative code",
                    limit=3,
                    filters=KnowledgeBaseRetrievalFilter(
                        domains=(
                            CreativeCodingDomain.P5_JS,
                            CreativeCodingDomain.GLSL,
                        )
                    ),
                )
            )

            self.assertEqual(len(response.results), 2)
            self.assertEqual(
                {result.domain for result in response.results},
                {
                    CreativeCodingDomain.P5_JS,
                    CreativeCodingDomain.GLSL,
                },
            )

    def test_retriever_applies_source_filter(self) -> None:
        with _kb_client() as client:
            _seed_kb_records(client)
            retriever = KnowledgeBaseRetriever(
                client=client,
                embedder=_FakeQueryEmbedder({"shader docs": [1.0, 0.0, 0.0]}),
            )

            response = retriever.search(
                KnowledgeBaseRetrievalRequest(
                    query="shader docs",
                    filters=KnowledgeBaseRetrievalFilter(
                        source_id="glsl_language_spec_460"
                    ),
                )
            )

            self.assertEqual(len(response.results), 1)
            self.assertEqual(
                response.results[0].source_id,
                "glsl_language_spec_460",
            )
            self.assertEqual(response.results[0].publisher, "Khronos Group")

    def test_retriever_uses_only_kb_collection(self) -> None:
        with _kb_client() as client:
            _seed_kb_records(client)
            _seed_non_kb_record(client)
            retriever = KnowledgeBaseRetriever(
                client=client,
                embedder=_FakeQueryEmbedder({"camera guidance": [1.0, 0.0, 0.0]}),
            )

            response = retriever.search(
                KnowledgeBaseRetrievalRequest(query="camera guidance", limit=1)
            )

            self.assertEqual(len(response.results), 1)
            self.assertEqual(response.results[0].source_id, "three_docs")
            self.assertNotEqual(
                response.results[0].record_id,
                "project_memory:project_memory:v1:manual",
            )

    def test_retriever_rejects_empty_query_embedding(self) -> None:
        with _kb_client() as client:
            _seed_kb_records(client)
            retriever = KnowledgeBaseRetriever(
                client=client,
                embedder=_FakeQueryEmbedder({"bad query": []}),
            )

            with self.assertRaisesRegex(ValueError, "must not be empty"):
                retriever.search(KnowledgeBaseRetrievalRequest(query="bad query"))

    def test_filter_removes_generic_landing_chunks_and_keeps_useful_docs(self) -> None:
        results = (
            _result(
                source_id="three_examples",
                source_type=OfficialSourceType.EXAMPLES,
                registry_title="three.js examples",
                document_title="three.js examples three.js examples",
                text=(
                    "three.js examples three.js examples "
                    "Select an example from the sidebar"
                ),
                score=0.9,
                distance=0.1,
            ),
            _result(
                source_id="three_docs",
                source_type=OfficialSourceType.API_REFERENCE,
                registry_title="three.js docs",
                document_title="three.js docs",
                text=(
                    "BoxGeometry is a geometry class for a rectangular cuboid "
                    "with width, height, and depth parameters."
                ),
                score=0.8,
                distance=0.2,
            ),
            _result(
                source_id="three_manual",
                source_type=OfficialSourceType.GUIDE,
                registry_title="three.js manual",
                document_title="three.js manual",
                text="three.js manual docs manual en fr ru 中文 日本語",
                score=0.7,
                distance=0.3,
            ),
        )

        filtered = select_retrieval_results(results, limit=3)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].source_id, "three_docs")

    def test_filter_falls_back_to_original_results_when_all_are_filtered(self) -> None:
        results = (
            _result(
                source_id="three_examples",
                source_type=OfficialSourceType.EXAMPLES,
                registry_title="three.js examples",
                document_title="three.js examples three.js examples",
                text=(
                    "three.js examples three.js examples "
                    "Select an example from the sidebar"
                ),
                score=0.9,
                distance=0.1,
            ),
            _result(
                source_id="three_manual",
                source_type=OfficialSourceType.GUIDE,
                registry_title="three.js manual",
                document_title="three.js manual",
                text="three.js manual docs manual en fr ru 中文 日本語",
                score=0.7,
                distance=0.3,
            ),
        )

        filtered = select_retrieval_results(results, limit=1)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].source_id, "three_examples")

    def test_dedup_removes_near_duplicate_chunks_from_same_source(self) -> None:
        results = (
            _result(
                source_id="r3f_introduction",
                source_type=OfficialSourceType.GUIDE,
                registry_title="Introduction - React Three Fiber",
                document_title="Introduction - React Three Fiber",
                text=(
                    "function Box(props) { const meshRef = useRef(null) "
                    "const [hovered, setHover] = useState(false) "
                    "useFrame((state, delta) => (meshRef.current.rotation.x += delta)) "
                    "return <mesh /> }"
                ),
                score=0.9,
                distance=0.1,
            ),
            _result(
                source_id="r3f_introduction",
                source_type=OfficialSourceType.GUIDE,
                registry_title="Introduction - React Three Fiber",
                document_title="Introduction - React Three Fiber",
                text=(
                    "function Box(props) { const meshRef = useRef(null) "
                    "const [hovered, setHover] = useState(false) "
                    "useFrame((state, delta) => (meshRef.current.rotation.x += delta)) "
                    "return <mesh/> }"
                ),
                score=0.89,
                distance=0.11,
            ),
            _result(
                source_id="r3f_hooks_api",
                source_type=OfficialSourceType.API_REFERENCE,
                registry_title="Hooks - React Three Fiber",
                document_title="Hooks - React Three Fiber",
                text=(
                    "This hook allows you to execute code on every rendered frame."
                ),
                score=0.8,
                distance=0.2,
            ),
        )

        deduplicated = select_retrieval_results(results, limit=3)

        self.assertEqual(len(deduplicated), 2)
        self.assertEqual(
            [result.source_id for result in deduplicated],
            ["r3f_introduction", "r3f_hooks_api"],
        )

    def test_dedup_preserves_non_duplicate_chunks_and_order(self) -> None:
        results = (
            _result(
                source_id="r3f_hooks_api",
                source_type=OfficialSourceType.API_REFERENCE,
                registry_title="Hooks - React Three Fiber",
                document_title="Hooks - React Three Fiber",
                text=(
                    "This hook allows you to execute code on every rendered frame."
                ),
                score=0.9,
                distance=0.1,
            ),
            _result(
                source_id="r3f_hooks_api",
                source_type=OfficialSourceType.API_REFERENCE,
                registry_title="Hooks - React Three Fiber",
                document_title="Hooks - React Three Fiber",
                text=(
                    "Callbacks will be executed in order of ascending priority "
                    "values, lowest first and highest last."
                ),
                score=0.85,
                distance=0.15,
            ),
            _result(
                source_id="r3f_canvas_api",
                source_type=OfficialSourceType.API_REFERENCE,
                registry_title="Canvas - React Three Fiber",
                document_title="Canvas - React Three Fiber",
                text="The Canvas object is your portal into three.js.",
                score=0.8,
                distance=0.2,
            ),
        )

        deduplicated = select_retrieval_results(results, limit=3)

        self.assertEqual(len(deduplicated), 3)
        self.assertEqual(
            [result.source_id for result in deduplicated],
            ["r3f_hooks_api", "r3f_hooks_api", "r3f_canvas_api"],
        )

    def test_dedup_removes_typed_and_untyped_variants_from_same_source(self) -> None:
        results = (
            _result(
                source_id="r3f_introduction",
                source_type=OfficialSourceType.GUIDE,
                registry_title="Introduction - React Three Fiber",
                document_title="Introduction - React Three Fiber",
                text=(
                    "function Box(props) { const meshRef = useRef(null) "
                    "const [hovered, setHover] = useState(false) "
                    "const [active, setActive] = useState(false) "
                    "useFrame((state, delta) => "
                    "(meshRef.current.rotation.x += delta)) "
                    "return <mesh /> }"
                ),
                score=0.9,
                distance=0.1,
            ),
            _result(
                source_id="r3f_introduction",
                source_type=OfficialSourceType.GUIDE,
                registry_title="Introduction - React Three Fiber",
                document_title="Introduction - React Three Fiber",
                text=(
                    "function Box(props: ThreeElements['mesh']) { "
                    "const meshRef = useRef<THREE.Mesh>(null!) "
                    "const [hovered, setHover] = useState(false) "
                    "const [active, setActive] = useState(false) "
                    "useFrame((state, delta) => "
                    "(meshRef.current.rotation.x += delta)) "
                    "return <mesh /> }"
                ),
                score=0.89,
                distance=0.11,
            ),
            _result(
                source_id="r3f_hooks_api",
                source_type=OfficialSourceType.API_REFERENCE,
                registry_title="Hooks - React Three Fiber",
                document_title="Hooks - React Three Fiber",
                text=(
                    "This hook allows you to execute code on every rendered frame."
                ),
                score=0.8,
                distance=0.2,
            ),
        )

        deduplicated = select_retrieval_results(results, limit=3)

        self.assertEqual(len(deduplicated), 2)
        self.assertEqual(
            [result.source_id for result in deduplicated],
            ["r3f_introduction", "r3f_hooks_api"],
        )

    def test_retriever_overfetches_to_replace_filtered_generic_hits(self) -> None:
        with _kb_client() as client:
            _seed_low_value_kb_records(client)
            retriever = KnowledgeBaseRetriever(
                client=client,
                embedder=_FakeQueryEmbedder({"rotating cube": [1.0, 0.0, 0.0]}),
            )

            response = retriever.search(
                KnowledgeBaseRetrievalRequest(query="rotating cube", limit=2)
            )

            self.assertEqual(len(response.results), 1)
            self.assertEqual(response.results[0].source_id, "three_box_geometry")
            self.assertIn("boxgeometry", response.results[0].text.lower())


def _seed_kb_records(client) -> None:
    indexer = OfficialKnowledgeBaseIndexer(client=client)
    chunks = (
        _chunk(
            source_id="three_docs",
            domain=CreativeCodingDomain.THREE_JS,
            source_type=OfficialSourceType.API_REFERENCE,
            publisher="three.js",
            registry_title="three.js Documentation",
            source_url="https://threejs.org/docs/",
            text="Camera setup covers perspective cameras and scene framing.",
        ),
        _chunk(
            source_id="p5_reference",
            domain=CreativeCodingDomain.P5_JS,
            source_type=OfficialSourceType.API_REFERENCE,
            publisher="p5.js",
            registry_title="p5.js Reference",
            source_url="https://p5js.org/reference/",
            text="noise() creates smooth motion for sketches and generative art.",
        ),
        _chunk(
            source_id="glsl_language_spec_460",
            domain=CreativeCodingDomain.GLSL,
            source_type=OfficialSourceType.SPECIFICATION,
            publisher="Khronos Group",
            registry_title="GLSL 4.60 Specification",
            source_url=(
                "https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html"
            ),
            text="Uniform variables provide external values to shader programs.",
        ),
    )
    embeddings = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )
    indexer.upsert_chunks(chunks, embeddings)


def _seed_non_kb_record(client) -> None:
    repository = ChromaRepository(
        client=client,
        definition=get_collection_definition(ChromaCollection.PROJECT_MEMORY),
    )
    repository.upsert(
        [
            VectorRecord(
                id="project_memory:project_memory:v1:manual",
                document="This project memory vector should never appear in KB search.",
                metadata=ChromaRecordMetadata(
                    collection=ChromaCollection.PROJECT_MEMORY,
                    record_kind=VectorRecordKind.PROJECT_MEMORY,
                    source_id="memory-note",
                    project_id="project-1",
                    domain=CreativeCodingDomain.THREE_JS,
                ),
                embedding=[1.0, 0.0, 0.0],
            )
        ]
    )


def _seed_low_value_kb_records(client) -> None:
    indexer = OfficialKnowledgeBaseIndexer(client=client)
    chunks = (
        _chunk(
            source_id="three_examples",
            domain=CreativeCodingDomain.THREE_JS,
            source_type=OfficialSourceType.EXAMPLES,
            publisher="three.js",
            registry_title="three.js examples",
            document_title="three.js examples three.js examples",
            source_url="https://threejs.org/examples/",
            text=(
                "three.js examples three.js examples "
                "Select an example from the sidebar"
            ),
        ),
        _chunk(
            source_id="three_docs",
            domain=CreativeCodingDomain.THREE_JS,
            source_type=OfficialSourceType.API_REFERENCE,
            publisher="three.js",
            registry_title="three.js docs",
            document_title="three.js docs",
            source_url="https://threejs.org/docs/",
            text=(
                "three.js docs docs manual AnimationAction BufferGeometry "
                "Object3D UniformsGroup"
            ),
        ),
        _chunk(
            source_id="three_box_geometry",
            domain=CreativeCodingDomain.THREE_JS,
            source_type=OfficialSourceType.API_REFERENCE,
            publisher="three.js",
            registry_title="BoxGeometry - three.js docs",
            document_title="BoxGeometry",
            source_url=(
                "https://threejs.org/docs/#api/en/geometries/BoxGeometry"
            ),
            text=(
                "BoxGeometry is a geometry class for a rectangular cuboid with "
                "width, height, and depth parameters."
            ),
        ),
    )
    embeddings = (
        [1.0, 0.0, 0.0],
        [0.99, 0.0, 0.0],
        [0.98, 0.0, 0.0],
    )
    indexer.upsert_chunks(chunks, embeddings)


def _chunk(
    *,
    source_id: str,
    domain: CreativeCodingDomain,
    source_type: OfficialSourceType,
    publisher: str,
    registry_title: str,
    document_title: str | None = None,
    source_url: str,
    text: str,
) -> OfficialSourceChunk:
    content_hash = _digest(source_id + source_url)
    chunk_hash = _digest(text)
    return OfficialSourceChunk(
        source_id=source_id,
        domain=domain,
        source_type=source_type,
        registry_title=registry_title,
        publisher=publisher,
        source_url=source_url,
        resolved_url=source_url,
        fetched_at=_time(),
        document_title=document_title or registry_title,
        content_hash=content_hash,
        chunk_index=0,
        text=text,
        chunk_hash=chunk_hash,
        char_count=len(text),
    )


def _result(
    *,
    source_id: str,
    source_type: OfficialSourceType,
    registry_title: str,
    document_title: str,
    text: str,
    score: float,
    distance: float,
) -> KnowledgeBaseSearchResult:
    return KnowledgeBaseSearchResult(
        record_id=f"record:{source_id}",
        source_id=source_id,
        domain=CreativeCodingDomain.THREE_JS,
        source_type=source_type,
        publisher="three.js",
        registry_title=registry_title,
        document_title=document_title,
        source_url="https://threejs.org/docs/",
        resolved_url="https://threejs.org/docs/",
        chunk_index=0,
        text=text,
        char_count=len(text),
        content_hash=_digest(source_id + document_title),
        chunk_hash=_digest(text),
        distance=distance,
        score=score,
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _time() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


class _FakeQueryEmbedder:
    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self._embeddings = embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings[text]


class _kb_client:
    def __enter__(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        return create_chroma_client(path=Path(self._temp_dir.name) / "chroma")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
