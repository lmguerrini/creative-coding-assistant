import ssl
import tempfile
import unittest
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError
from urllib.request import HTTPSHandler, Request

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag.sources import (
    OfficialSourceType,
    get_official_source,
)
from creative_coding_assistant.rag.sync import (
    ChunkingPolicy,
    FetchedSourceDocument,
    NormalizedSourceDocument,
    OfficialKnowledgeBaseIndexer,
    OfficialSourceChunker,
    OfficialSourceFetcher,
    OfficialSourceNormalizer,
    OfficialSourceSyncRequest,
    SourceContentFormat,
    TransportResponse,
    UrllibSourceTransport,
)
from creative_coding_assistant.rag.sync.fetcher import (
    _SameOriginHttpsRedirectHandler,
)
from creative_coding_assistant.vectorstore import create_chroma_client


class OfficialKnowledgeBaseSyncFoundationTests(unittest.TestCase):
    def test_urllib_transport_sends_browser_like_headers(self) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = None
        response.headers.get_content_charset.return_value = "utf-8"
        response.headers.get_content_type.return_value = "text/html"
        response.read.return_value = b"<html><body>Docs</body></html>"
        response.geturl.return_value = "https://p5js.org/reference/"
        response.status = 200

        opener = MagicMock()
        opener.open.return_value = response
        with (
            patch(
                "creative_coding_assistant.rag.sync.fetcher._validate_public_https_target"
            ) as validate_target_mock,
            patch(
                "creative_coding_assistant.rag.sync.fetcher.build_opener",
                return_value=opener,
            ) as build_opener_mock,
        ):
            result = UrllibSourceTransport().fetch("https://p5js.org/reference/")

        request = opener.open.call_args.args[0]
        self.assertIn("Mozilla/5.0", request.headers["User-agent"])
        self.assertIn("text/html", request.headers["Accept"])
        self.assertEqual(request.headers["Accept-language"], "en-US,en;q=0.9")
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 10.0)
        validate_target_mock.assert_called_once_with("https://p5js.org/reference/")
        handlers = build_opener_mock.call_args.args
        self.assertTrue(
            any(isinstance(handler, _SameOriginHttpsRedirectHandler) for handler in handlers)
        )
        https_handler = next(
            handler for handler in handlers if isinstance(handler, HTTPSHandler)
        )
        self.assertIsInstance(https_handler._context, ssl.SSLContext)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.content_type, "text/html")

    def test_urllib_transport_rejects_oversized_source_responses(self) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = None
        response.headers.get_content_charset.return_value = "utf-8"
        response.read.return_value = b"x" * (UrllibSourceTransport.MAX_RESPONSE_BYTES + 1)

        opener = MagicMock()
        opener.open.return_value = response
        with (
            patch(
                "creative_coding_assistant.rag.sync.fetcher._validate_public_https_target"
            ),
            patch(
                "creative_coding_assistant.rag.sync.fetcher.build_opener",
                return_value=opener,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "5 MiB safety limit"):
                UrllibSourceTransport().fetch("https://p5js.org/reference/")

        response.read.assert_called_once_with(
            UrllibSourceTransport.MAX_RESPONSE_BYTES + 1
        )

    def test_redirect_handler_rejects_cross_origin_before_target_resolution(self) -> None:
        handler = _SameOriginHttpsRedirectHandler(origin=("threejs.org", 443))

        with patch(
            "creative_coding_assistant.rag.sync.fetcher._validate_public_https_target"
        ) as validate_target_mock:
            with self.assertRaisesRegex(URLError, "safe network scope"):
                handler.redirect_request(
                    Request("https://threejs.org/docs/"),
                    None,
                    302,
                    "Found",
                    Message(),
                    "https://127.0.0.1/private",
                )

        validate_target_mock.assert_not_called()

    def test_redirect_handler_allows_public_same_origin_redirect(self) -> None:
        handler = _SameOriginHttpsRedirectHandler(origin=("threejs.org", 443))

        with patch(
            "creative_coding_assistant.rag.sync.fetcher._validate_public_https_target"
        ) as validate_target_mock:
            redirected = handler.redirect_request(
                Request("https://threejs.org/docs/"),
                None,
                302,
                "Found",
                Message(),
                "https://threejs.org/docs/index.html",
            )

        self.assertIsNotNone(redirected)
        assert redirected is not None
        self.assertEqual(redirected.full_url, "https://threejs.org/docs/index.html")
        validate_target_mock.assert_called_once_with(
            "https://threejs.org/docs/index.html"
        )

    def test_urllib_transport_rejects_private_dns_before_opening_connection(self) -> None:
        with (
            patch(
                "creative_coding_assistant.rag.sync.fetcher.getaddrinfo",
                return_value=((2, 1, 6, "", ("127.0.0.1", 443)),),
            ),
            patch(
                "creative_coding_assistant.rag.sync.fetcher.build_opener"
            ) as build_opener_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, "target was rejected"):
                UrllibSourceTransport().fetch("https://threejs.org/docs/")

        build_opener_mock.assert_not_called()

    def test_fetcher_requires_registered_source_id(self) -> None:
        fetcher = OfficialSourceFetcher(transport=_FakeTransport({}))
        request = OfficialSourceSyncRequest(
            source_id="unknown_source",
            requested_at=_time(),
        )

        with self.assertRaises(ValueError):
            fetcher.fetch(request)

    def test_fetcher_rejects_resolved_url_outside_approved_scope(self) -> None:
        fetcher = OfficialSourceFetcher(
            transport=_FakeTransport(
                {
                    "https://threejs.org/docs/": TransportResponse(
                        resolved_url="https://threejs.org/manual/",
                        status_code=200,
                        content_type="text/html; charset=utf-8",
                        content="<html><body>Manual</body></html>",
                    )
                }
            )
        )
        request = OfficialSourceSyncRequest(
            source_id="three_docs",
            requested_at=_time(),
        )

        with self.assertRaisesRegex(ValueError, "approved path scope"):
            fetcher.fetch(request)

    def test_fetcher_builds_fetched_document_from_approved_source(self) -> None:
        fetcher = OfficialSourceFetcher(
            transport=_FakeTransport(
                {
                    "https://threejs.org/docs/": TransportResponse(
                        resolved_url="https://threejs.org/docs/",
                        status_code=200,
                        content_type="text/html; charset=utf-8",
                        content="<html><title>Docs</title><body>Hello</body></html>",
                    )
                }
            )
        )
        request = OfficialSourceSyncRequest(
            source_id="three_docs",
            requested_at=_time(),
        )

        document = fetcher.fetch(request)

        self.assertEqual(document.domain, CreativeCodingDomain.THREE_JS)
        self.assertEqual(document.source_type, OfficialSourceType.API_REFERENCE)
        self.assertEqual(document.content_format, SourceContentFormat.HTML)
        self.assertEqual(document.resolved_url, "https://threejs.org/docs/")
        self.assertEqual(len(document.raw_content_hash), 64)

    def test_fetcher_aggregates_curated_source_pages_into_single_document(self) -> None:
        source = get_official_source("p5_examples")
        responses = {
            url: TransportResponse(
                resolved_url=url,
                status_code=200,
                content_type="text/html; charset=utf-8",
                content=(
                    "<html><head><title>Example Page</title></head><body>"
                    "<nav>Examples Menu</nav>"
                    "<main><p>function draw() { circle(40, 40, 20); }</p></main>"
                    "</body></html>"
                ),
            )
            for url in (source.url, *source.additional_urls)
        }
        fetcher = OfficialSourceFetcher(transport=_FakeTransport(responses))

        document = fetcher.fetch(
            OfficialSourceSyncRequest(source_id="p5_examples", requested_at=_time())
        )

        self.assertEqual(document.content_format, SourceContentFormat.HTML)
        self.assertIn("Page: Example Page", document.raw_content)
        self.assertGreater(document.raw_content.count("<section>"), 1)

    def test_normalizer_extracts_html_text_and_title(self) -> None:
        document = _fetched_document(
            raw_content="""
            <html>
              <head><title>Three.js API</title><style>.x { color: red; }</style></head>
              <body>
                <h1>Raycaster</h1>
                <p>Intersects objects in 3D scenes.</p>
                <script>window.secret = true;</script>
              </body>
            </html>
            """,
        )

        normalized = OfficialSourceNormalizer().normalize(document)

        self.assertEqual(normalized.document_title, "Three.js API")
        self.assertIn("Raycaster", normalized.normalized_text)
        self.assertIn("Intersects objects in 3D scenes.", normalized.normalized_text)
        self.assertNotIn("window.secret", normalized.normalized_text)

    def test_normalizer_skips_navigation_and_preserves_code_blocks(self) -> None:
        document = _fetched_document(
            source_id="p5_examples",
            domain=CreativeCodingDomain.P5_JS,
            registry_title="p5.js Runnable Examples",
            source_url="https://p5js.org/examples/calculating-values-constrain/",
            resolved_url="https://p5js.org/examples/calculating-values-constrain/",
            raw_content="""
            <html>
              <head><title>Example</title></head>
              <body>
                <nav>Skip to main content Menu Reference Tutorials</nav>
                <main>
                  <h1>Constrain</h1>
                  <p>Move a circle while keeping it inside a box.</p>
                  <pre>function draw() { background(220); circle(x, y, 40); }</pre>
                </main>
                <footer>Edited and maintained by p5.js Contributors.</footer>
              </body>
            </html>
            """,
        )

        normalized = OfficialSourceNormalizer().normalize(document)

        self.assertIn("function draw()", normalized.normalized_text)
        self.assertIn("circle(x, y, 40)", normalized.normalized_text)
        self.assertNotIn("Skip to main content", normalized.normalized_text)
        self.assertNotIn("Edited and maintained", normalized.normalized_text)

    def test_chunker_respects_character_boundaries(self) -> None:
        document = _normalized_document(
            normalized_text=(
                "Paragraph one explains setup and renderer wiring for a basic scene. "
                "It includes camera setup, sizing, and render loop notes.\n\n"
                "Paragraph two explains cameras, lighting, and scene composition "
                "for a more detailed example with orbit controls.\n\n"
                "Paragraph three covers animation loops, resizing, and cleanup steps "
                "that keep the example maintainable over time."
            )
        )
        chunker = OfficialSourceChunker(
            policy=ChunkingPolicy(max_chars=200, min_chunk_chars=50)
        )

        chunks = chunker.chunk(document)

        self.assertEqual(
            [chunk.chunk_index for chunk in chunks],
            list(range(len(chunks))),
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.char_count <= 200 for chunk in chunks))
        self.assertIn("Paragraph one", chunks[0].text)
        self.assertIn("Paragraph three", chunks[-1].text)

    def test_chunker_splits_large_paragraph(self) -> None:
        long_paragraph = " ".join(f"token{index}" for index in range(80))
        document = _normalized_document(normalized_text=long_paragraph)
        chunker = OfficialSourceChunker(
            policy=ChunkingPolicy(max_chars=200, min_chunk_chars=50)
        )

        chunks = chunker.chunk(document)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.char_count <= 200 for chunk in chunks))

    def test_chunker_drops_navigation_only_p5_sections_and_keeps_code(self) -> None:
        document = _normalized_document(
            source_id="p5_examples",
            domain=CreativeCodingDomain.P5_JS,
            source_url="https://p5js.org/examples/calculating-values-constrain/",
            resolved_url="https://p5js.org/examples/calculating-values-constrain/",
            normalized_text=(
                "Page: Constrain\n\n"
                "Move a circle while keeping it inside a box.\n\n"
                "```text function draw() { background(220); circle(x, y, 40); } ```\n\n"
                "Related Examples\n\n"
                "Edited and maintained by p5.js Contributors."
            ),
        )

        chunks = OfficialSourceChunker().chunk(document)

        self.assertEqual(len(chunks), 1)
        self.assertIn("circle(x, y, 40)", chunks[0].text)
        self.assertNotIn("Related Examples", chunks[0].text)

    def test_chunker_keeps_shader_like_glsl_sections(self) -> None:
        document = _normalized_document(
            source_id="glsl_language_spec_460",
            domain=CreativeCodingDomain.GLSL,
            source_type=OfficialSourceType.GUIDE,
            registry_title="OpenGL Wiki: OpenGL Shading Language",
            publisher="Khronos Group",
            source_url="https://wikis.khronos.org/opengl/OpenGL_Shading_Language",
            resolved_url="https://wikis.khronos.org/opengl/OpenGL_Shading_Language",
            normalized_text=(
                "Page: Fragment Shader\n\n"
                "A fragment shader writes a color for each fragment.\n\n"
                "```text void main() { gl_FragColor = "
                "vec4(vec3(0.2, 0.4, 1.0), 1.0); } ```\n\n"
                "Retrieved from https://wikis.khronos.org/opengl/Fragment_Shader"
            ),
        )

        chunks = OfficialSourceChunker().chunk(document)

        self.assertEqual(len(chunks), 1)
        self.assertIn("gl_FragColor", chunks[0].text)
        self.assertIn("vec3", chunks[0].text)
        self.assertNotIn("Retrieved from", chunks[0].text)

    def test_indexer_upserts_official_kb_chunks_into_chroma(self) -> None:
        with _kb_client() as client:
            indexer = OfficialKnowledgeBaseIndexer(client=client)
            chunks = OfficialSourceChunker().chunk(
                _normalized_document(
                    normalized_text="Official docs chunk for indexing boundaries."
                )
            )

            record_ids = indexer.upsert_chunks(chunks, embeddings=[[0.1, 0.2, 0.3]])
            stored_records = indexer.list_source_chunks(source_id="three_docs")

            self.assertEqual(len(record_ids), 1)
            self.assertEqual(len(stored_records), 1)
            self.assertEqual(
                stored_records[0].metadata["collection"],
                "kb_official_docs",
            )
            self.assertEqual(stored_records[0].metadata["source_type"], "api_reference")
            self.assertEqual(stored_records[0].metadata["chunk_index"], 0)

    def test_indexer_requires_chunk_embedding_alignment(self) -> None:
        with _kb_client() as client:
            indexer = OfficialKnowledgeBaseIndexer(client=client)
            chunks = OfficialSourceChunker(
                policy=ChunkingPolicy(max_chars=200, min_chunk_chars=50)
            ).chunk(
                _normalized_document(
                    normalized_text=(
                        "First long chunk contains setup details, rendering notes, "
                        "and animation loop guidance for a maintained example.\n\n"
                        "Second long chunk contains scene graph guidance, camera "
                        "positioning, and cleanup details for another example."
                    )
                )
            )

            with self.assertRaisesRegex(ValueError, "align one-to-one"):
                indexer.upsert_chunks(chunks, embeddings=[[0.1, 0.2, 0.3]])


def _fetched_document(
    *,
    source_id: str = "three_docs",
    domain: CreativeCodingDomain = CreativeCodingDomain.THREE_JS,
    source_type: OfficialSourceType = OfficialSourceType.API_REFERENCE,
    registry_title: str = "three.js Documentation",
    publisher: str = "three.js",
    source_url: str = "https://threejs.org/docs/",
    resolved_url: str = "https://threejs.org/docs/",
    raw_content: str,
    content_format: SourceContentFormat = SourceContentFormat.HTML,
) -> FetchedSourceDocument:
    return FetchedSourceDocument.from_content(
        source_id=source_id,
        domain=domain,
        source_type=source_type,
        registry_title=registry_title,
        publisher=publisher,
        source_url=source_url,
        resolved_url=resolved_url,
        fetched_at=_time(),
        content_format=content_format,
        raw_content=raw_content,
    )


def _normalized_document(
    *,
    normalized_text: str,
    source_id: str = "three_docs",
    domain: CreativeCodingDomain = CreativeCodingDomain.THREE_JS,
    source_type: OfficialSourceType = OfficialSourceType.API_REFERENCE,
    registry_title: str = "three.js Documentation",
    publisher: str = "three.js",
    source_url: str = "https://threejs.org/docs/",
    resolved_url: str = "https://threejs.org/docs/",
) -> NormalizedSourceDocument:
    return NormalizedSourceDocument.from_text(
        fetched_document=_fetched_document(
            source_id=source_id,
            domain=domain,
            source_type=source_type,
            registry_title=registry_title,
            publisher=publisher,
            source_url=source_url,
            resolved_url=resolved_url,
            raw_content="<html><body>stub</body></html>",
        ),
        document_title=registry_title,
        normalized_text=normalized_text,
    )


def _time() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


class _FakeTransport:
    def __init__(self, responses: dict[str, TransportResponse]) -> None:
        self._responses = responses

    def fetch(self, url: str) -> TransportResponse:
        return self._responses[url]


class _kb_client:
    def __enter__(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        return create_chroma_client(path=Path(self._temp_dir.name) / "chroma")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
