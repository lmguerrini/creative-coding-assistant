import hashlib
import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEventType,
)
from creative_coding_assistant.orchestration import (
    AssistantService,
    KnowledgeBaseRetrievalAdapter,
    RetrievalContextFilter,
    RetrievalContextRequest,
    RetrievalContextResponse,
    RetrievalContextSource,
    RetrievedKnowledgeChunk,
    RouteCapability,
    RouteDecision,
    RouteName,
    build_retrieval_context_request,
)
from creative_coding_assistant.rag.retrieval import (
    KnowledgeBaseRetrievalRequest,
    KnowledgeBaseRetrievalResponse,
    KnowledgeBaseSearchResult,
)
from creative_coding_assistant.rag.sources import OfficialSourceType


class RetrievalIntegrationBoundaryTests(unittest.TestCase):
    def test_build_retrieval_request_uses_query_and_domain_filter(self) -> None:
        assistant_request = AssistantRequest(
            query="Explain camera setup in Three.js.",
            domain=CreativeCodingDomain.THREE_JS,
            mode=AssistantMode.EXPLAIN,
        )
        route_decision = RouteDecision(
            route=RouteName.EXPLAIN,
            mode=AssistantMode.EXPLAIN,
            domain=CreativeCodingDomain.THREE_JS,
            capabilities=(RouteCapability.OFFICIAL_DOCS,),
        )

        retrieval_request = build_retrieval_context_request(
            assistant_request,
            route_decision,
        )

        self.assertIsNotNone(retrieval_request)
        assert retrieval_request is not None
        self.assertEqual(retrieval_request.query, assistant_request.query)
        self.assertEqual(retrieval_request.route, RouteName.EXPLAIN)
        self.assertEqual(
            retrieval_request.filters.domain,
            CreativeCodingDomain.THREE_JS,
        )
        self.assertEqual(
            retrieval_request.filters.domains,
            (CreativeCodingDomain.THREE_JS,),
        )

    def test_build_retrieval_request_uses_multi_domain_filters(self) -> None:
        assistant_request = AssistantRequest(
            query="Explain how R3F and GLSL fit together.",
            domains=(
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
            mode=AssistantMode.EXPLAIN,
        )
        route_decision = RouteDecision(
            route=RouteName.EXPLAIN,
            mode=AssistantMode.EXPLAIN,
            domain=None,
            capabilities=(RouteCapability.OFFICIAL_DOCS,),
        )

        retrieval_request = build_retrieval_context_request(
            assistant_request,
            route_decision,
        )

        self.assertIsNotNone(retrieval_request)
        assert retrieval_request is not None
        self.assertIsNone(retrieval_request.filters.domain)
        self.assertEqual(
            retrieval_request.filters.domains,
            (
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
        )

    def test_build_retrieval_request_prefers_explicit_query_domain_over_ui_domains(
        self,
    ) -> None:
        assistant_request = AssistantRequest(
            query="Create a p5.js sketch with a bouncing ball.",
            domains=(
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
            ),
            mode=AssistantMode.GENERATE,
        )
        route_decision = RouteDecision(
            route=RouteName.GENERATE,
            mode=AssistantMode.GENERATE,
            capabilities=(RouteCapability.OFFICIAL_DOCS,),
        )

        retrieval_request = build_retrieval_context_request(
            assistant_request,
            route_decision,
        )

        self.assertIsNotNone(retrieval_request)
        assert retrieval_request is not None
        self.assertEqual(retrieval_request.filters.domain, CreativeCodingDomain.P5_JS)
        self.assertEqual(
            retrieval_request.filters.domains,
            (CreativeCodingDomain.P5_JS,),
        )

    def test_build_retrieval_request_detects_explicit_three_js_domain(self) -> None:
        assistant_request = AssistantRequest(
            query="Create a rotating cube in three.js.",
            domains=(
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
            ),
            mode=AssistantMode.GENERATE,
        )
        route_decision = RouteDecision(
            route=RouteName.GENERATE,
            mode=AssistantMode.GENERATE,
            capabilities=(RouteCapability.OFFICIAL_DOCS,),
        )

        retrieval_request = build_retrieval_context_request(
            assistant_request,
            route_decision,
        )

        self.assertIsNotNone(retrieval_request)
        assert retrieval_request is not None
        self.assertEqual(
            retrieval_request.filters.domain,
            CreativeCodingDomain.THREE_JS,
        )
        self.assertEqual(
            retrieval_request.filters.domains,
            (CreativeCodingDomain.THREE_JS,),
        )

    def test_build_retrieval_request_detects_multi_domain_query_intent(self) -> None:
        assistant_request = AssistantRequest(
            query="Create a shader material in react three fiber using GLSL.",
            domains=(CreativeCodingDomain.THREE_JS,),
            mode=AssistantMode.GENERATE,
        )
        route_decision = RouteDecision(
            route=RouteName.GENERATE,
            mode=AssistantMode.GENERATE,
            capabilities=(RouteCapability.OFFICIAL_DOCS,),
        )

        retrieval_request = build_retrieval_context_request(
            assistant_request,
            route_decision,
        )

        self.assertIsNotNone(retrieval_request)
        assert retrieval_request is not None
        self.assertIsNone(retrieval_request.filters.domain)
        self.assertEqual(
            retrieval_request.filters.domains,
            (
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
        )

    def test_build_retrieval_request_uses_ui_domains_when_query_is_ambiguous(
        self,
    ) -> None:
        assistant_request = AssistantRequest(
            query="Create a rotating cube.",
            domains=(
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
            ),
            mode=AssistantMode.GENERATE,
        )
        route_decision = RouteDecision(
            route=RouteName.GENERATE,
            mode=AssistantMode.GENERATE,
            capabilities=(RouteCapability.OFFICIAL_DOCS,),
        )

        retrieval_request = build_retrieval_context_request(
            assistant_request,
            route_decision,
        )

        self.assertIsNotNone(retrieval_request)
        assert retrieval_request is not None
        self.assertIsNone(retrieval_request.filters.domain)
        self.assertEqual(
            retrieval_request.filters.domains,
            (
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
            ),
        )

    def test_build_retrieval_request_preserves_empty_ui_selection_when_ambiguous(
        self,
    ) -> None:
        assistant_request = AssistantRequest(
            query="Create a rotating cube.",
            domains=(),
            mode=AssistantMode.GENERATE,
        )
        route_decision = RouteDecision(
            route=RouteName.GENERATE,
            mode=AssistantMode.GENERATE,
            capabilities=(RouteCapability.OFFICIAL_DOCS,),
        )

        retrieval_request = build_retrieval_context_request(
            assistant_request,
            route_decision,
        )

        self.assertIsNotNone(retrieval_request)
        assert retrieval_request is not None
        self.assertIsNone(retrieval_request.filters.domain)
        self.assertEqual(retrieval_request.filters.domains, ())

    def test_build_retrieval_request_skips_routes_without_docs_capability(self) -> None:
        assistant_request = AssistantRequest(query="Summarize this sketch.")
        route_decision = RouteDecision(
            route=RouteName.GENERATE,
            mode=AssistantMode.GENERATE,
            capabilities=(RouteCapability.TOOL_USE,),
        )

        retrieval_request = build_retrieval_context_request(
            assistant_request,
            route_decision,
        )

        self.assertIsNone(retrieval_request)

    def test_retrieval_adapter_translates_between_boundaries(self) -> None:
        fake_retriever = _FakeRetriever(
            response=KnowledgeBaseRetrievalResponse(
                request=KnowledgeBaseRetrievalRequest(query="placeholder"),
                results=(
                    KnowledgeBaseSearchResult(
                        record_id="kb_official_docs:official_doc_chunk:v1:one",
                        source_id="three_docs",
                        domain=CreativeCodingDomain.THREE_JS,
                        source_type=OfficialSourceType.API_REFERENCE,
                        publisher="three.js",
                        registry_title="three.js Documentation",
                        document_title="Camera",
                        source_url="https://threejs.org/docs/",
                        resolved_url="https://threejs.org/docs/",
                        chunk_index=0,
                        text="PerspectiveCamera defines frustum settings.",
                        char_count=44,
                        content_hash=_digest("content"),
                        chunk_hash=_digest("chunk"),
                        distance=0.1,
                        score=0.909090909,
                    ),
                ),
            )
        )
        adapter = KnowledgeBaseRetrievalAdapter(retriever=fake_retriever)
        request = RetrievalContextRequest(
            query="camera",
            route=RouteName.EXPLAIN,
        )

        context = adapter.retrieve_context(request)

        self.assertEqual(len(fake_retriever.requests), 1)
        self.assertEqual(fake_retriever.requests[0].query, "camera")
        self.assertEqual(
            fake_retriever.requests[0].filters.domains,
            (),
        )
        self.assertEqual(context.source, RetrievalContextSource.OFFICIAL_KB)
        self.assertEqual(
            context.chunks[0].excerpt,
            "PerspectiveCamera defines frustum settings.",
        )

    def test_retrieval_adapter_preserves_multi_domain_filters(self) -> None:
        fake_retriever = _FakeRetriever(
            response=KnowledgeBaseRetrievalResponse(
                request=KnowledgeBaseRetrievalRequest(query="placeholder"),
                results=(),
            )
        )
        adapter = KnowledgeBaseRetrievalAdapter(retriever=fake_retriever)
        request = RetrievalContextRequest(
            query="shaders",
            route=RouteName.EXPLAIN,
            filters=RetrievalContextFilter(
                domains=(
                    CreativeCodingDomain.REACT_THREE_FIBER,
                    CreativeCodingDomain.GLSL,
                ),
            ),
        )

        adapter.retrieve_context(request)

        self.assertEqual(len(fake_retriever.requests), 1)
        self.assertIsNone(fake_retriever.requests[0].filters.domain)
        self.assertEqual(
            fake_retriever.requests[0].filters.domains,
            (
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
        )

    def test_service_emits_retrieval_events_when_gateway_present(self) -> None:
        retrieval_context = RetrievalContextResponse(
            request=RetrievalContextRequest(
                query="Explain lighting.",
                route=RouteName.EXPLAIN,
            ),
            chunks=(
                RetrievedKnowledgeChunk(
                    source_id="three_docs",
                    domain=CreativeCodingDomain.THREE_JS,
                    source_type=OfficialSourceType.API_REFERENCE,
                    publisher="three.js",
                    registry_title="three.js Documentation",
                    document_title="Lighting",
                    source_url="https://threejs.org/docs/",
                    resolved_url="https://threejs.org/docs/",
                    chunk_index=0,
                    excerpt="AmbientLight adds constant illumination.",
                    score=0.8,
                ),
            ),
        )
        gateway = _FakeGateway(response=retrieval_context)
        service = AssistantService(
            route_fn=_route_with_docs,
            retrieval_gateway=gateway,
        )
        request = AssistantRequest(
            query="Explain lighting.",
            domain=CreativeCodingDomain.THREE_JS,
            mode=AssistantMode.EXPLAIN,
        )

        events = tuple(service.stream(request))

        self.assertEqual(
            [event.event_type for event in events],
            [
                StreamEventType.STATUS,
                StreamEventType.STATUS,
                StreamEventType.RETRIEVAL,
                StreamEventType.RETRIEVAL,
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual(events[2].payload["code"], "retrieval_requested")
        self.assertEqual(events[3].payload["code"], "retrieval_completed")
        self.assertEqual(len(gateway.requests), 1)
        self.assertEqual(
            gateway.requests[0].filters.domain,
            CreativeCodingDomain.THREE_JS,
        )
        self.assertEqual(
            gateway.requests[0].filters.domains,
            (CreativeCodingDomain.THREE_JS,),
        )

    def test_service_preserves_multi_domain_retrieval_filters(self) -> None:
        retrieval_context = RetrievalContextResponse(
            request=RetrievalContextRequest(
                query="Explain shaders.",
                route=RouteName.EXPLAIN,
            ),
        )
        gateway = _FakeGateway(response=retrieval_context)
        service = AssistantService(
            route_fn=_route_with_docs,
            retrieval_gateway=gateway,
        )
        request = AssistantRequest(
            query="Explain how R3F and GLSL work together.",
            domains=(
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
            mode=AssistantMode.EXPLAIN,
        )

        tuple(service.stream(request))

        self.assertEqual(len(gateway.requests), 1)
        self.assertIsNone(gateway.requests[0].filters.domain)
        self.assertEqual(
            gateway.requests[0].filters.domains,
            (
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
        )

    def test_service_prefers_explicit_query_domain_over_ui_domain_selection(
        self,
    ) -> None:
        retrieval_context = RetrievalContextResponse(
            request=RetrievalContextRequest(
                query="Create a p5.js sketch with a bouncing ball.",
                route=RouteName.GENERATE,
            ),
        )
        gateway = _FakeGateway(response=retrieval_context)
        service = AssistantService(
            route_fn=_route_with_docs,
            retrieval_gateway=gateway,
        )
        request = AssistantRequest(
            query="Create a p5.js sketch with a bouncing ball.",
            domains=(
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.REACT_THREE_FIBER,
            ),
            mode=AssistantMode.GENERATE,
        )

        tuple(service.stream(request))

        self.assertEqual(len(gateway.requests), 1)
        self.assertEqual(gateway.requests[0].filters.domain, CreativeCodingDomain.P5_JS)
        self.assertEqual(
            gateway.requests[0].filters.domains,
            (CreativeCodingDomain.P5_JS,),
        )

    def test_assistant_service_skips_retrieval_without_docs_capability(self) -> None:
        gateway = _FakeGateway(
            response=RetrievalContextResponse(
                request=RetrievalContextRequest(
                    query="Unused",
                    route=RouteName.GENERATE,
                ),
            )
        )
        service = AssistantService(
            route_fn=_route_without_docs,
            retrieval_gateway=gateway,
        )
        request = AssistantRequest(query="Generate a scene.")

        events = tuple(service.stream(request))

        self.assertEqual(
            [event.event_type for event in events],
            [
                StreamEventType.STATUS,
                StreamEventType.STATUS,
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual(len(gateway.requests), 0)


def _route_with_docs(request: AssistantRequest) -> RouteDecision:
    return RouteDecision(
        route=RouteName.EXPLAIN,
        mode=request.mode,
        domain=request.domain,
        capabilities=(RouteCapability.OFFICIAL_DOCS,),
    )


def _route_without_docs(request: AssistantRequest) -> RouteDecision:
    return RouteDecision(
        route=RouteName.GENERATE,
        mode=request.mode,
        domain=request.domain,
        capabilities=(RouteCapability.TOOL_USE,),
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _FakeRetriever:
    def __init__(self, *, response: KnowledgeBaseRetrievalResponse) -> None:
        self.response = response
        self.requests: list[KnowledgeBaseRetrievalRequest] = []

    def search(
        self,
        request: KnowledgeBaseRetrievalRequest,
    ) -> KnowledgeBaseRetrievalResponse:
        self.requests.append(request)
        return self.response


class _FakeGateway:
    def __init__(self, *, response: RetrievalContextResponse) -> None:
        self.response = response
        self.requests: list[RetrievalContextRequest] = []

    def retrieve_context(
        self,
        request: RetrievalContextRequest,
    ) -> RetrievalContextResponse:
        self.requests.append(request)
        return self.response


if __name__ == "__main__":
    unittest.main()
