import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEventType,
)
from creative_coding_assistant.orchestration import (
    AssistantService,
    DomainSelectionShape,
    RouteCapability,
    RouteDecision,
    RouteName,
    StreamEventBuilder,
    route_request,
)


class CoreBackendServiceTests(unittest.TestCase):
    def test_route_request_maps_mode_to_explicit_route(self) -> None:
        request = AssistantRequest(
            query="Preview this p5.js sketch.",
            domain=CreativeCodingDomain.P5_JS,
            mode=AssistantMode.PREVIEW,
        )

        decision = route_request(request)

        self.assertEqual(decision.route, RouteName.PREVIEW)
        self.assertEqual(decision.mode, AssistantMode.PREVIEW)
        self.assertEqual(decision.domain, CreativeCodingDomain.P5_JS)
        self.assertEqual(decision.domains, (CreativeCodingDomain.P5_JS,))
        self.assertEqual(decision.domain_selection, DomainSelectionShape.SINGLE)
        self.assertIn(RouteCapability.PREVIEW_ARTIFACTS, decision.capabilities)
        self.assertIn(RouteCapability.LIVE_EVALUATION, decision.capabilities)

    def test_route_request_exposes_empty_domain_selection(self) -> None:
        decision = route_request(
            AssistantRequest(
                query="Explain this sketch.",
                domains=(),
                mode=AssistantMode.EXPLAIN,
            )
        )

        self.assertIsNone(decision.domain)
        self.assertEqual(decision.domains, ())
        self.assertEqual(decision.domain_selection, DomainSelectionShape.NONE)

    def test_route_request_exposes_multi_domain_selection(self) -> None:
        decision = route_request(
            AssistantRequest(
                query="Explain how R3F and GLSL fit together.",
                domains=(
                    CreativeCodingDomain.REACT_THREE_FIBER,
                    CreativeCodingDomain.GLSL,
                ),
                mode=AssistantMode.EXPLAIN,
            )
        )

        self.assertIsNone(decision.domain)
        self.assertEqual(
            decision.domains,
            (
                CreativeCodingDomain.REACT_THREE_FIBER,
                CreativeCodingDomain.GLSL,
            ),
        )
        self.assertEqual(decision.domain_selection, DomainSelectionShape.MULTI)

    def test_route_decision_normalizes_legacy_domain_to_single_selection(self) -> None:
        decision = RouteDecision(
            route=RouteName.DEBUG,
            mode=AssistantMode.DEBUG,
            domain=CreativeCodingDomain.THREE_JS,
            capabilities=(RouteCapability.TOOL_USE,),
        )

        self.assertEqual(decision.domain, CreativeCodingDomain.THREE_JS)
        self.assertEqual(decision.domains, (CreativeCodingDomain.THREE_JS,))
        self.assertEqual(decision.domain_selection, DomainSelectionShape.SINGLE)

    def test_event_builder_assigns_monotonic_sequences(self) -> None:
        builder = StreamEventBuilder()

        events = (
            builder.status(code="request_received", message="Request accepted."),
            builder.token_delta("Hello"),
            builder.final(answer="Done"),
        )

        self.assertEqual([event.sequence for event in events], [0, 1, 2])
        self.assertEqual(events[1].event_type, StreamEventType.TOKEN_DELTA)
        self.assertEqual(events[2].payload["answer"], "Done")

    def test_assistant_service_streams_route_and_final_events(self) -> None:
        service = AssistantService()
        request = AssistantRequest(
            query="Generate a Three.js particle field.",
            domain=CreativeCodingDomain.THREE_JS,
            mode=AssistantMode.GENERATE,
        )

        events = tuple(service.stream(request))

        self.assertEqual(
            [event.event_type for event in events],
            [
                StreamEventType.STATUS,
                StreamEventType.STATUS,
                StreamEventType.FINAL,
            ],
        )
        self.assertEqual(events[0].payload["code"], "request_received")
        self.assertEqual(events[1].payload["route"]["route"], "generate")
        self.assertEqual(events[1].payload["route"]["domains"], ["three_js"])
        self.assertEqual(events[1].payload["route"]["domain_selection"], "single")
        self.assertIn("generate route", events[2].payload["answer"])

    def test_assistant_service_collects_streamed_response(self) -> None:
        service = AssistantService()
        request = AssistantRequest(query="Explain a GLSL fragment shader.")

        response = service.respond(request)

        self.assertEqual(len(response.events), 3)
        self.assertEqual(response.events[-1].event_type, StreamEventType.FINAL)
        self.assertEqual(response.answer, response.events[-1].payload["answer"])

    def test_assistant_service_accepts_custom_router(self) -> None:
        def route_debug(request: AssistantRequest) -> RouteDecision:
            return RouteDecision(
                route=RouteName.DEBUG,
                mode=request.mode,
                domain=request.domain,
                capabilities=(RouteCapability.TOOL_USE,),
            )

        service = AssistantService(route_fn=route_debug)
        request = AssistantRequest(query="Why is this shader black?")

        events = tuple(service.stream(request))

        self.assertEqual(events[1].payload["route"]["route"], "debug")
        self.assertEqual(events[1].payload["route"]["capabilities"], ["tool_use"])


if __name__ == "__main__":
    unittest.main()
