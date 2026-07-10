import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration.runtime.artifacts import (
    extract_workflow_artifacts,
)
from creative_coding_assistant.orchestration.runtime.product_outcome import (
    derive_product_outcome,
)
from creative_coding_assistant.orchestration.runtime.routing import route_request
from creative_coding_assistant.orchestration.runtime.workflow import (
    AssistantWorkflowState,
    WorkflowFailureInfo,
    WorkflowStatus,
    WorkflowStep,
)


class ProductOutcomeTests(unittest.TestCase):
    def test_code_only_react_three_fiber_preview_request_is_partial(self) -> None:
        request = AssistantRequest(
            query=(
                "Create a runnable browser preview React Three Fiber installation "
                "component."
            ),
            domains=(CreativeCodingDomain.REACT_THREE_FIBER,),
            mode=AssistantMode.GENERATE,
        )
        route_decision = route_request(request)
        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```tsx generated-study.r3f.tsx",
                    'import { Canvas } from "@react-three/fiber";',
                    "export default function Study() { return <Canvas />; }",
                    "```",
                ]
            ),
            request=request,
            route_decision=route_decision,
        )
        state = AssistantWorkflowState(
            request=request,
            route_decision=route_decision,
            status=WorkflowStatus.COMPLETED,
            completed_steps=(
                WorkflowStep.GENERATION,
                WorkflowStep.ARTIFACT_EXTRACTION,
                WorkflowStep.FINALIZATION,
            ),
            artifacts=artifacts,
        )

        outcome = derive_product_outcome(state)

        self.assertEqual(outcome["deliverable_status"], "USABLE")
        self.assertEqual(outcome["artifact_extraction_status"], "EXTRACTED")
        self.assertEqual(outcome["artifact_runnability"], "UNSUPPORTED")
        self.assertEqual(outcome["preview_status"], "UNAVAILABLE")
        self.assertEqual(outcome["runtime_health"], "NOT_AVAILABLE")
        self.assertEqual(outcome["product_outcome"], "PARTIAL")

    def test_browser_safe_p5_request_without_a_runnable_preview_is_partial(self) -> None:
        request = AssistantRequest(
            query=(
                "Create a browser-safe p5.js morphogenesis sketch with setup(), "
                "draw(), and interaction controls."
            ),
            domains=(CreativeCodingDomain.P5_JS,),
            mode=AssistantMode.GENERATE,
        )
        route_decision = route_request(request)
        artifacts = extract_workflow_artifacts(
            "\n".join(
                [
                    "```js generated-sketch-1.p5.js",
                    "const trail = new Float32Array(64);",
                    "function setup() { createCanvas(640, 360); }",
                    "function draw() { background(8); circle(20, 20, 10); }",
                    "```",
                ]
            ),
            request=request,
            route_decision=route_decision,
        )
        state = AssistantWorkflowState(
            request=request,
            route_decision=route_decision,
            status=WorkflowStatus.COMPLETED,
            completed_steps=(
                WorkflowStep.GENERATION,
                WorkflowStep.ARTIFACT_EXTRACTION,
                WorkflowStep.FINALIZATION,
            ),
            artifacts=artifacts,
        )

        outcome = derive_product_outcome(state)

        self.assertFalse(artifacts[0].preview_eligible)
        self.assertEqual(outcome["artifact_runnability"], "UNSUPPORTED")
        self.assertEqual(outcome["preview_status"], "UNAVAILABLE")
        self.assertEqual(outcome["product_outcome"], "PARTIAL")
        self.assertIn("live preview is unavailable", outcome["summary"])

    def test_missing_requested_browser_artifact_is_failure(self) -> None:
        request = AssistantRequest(
            query="Write runnable browser-ready Three.js code.",
            domains=(CreativeCodingDomain.THREE_JS,),
            mode=AssistantMode.GENERATE,
        )
        state = AssistantWorkflowState(
            request=request,
            route_decision=route_request(request),
            status=WorkflowStatus.FAILED,
            completed_steps=(WorkflowStep.GENERATION,),
            failure_info=WorkflowFailureInfo(
                step=WorkflowStep.REVIEW,
                code="required_deliverable_not_produced",
                message="The requested deliverable was not produced.",
            ),
        )

        outcome = derive_product_outcome(state)

        self.assertEqual(outcome["deliverable_status"], "NOT_PRODUCED")
        self.assertEqual(outcome["artifact_extraction_status"], "NOT_PRODUCED")
        self.assertEqual(outcome["artifact_runnability"], "NOT_PRODUCED")
        self.assertEqual(outcome["preview_status"], "PENDING")
        self.assertEqual(outcome["product_outcome"], "FAILURE")
        self.assertIn("not produced", outcome["summary"])


if __name__ == "__main__":
    unittest.main()
