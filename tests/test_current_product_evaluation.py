from __future__ import annotations

import hashlib
import json
import os
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativityProfile,
    GenerationControls,
)
from creative_coding_assistant.core import Settings
from creative_coding_assistant.eval.current_product import (
    _RETRIEVAL_PIPELINE_MODULES,
    AssistantServiceCaseExecutor,
    CurrentProductCaseExecution,
    CurrentProductEvaluationBlockedError,
    CurrentProductEvaluationRunner,
    CurrentProductRunOptions,
    _CapturingGenerationProvider,
    _validate_benchmark_generation_request,
    _validate_prompt_official_chunks,
    build_safe_current_product_evidence,
    write_safe_current_product_evidence,
)
from creative_coding_assistant.eval.current_product_cli import (
    _disable_secondary_observability,
    _resolve_private_diagnostic_path,
)
from creative_coding_assistant.eval.current_product_cli import (
    main as current_product_cli_main,
)
from creative_coding_assistant.eval.live_session import (
    LiveSessionEvalSample,
    LiveSessionProviderMetadata,
    LiveSessionRetrievedContext,
    LiveSessionRouteMetadata,
)
from creative_coding_assistant.eval.ragas_runner import RagasProviderBoundaryError
from creative_coding_assistant.eval.retrieval_demo_pack import (
    CURRENT_PRODUCT_RETRIEVAL_METRICS,
    RetrievalDemoScenario,
    build_capstone_retrieval_demo_pack,
)
from creative_coding_assistant.rag import get_official_source


class _FakeCaseExecutor:
    def __init__(self, *, resolved_model: str = "gpt-5-mini") -> None:
        self.executed: list[str] = []
        self.resolved_model = resolved_model

    def execute(
        self,
        scenario: RetrievalDemoScenario,
    ) -> CurrentProductCaseExecution:
        self.executed.append(scenario.demo_id)
        source = get_official_source(scenario.reference_source_ids[0])
        timestamp = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        sample = LiveSessionEvalSample(
            sample_id=f"sample-{scenario.demo_id}",
            question=scenario.query,
            answer=f"Current generated answer for {scenario.demo_id}.",
            route=LiveSessionRouteMetadata(
                mode=AssistantMode.EXPLAIN,
                domains=scenario.domains,
            ),
            retrieved_contexts=(
                LiveSessionRetrievedContext(
                    source_id=source.source_id,
                    domain=source.domain,
                    source_type=source.source_type,
                    publisher=source.publisher,
                    registry_title=source.title,
                    document_title=source.title,
                    source_url=source.url,
                    chunk_index=0,
                    excerpt=scenario.reference_context[0],
                    score=0.9,
                ),
            ),
            provider_metadata=LiveSessionProviderMetadata(
                provider="openai",
                model=self.resolved_model,
            ),
            started_at=timestamp,
            completed_at=timestamp,
            recorded_at=timestamp,
        )
        return CurrentProductCaseExecution(
            sample=sample,
            prompt_fingerprint=("sha256:" + hashlib.sha256(scenario.demo_id.encode("utf-8")).hexdigest()),
        )


class _FakeEvaluator:
    def __init__(self, *, missing_metric: str | None = None) -> None:
        self.missing_metric = missing_metric
        self.rows = ()
        self.metrics = ()

    def evaluate(self, rows, metric_names):
        self.rows = tuple(rows)
        self.metrics = tuple(metric_names)
        return tuple({metric: None if metric == self.missing_metric else 0.8 for metric in metric_names} for _ in rows)


class _OutOfRangeEvaluator:
    def evaluate(self, rows, metric_names):
        return tuple({metric: 1.2 if metric == "context_precision" else 0.8 for metric in metric_names} for _ in rows)


class _UnavailableEvaluator:
    def evaluate(self, rows, metric_names):
        del rows, metric_names
        raise RagasProviderBoundaryError("provider unavailable")


class _DelegateProvider:
    def __init__(self) -> None:
        self.calls = 0

    def stream(self, request):
        del request
        self.calls += 1
        return ()


class _FingerprintableInput:
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier

    def model_dump(self, *, mode: str):
        assert mode == "json"
        return {"identifier": self.identifier}


class _ExecutorService:
    def __init__(self, *, recorder, provider, input_count: int) -> None:
        self._recorder = recorder
        self._provider = provider
        self._input_count = input_count

    def respond(self, request):
        del request
        self._provider.inputs.extend(
            _FingerprintableInput(f"prompt-{index}")
            for index in range(self._input_count)
        )
        self._recorder.samples.append(SimpleNamespace(sample_id="sample"))
        return SimpleNamespace(events=())


class CurrentProductEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            _env_file=None,
            openai_api_key="test-key",
            openai_model="gpt-5-mini",
            eval_ragas_model="gpt-4o-mini",
        )

    def _completed_result(self):
        return CurrentProductEvaluationRunner(
            settings=self.settings,
            case_executor_factory=lambda _: _FakeCaseExecutor(),
            evaluator_factory=lambda _: _FakeEvaluator(),
            kb_fingerprint_factory=lambda _: "sha256:" + "a" * 64,
            now=lambda: datetime(2026, 7, 14, 12, 30, tzinfo=UTC),
        ).run(
            run_id="run-current",
            options=CurrentProductRunOptions(
                scope="rag",
                allow_provider_calls=True,
                dry_run=False,
            ),
        )

    def test_full_run_scores_current_outputs_and_safe_projection_drops_raw_text(
        self,
    ) -> None:
        executor = _FakeCaseExecutor()
        evaluator = _FakeEvaluator()
        completed_at = datetime(2026, 7, 14, 12, 30, tzinfo=UTC)
        runner = CurrentProductEvaluationRunner(
            settings=self.settings,
            case_executor_factory=lambda _: executor,
            evaluator_factory=lambda _: evaluator,
            kb_fingerprint_factory=lambda _: "sha256:" + "a" * 64,
            now=lambda: completed_at,
        )

        result = runner.run(
            run_id="run-current",
            options=CurrentProductRunOptions(
                scope="rag",
                allow_provider_calls=True,
                dry_run=False,
            ),
        )
        safe = build_safe_current_product_evidence(result)
        serialized = json.dumps(safe)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.score_origin, "current_product")
        self.assertEqual(result.retrieval_score, 0.8)
        self.assertEqual(result.evaluated_at, completed_at)
        self.assertEqual(len(result.case_results), 7)
        self.assertEqual(evaluator.metrics, CURRENT_PRODUCT_RETRIEVAL_METRICS)
        self.assertTrue(all(row.ground_truth for row in evaluator.rows))
        self.assertIn("context_recall", safe["metricScores"])
        self.assertEqual(safe["totalSamples"], 7)
        self.assertEqual(safe["eligibleSamples"], 7)
        self.assertIn("gpt-4o-mini", safe["evaluator"])
        self.assertNotIn("generatedAnswer", serialized)
        self.assertNotIn("referenceAnswer", serialized)
        self.assertNotIn("referenceContext", serialized)
        self.assertNotIn("retrievedContexts", serialized)
        self.assertNotIn("Current generated answer", serialized)

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evidence.json"
            write_safe_current_product_evidence(result, path)
            persisted = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["scoreOrigin"], "current_product")
        with TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "provider_evaluator_identity"):
                write_safe_current_product_evidence(
                    result.model_copy(update={"provider": None}),
                    Path(temp_dir) / "missing-provider.json",
                )

    def test_partial_metric_contract_never_promotes_current_score_or_evidence(
        self,
    ) -> None:
        evaluator = _FakeEvaluator(missing_metric="context_recall")
        runner = CurrentProductEvaluationRunner(
            settings=self.settings,
            case_executor_factory=lambda _: _FakeCaseExecutor(),
            evaluator_factory=lambda _: evaluator,
            kb_fingerprint_factory=lambda _: "sha256:" + "a" * 64,
        )

        result = runner.run(
            run_id="run-partial",
            options=CurrentProductRunOptions(
                scope="full",
                allow_provider_calls=True,
                dry_run=False,
            ),
        )

        self.assertEqual(result.score_origin, "unscored")
        self.assertIsNone(result.retrieval_score)
        self.assertEqual(result.metric_failures, 7)
        with TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "fully scored"):
                write_safe_current_product_evidence(
                    result,
                    Path(temp_dir) / "must-not-write.json",
                )

    def test_canonical_writer_rejects_tampered_dataset_lineage_and_arithmetic(
        self,
    ) -> None:
        result = self._completed_result()
        changed_metric_scores = dict(result.metric_scores)
        changed_metric_scores["faithfulness"] = 0.79
        first_case = result.case_results[0]
        tampered_results = (
            (
                result.model_copy(
                    update={"dataset_fingerprint": "sha256:" + "b" * 64}
                ),
                "immutable_dataset_contract",
            ),
            (
                result.model_copy(update={"privacy_class": "private"}),
                "immutable_dataset_contract",
            ),
            (
                result.model_copy(update={"metric_scores": changed_metric_scores}),
                "aggregate_metric_arithmetic",
            ),
            (
                result.model_copy(update={"retrieval_score": 0.79}),
                "retrieval_score_arithmetic",
            ),
            (
                result.model_copy(
                    update={
                        "case_results": (
                            first_case.model_copy(
                                update={"source_ids": ("not_an_official_source",)}
                            ),
                            *result.case_results[1:],
                        )
                    }
                ),
                "official_source_lineage",
            ),
            (
                result.model_copy(
                    update={
                        "case_results": (
                            first_case.model_copy(
                                update={"expected_domains": ("tone_js",)}
                            ),
                            *result.case_results[1:],
                        )
                    }
                ),
                "immutable_case_contract",
            ),
        )

        with TemporaryDirectory() as temp_dir:
            for index, (tampered, expected_issue) in enumerate(tampered_results):
                with self.subTest(expected_issue=expected_issue):
                    with self.assertRaisesRegex(ValueError, expected_issue):
                        write_safe_current_product_evidence(
                            tampered,
                            Path(temp_dir) / f"tampered-{index}.json",
                        )

    def test_canonical_writer_recomputes_derivable_fingerprints(self) -> None:
        result = self._completed_result()
        changed_fingerprint = "sha256:" + "c" * 64
        tampered_results = [
            result.model_copy(update={field: changed_fingerprint})
            for field in (
                "retrieval_fingerprint",
                "prompt_fingerprint",
                "generation_fingerprint",
                "output_fingerprint",
                "selection_fingerprint",
            )
        ]
        first_case = result.case_results[0]
        tampered_results.append(
            result.model_copy(
                update={
                    "case_results": (
                        first_case.model_copy(
                            update={"generation_fingerprint": changed_fingerprint}
                        ),
                        *result.case_results[1:],
                    )
                }
            )
        )
        assert result.generation_configuration is not None
        tampered_results.append(
            result.model_copy(
                update={
                    "generation_configuration": (
                        result.generation_configuration.model_copy(
                            update={
                                "timeout_seconds": (
                                    result.generation_configuration.timeout_seconds
                                    + 1
                                )
                            }
                        )
                    )
                }
            )
        )

        with TemporaryDirectory() as temp_dir:
            for index, tampered in enumerate(tampered_results):
                with self.subTest(index=index):
                    with self.assertRaisesRegex(
                        ValueError,
                        "derivable_fingerprint_arithmetic",
                    ):
                        write_safe_current_product_evidence(
                            tampered,
                            Path(temp_dir) / f"fingerprint-{index}.json",
                        )

    def test_canonical_writer_accepts_provider_resolved_model_identity(self) -> None:
        result = CurrentProductEvaluationRunner(
            settings=self.settings,
            case_executor_factory=lambda _: _FakeCaseExecutor(
                resolved_model="gpt-5-mini-2025-08-07"
            ),
            evaluator_factory=lambda _: _FakeEvaluator(),
            kb_fingerprint_factory=lambda _: "sha256:" + "a" * 64,
        ).run(
            run_id="run-resolved-model",
            options=CurrentProductRunOptions(
                scope="rag",
                allow_provider_calls=True,
                dry_run=False,
            ),
        )

        self.assertEqual(result.generation_configuration.model, "gpt-5-mini")
        self.assertEqual(result.generation_model, "gpt-5-mini-2025-08-07")
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "resolved-model.json"
            write_safe_current_product_evidence(result, path)
            self.assertTrue(path.exists())

    def test_canonical_writer_validates_payload_against_committed_schema(
        self,
    ) -> None:
        result = self._completed_result()
        invalid_payload = build_safe_current_product_evidence(result)
        invalid_payload["rawAnswer"] = "forbidden raw text"

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid-schema.json"
            with (
                mock.patch(
                    "creative_coding_assistant.eval.current_product."
                    "build_safe_current_product_evidence",
                    return_value=invalid_payload,
                ),
                self.assertRaisesRegex(ValueError, "committed JSON schema"),
            ):
                write_safe_current_product_evidence(result, path)

            self.assertFalse(path.exists())

    def test_cli_routes_private_complete_result_through_canonical_writer(
        self,
    ) -> None:
        result = self._completed_result()
        self.assertTrue(result.case_results[0].generated_answer)
        self.assertTrue(result.case_results[0].retrieved_contexts)

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "canonical.json"
            with (
                mock.patch(
                    "creative_coding_assistant.eval.current_product_cli.load_settings",
                    return_value=self.settings,
                ),
                mock.patch(
                    "creative_coding_assistant.eval.current_product_cli."
                    "CurrentProductEvaluationRunner"
                ) as runner_type,
                mock.patch(
                    "creative_coding_assistant.eval.current_product_cli."
                    "write_safe_current_product_evidence",
                    wraps=write_safe_current_product_evidence,
                ) as canonical_writer,
                mock.patch(
                    "creative_coding_assistant.eval.current_product_cli."
                    "_disable_secondary_observability"
                ),
                mock.patch("builtins.print"),
            ):
                runner_type.return_value.run.return_value = result
                exit_code = current_product_cli_main(
                    (
                        "--allow-provider-calls",
                        "--publish-canonical",
                        "--output-path",
                        str(path),
                        "--run-id",
                        result.run_id,
                    )
                )

            self.assertEqual(exit_code, 0)
            canonical_writer.assert_called_once_with(result, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            serialized = json.dumps(payload)
            self.assertEqual(payload["runId"], result.run_id)
            self.assertNotIn("generatedAnswer", serialized)
            self.assertNotIn("retrievedContexts", serialized)

    def test_systemic_evaluator_failure_blocks_instead_of_completing(self) -> None:
        runner = CurrentProductEvaluationRunner(
            settings=self.settings,
            case_executor_factory=lambda _: _FakeCaseExecutor(),
            evaluator_factory=lambda _: _UnavailableEvaluator(),
            kb_fingerprint_factory=lambda _: "sha256:" + "a" * 64,
        )

        with self.assertRaisesRegex(
            CurrentProductEvaluationBlockedError,
            "no completed score",
        ):
            runner.run(
                run_id="run-provider-failure",
                options=CurrentProductRunOptions(
                    scope="full",
                    allow_provider_calls=True,
                    dry_run=False,
                ),
            )

    def test_out_of_range_metric_never_promotes_current_score(self) -> None:
        result = CurrentProductEvaluationRunner(
            settings=self.settings,
            case_executor_factory=lambda _: _FakeCaseExecutor(),
            evaluator_factory=lambda _: _OutOfRangeEvaluator(),
            kb_fingerprint_factory=lambda _: "sha256:" + "a" * 64,
        ).run(
            run_id="run-out-of-range",
            options=CurrentProductRunOptions(
                scope="full",
                allow_provider_calls=True,
                dry_run=False,
            ),
        )

        self.assertEqual(result.score_origin, "unscored")
        self.assertIsNone(result.retrieval_score)
        self.assertEqual(result.metric_failures, 7)

    def test_subset_run_is_diagnostic_only_and_unknown_case_is_rejected(self) -> None:
        case_id = build_capstone_retrieval_demo_pack().scenarios[0].demo_id
        runner = CurrentProductEvaluationRunner(
            settings=self.settings,
            case_executor_factory=lambda _: _FakeCaseExecutor(),
            evaluator_factory=lambda _: _FakeEvaluator(),
            kb_fingerprint_factory=lambda _: "sha256:" + "a" * 64,
        )

        result = runner.run(
            run_id="run-subset",
            options=CurrentProductRunOptions(
                scope="cases",
                case_ids=(case_id,),
                allow_provider_calls=True,
                dry_run=False,
            ),
        )

        self.assertEqual(result.result_rows, 1)
        self.assertEqual(result.score_origin, "unscored")
        self.assertIsNone(result.retrieval_score)
        with self.assertRaisesRegex(ValueError, "Unknown current-product"):
            runner.run(
                run_id="run-unknown",
                options=CurrentProductRunOptions(
                    scope="cases",
                    case_ids=("unknown_case",),
                    dry_run=True,
                ),
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            runner.run(
                run_id="run-empty",
                options=CurrentProductRunOptions(scope="cases", dry_run=True),
            )

    def test_dry_run_does_not_construct_provider_bound_dependencies(self) -> None:
        def forbidden(*args):
            del args
            raise AssertionError("provider-bound factory was called")

        result = CurrentProductEvaluationRunner(
            settings=Settings(_env_file=None),
            case_executor_factory=forbidden,
            evaluator_factory=forbidden,
            kb_fingerprint_factory=forbidden,
        ).run(
            run_id="run-dry",
            options=CurrentProductRunOptions(scope="rag", dry_run=True),
        )

        self.assertEqual(result.status, "prepared")
        self.assertEqual(result.score_origin, "unscored")
        self.assertEqual(result.metric_scores, {})
        self.assertEqual(result.case_results, ())

    def test_live_runner_forces_secondary_observability_off_for_all_factories(self) -> None:
        received_settings: list[Settings] = []
        settings = self.settings.model_copy(
            update={
                "langsmith_tracing": True,
                "langsmith_api_key": "secondary-secret",
            }
        )

        def case_factory(factory_settings: Settings) -> _FakeCaseExecutor:
            received_settings.append(factory_settings)
            return _FakeCaseExecutor()

        def evaluator_factory(factory_settings: Settings) -> _FakeEvaluator:
            received_settings.append(factory_settings)
            return _FakeEvaluator()

        CurrentProductEvaluationRunner(
            settings=settings,
            case_executor_factory=case_factory,
            evaluator_factory=evaluator_factory,
            kb_fingerprint_factory=lambda _: "sha256:" + "a" * 64,
        ).run(
            run_id="run-no-secondary-observability",
            options=CurrentProductRunOptions(
                scope="full",
                allow_provider_calls=True,
                dry_run=False,
            ),
        )

        self.assertTrue(settings.langsmith_tracing)
        self.assertTrue(all(not item.langsmith_tracing for item in received_settings))
        self.assertTrue(all(item.get_langsmith_api_key() is None for item in received_settings))

    def test_case_executor_captures_initial_prompt_and_two_product_refinements(
        self,
    ) -> None:
        scenario = build_capstone_retrieval_demo_pack().scenarios[0]
        recorder = SimpleNamespace(samples=[])
        provider = SimpleNamespace(inputs=[])
        executor = object.__new__(AssistantServiceCaseExecutor)
        executor._recorder = recorder
        executor._provider = provider
        executor._service = _ExecutorService(
            recorder=recorder,
            provider=provider,
            input_count=3,
        )

        result = executor.execute(scenario)

        self.assertEqual(result.sample.sample_id, "sample")
        self.assertRegex(result.prompt_fingerprint, r"^sha256:[0-9a-f]{64}$")

    def test_case_executor_rejects_prompts_beyond_product_refinement_limit(
        self,
    ) -> None:
        scenario = build_capstone_retrieval_demo_pack().scenarios[0]
        recorder = SimpleNamespace(samples=[])
        provider = SimpleNamespace(inputs=[])
        executor = object.__new__(AssistantServiceCaseExecutor)
        executor._recorder = recorder
        executor._provider = provider
        executor._service = _ExecutorService(
            recorder=recorder,
            provider=provider,
            input_count=4,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "at most 2 bounded refinement prompts",
        ):
            executor.execute(scenario)

    def test_invalid_context_is_rejected_before_generation_delegate(self) -> None:
        delegate = _DelegateProvider()
        provider = _CapturingGenerationProvider(delegate)
        source = get_official_source("p5_reference")
        invalid_chunk = SimpleNamespace(
            source_id=source.source_id,
            domain=source.domain,
            publisher="Unapproved publisher",
            registry_title=source.title,
            source_url=source.url,
        )
        generation_input = SimpleNamespace(
            request=SimpleNamespace(
                rendered_prompt=SimpleNamespace(
                    request=SimpleNamespace(
                        prompt_input=SimpleNamespace(retrieval_input=SimpleNamespace(chunks=(invalid_chunk,)))
                    )
                )
            )
        )

        with self.assertRaisesRegex(ValueError, "official registry"):
            provider.stream(generation_input)

        self.assertEqual(delegate.calls, 0)
        self.assertEqual(provider.inputs, [])

    def test_generation_boundary_accepts_only_frozen_public_benchmark_state(
        self,
    ) -> None:
        scenario = build_capstone_retrieval_demo_pack().scenarios[0]
        assistant_request = AssistantRequest(
            query=scenario.query,
            domains=scenario.domains,
            mode=AssistantMode.EXPLAIN,
            generation_controls=GenerationControls(profile=CreativityProfile.CONTROLLED),
        )
        prompt_input = SimpleNamespace(
            request=SimpleNamespace(assistant_request=assistant_request),
            user_input=SimpleNamespace(
                query=scenario.query,
                is_follow_up=False,
                image_references=(),
                artifact_refinement=None,
                clarification_response=None,
            ),
            memory_input=None,
        )
        generation_input = SimpleNamespace(
            request=SimpleNamespace(
                rendered_prompt=SimpleNamespace(request=SimpleNamespace(prompt_input=prompt_input))
            ),
            image_inputs=(),
        )

        _validate_benchmark_generation_request(generation_input)

        prompt_input.request.assistant_request = assistant_request.model_copy(
            update={"conversation_id": "private-session"}
        )
        with self.assertRaisesRegex(ValueError, "public benchmark contract"):
            _validate_benchmark_generation_request(generation_input)

    def test_official_lineage_alone_cannot_bypass_public_text_allowlist(self) -> None:
        source = get_official_source("p5_reference")
        chunk = SimpleNamespace(
            source_id=source.source_id,
            domain=source.domain,
            publisher=source.publisher,
            registry_title=source.title,
            source_url=source.url,
            excerpt="Unvalidated local text with plausible official metadata.",
        )

        with self.assertRaisesRegex(ValueError, "validated public KB snapshot"):
            _validate_prompt_official_chunks(
                (chunk,),
                allowed_public_chunks=frozenset(),
            )

    def test_live_cli_disables_secondary_observability_credentials(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "LANGCHAIN_API_KEY": "secondary-secret",
                "LANGSMITH_API_KEY": "secondary-secret",
                "LANGSMITH_TRACING": "true",
            },
            clear=False,
        ):
            _disable_secondary_observability()

            self.assertEqual(os.environ["LANGSMITH_TRACING"], "false")
            self.assertEqual(os.environ["LANGCHAIN_TRACING_V2"], "false")
            self.assertEqual(os.environ["RAGAS_DO_NOT_TRACK"], "true")
            self.assertEqual(os.environ["ANONYMIZED_TELEMETRY"], "False")
            self.assertEqual(os.environ["OTEL_SDK_DISABLED"], "true")
            self.assertNotIn("LANGCHAIN_API_KEY", os.environ)
            self.assertNotIn("LANGSMITH_API_KEY", os.environ)

    def test_retrieval_fingerprint_surface_includes_postprocessing_modules(self) -> None:
        expected_modules = {
            "creative_coding_assistant.app.bootstrap",
            "creative_coding_assistant.orchestration.runtime.routing",
            "creative_coding_assistant.orchestration.runtime.retrieval",
            "creative_coding_assistant.rag.embeddings.openai",
            "creative_coding_assistant.rag.retrieval.embedder",
            "creative_coding_assistant.rag.retrieval.factory",
            "creative_coding_assistant.rag.retrieval.models",
            "creative_coding_assistant.rag.retrieval.openai_embedder",
            "creative_coding_assistant.rag.retrieval.postprocess",
            "creative_coding_assistant.rag.retrieval.domain_intent",
            "creative_coding_assistant.rag.sources",
            "creative_coding_assistant.vectorstore.client",
            "creative_coding_assistant.vectorstore.repository",
        }

        self.assertTrue(expected_modules.issubset(_RETRIEVAL_PIPELINE_MODULES))

    def test_private_diagnostics_are_constrained_to_ignored_data_eval(self) -> None:
        accepted = _resolve_private_diagnostic_path(Path("data/eval/baseline-private.json"))

        assert accepted is not None
        self.assertTrue(accepted.as_posix().endswith("data/eval/baseline-private.json"))
        with self.assertRaisesRegex(ValueError, "data/eval"):
            _resolve_private_diagnostic_path(Path("demo/evaluation/private-must-not-appear.json"))

    def test_public_evidence_schema_forbids_raw_case_text_fields(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        schema = json.loads(
            (repository_root / "demo/evaluation/current_product_ragas_evidence.schema.json").read_text(encoding="utf-8")
        )
        case_contract = schema["$defs"]["caseResult"]

        self.assertFalse(case_contract["additionalProperties"])
        self.assertNotIn("generatedAnswer", case_contract["properties"])
        self.assertNotIn("referenceAnswer", case_contract["properties"])
        self.assertNotIn("referenceContext", case_contract["properties"])
        self.assertNotIn("retrievedContexts", case_contract["properties"])


if __name__ == "__main__":
    unittest.main()
