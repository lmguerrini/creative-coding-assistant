from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from creative_coding_assistant.contracts import AssistantMode, CreativeCodingDomain
from creative_coding_assistant.core import Settings
from creative_coding_assistant.eval.live_session import (
    LiveSessionEvalSample,
    LiveSessionProviderMetadata,
    LiveSessionRetrievedContext,
    LiveSessionRouteMetadata,
)
from creative_coding_assistant.eval.ragas_cli import main as ragas_cli_main
from creative_coding_assistant.eval.ragas_models import (
    DEFAULT_RAGAS_METRICS,
    SUPPORTED_RAGAS_METRICS,
    RagasLiveEvalDataset,
    load_live_session_samples,
    prepare_ragas_live_eval_dataset,
    resolve_ragas_metric_names,
    select_ragas_live_eval_rows,
)
from creative_coding_assistant.eval.ragas_runner import (
    DefaultRagasEvaluationBackend,
    RagasDependencyError,
    RagasEvaluatorConfig,
    RagasLiveEvalRunManifest,
    RagasLiveEvalRunResult,
    RagasProviderCostBoundaryError,
    ragas_run_manifest_path,
    run_ragas_live_eval,
)
from creative_coding_assistant.orchestration import RouteCapability, RouteName
from creative_coding_assistant.rag.sources import OfficialSourceType

_ORIGINAL_IMPORT = __import__


class RagasLiveEvalFoundationTests(unittest.TestCase):
    def test_select_ragas_rows_skips_samples_without_retrieved_contexts(self) -> None:
        selection = select_ragas_live_eval_rows(
            (
                _sample(sample_id="eligible", with_context=True),
                _sample(sample_id="missing-context", with_context=False),
            )
        )

        self.assertEqual(selection.total_samples, 2)
        self.assertEqual(selection.eligible_samples, 1)
        self.assertEqual(selection.skipped_samples, 1)
        self.assertEqual(selection.rows[0].sample_id, "eligible")
        self.assertEqual(
            selection.rows[0].ragas_payload(),
            {
                "user_input": "How does draw work in p5.js?",
                "response": "draw runs repeatedly after setup.",
                "retrieved_contexts": ["draw() continuously executes code."],
            },
        )
        self.assertEqual(
            selection.skipped[0].reason,
            "missing_retrieved_contexts",
        )

    def test_prepare_ragas_dataset_preserves_optional_eval_metadata(self) -> None:
        created_at = datetime(2026, 4, 29, 12, 30, tzinfo=UTC)
        dataset = prepare_ragas_live_eval_dataset(
            (
                _sample(
                    sample_id="eligible",
                    with_context=True,
                    ground_truth="draw() loops after setup().",
                    with_provider_metadata=True,
                ),
                _sample(sample_id="missing-context", with_context=False),
            ),
            dataset_id="dataset-1",
            source_path=Path("data/eval/live_sessions.jsonl"),
            metric_names=("context_precision", "faithfulness"),
            created_at=created_at,
        )

        self.assertEqual(dataset.dataset_id, "dataset-1")
        self.assertEqual(dataset.created_at, created_at)
        self.assertEqual(dataset.total_samples, 2)
        self.assertEqual(dataset.eligible_samples, 1)
        self.assertEqual(dataset.skipped_samples, 1)
        self.assertEqual(dataset.metrics, ("context_precision", "faithfulness"))
        self.assertEqual(
            dataset.ragas_payloads()[0]["reference"],
            "draw() loops after setup().",
        )
        self.assertEqual(dataset.rows[0].retrieval_scores, (0.91,))
        assert dataset.rows[0].provider_metadata is not None
        self.assertEqual(dataset.rows[0].provider_metadata.provider, "openai")
        self.assertEqual(
            dataset.summary_payload(),
            {
                "dataset_id": "dataset-1",
                "created_at": created_at.isoformat(),
                "source_path": "data/eval/live_sessions.jsonl",
                "total_samples": 2,
                "eligible_samples": 1,
                "skipped_samples": 1,
                "metrics": ["context_precision", "faithfulness"],
            },
        )

    def test_select_ragas_rows_limits_eligible_samples(self) -> None:
        selection = select_ragas_live_eval_rows(
            (
                _sample(sample_id="eligible-1", with_context=True),
                _sample(sample_id="missing-context", with_context=False),
                _sample(sample_id="eligible-2", with_context=True),
            ),
            limit=1,
        )

        self.assertEqual(selection.total_samples, 3)
        self.assertEqual(selection.eligible_samples, 1)
        self.assertEqual(selection.rows[0].sample_id, "eligible-1")
        self.assertEqual(
            [skipped.reason for skipped in selection.skipped],
            ["missing_retrieved_contexts", "limit_exceeded"],
        )

    def test_select_ragas_rows_latest_uses_newest_eligible_samples(self) -> None:
        selection = select_ragas_live_eval_rows(
            (
                _sample(sample_id="eligible-1", with_context=True),
                _sample(sample_id="missing-context", with_context=False),
                _sample(sample_id="eligible-2", with_context=True),
                _sample(sample_id="eligible-3", with_context=True),
            ),
            latest=2,
        )

        self.assertEqual(selection.total_samples, 4)
        self.assertEqual(selection.eligible_samples, 2)
        self.assertEqual(
            [row.sample_id for row in selection.rows],
            ["eligible-2", "eligible-3"],
        )
        self.assertEqual(
            [(skipped.sample_id, skipped.reason) for skipped in selection.skipped],
            [
                ("missing-context", "missing_retrieved_contexts"),
                ("eligible-1", "limit_exceeded"),
            ],
        )

    def test_select_ragas_rows_latest_takes_precedence_over_limit(self) -> None:
        selection = select_ragas_live_eval_rows(
            (
                _sample(sample_id="eligible-1", with_context=True),
                _sample(sample_id="eligible-2", with_context=True),
                _sample(sample_id="eligible-3", with_context=True),
            ),
            limit=1,
            latest=2,
        )

        self.assertEqual(
            [row.sample_id for row in selection.rows],
            ["eligible-2", "eligible-3"],
        )
        self.assertEqual(
            [(skipped.sample_id, skipped.reason) for skipped in selection.skipped],
            [("eligible-1", "limit_exceeded")],
        )

    def test_resolve_ragas_metric_names_deduplicates_and_validates(self) -> None:
        self.assertIn("answer_relevancy", SUPPORTED_RAGAS_METRICS)
        self.assertIn("context_relevancy", SUPPORTED_RAGAS_METRICS)
        self.assertEqual(resolve_ragas_metric_names(None), DEFAULT_RAGAS_METRICS)
        self.assertEqual(
            resolve_ragas_metric_names(
                (
                    "faithfulness",
                    "context_precision",
                    "context_relevancy",
                    "faithfulness",
                )
            ),
            ("faithfulness", "context_precision", "context_relevancy"),
        )
        with self.assertRaises(ValueError):
            resolve_ragas_metric_names(("not-a-metric",))

    def test_load_live_session_samples_reads_jsonl_records(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "live_sessions.jsonl"
            input_path.write_text(
                _sample(sample_id="sample-1", with_context=True).model_dump_json()
                + "\n\n",
                encoding="utf-8",
            )

            samples = load_live_session_samples(input_path)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].sample_id, "sample-1")

    def test_run_ragas_live_eval_writes_safe_result_rows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "live_sessions.jsonl"
            output_path = Path(temp_dir) / "ragas_results.jsonl"
            input_path.write_text(
                "\n".join(
                    (
                        _sample(
                            sample_id="sample-1", with_context=True
                        ).model_dump_json(),
                        _sample(
                            sample_id="sample-2", with_context=False
                        ).model_dump_json(),
                    )
                ),
                encoding="utf-8",
            )

            result = run_ragas_live_eval(
                input_path=input_path,
                output_path=output_path,
                backend=_FakeRagasBackend(),
                run_id="run-1",
                evaluated_at=datetime(2026, 4, 29, 12, 0, tzinfo=UTC),
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            manifest = json.loads(
                ragas_run_manifest_path(output_path).read_text(encoding="utf-8")
            )

        self.assertEqual(result.total_samples, 2)
        self.assertEqual(result.eligible_samples, 1)
        self.assertEqual(result.skipped_samples, 1)
        self.assertFalse(result.dry_run)
        self.assertFalse(result.provider_calls_allowed)
        self.assertEqual(result.metrics, DEFAULT_RAGAS_METRICS)
        self.assertEqual(result.metric_failures, 0)
        self.assertEqual(payload["run_id"], "run-1")
        self.assertEqual(payload["sample_id"], "sample-1")
        self.assertEqual(payload["metrics"]["context_precision"], 0.9)
        self.assertEqual(payload["metric_errors"], {})
        self.assertEqual(payload["source_ids"], ["p5_reference"])
        self.assertEqual(payload["retrieval_scores"], [0.91])
        self.assertNotIn("response", payload)
        self.assertNotIn("retrieved_contexts", payload)
        self.assertNotIn("embedding", json.dumps(payload))
        self.assertEqual(manifest["run_id"], "run-1")
        self.assertEqual(manifest["dataset"]["eligible_samples"], 1)
        self.assertEqual(manifest["result_rows"], 1)

    def test_run_ragas_live_eval_dry_run_persists_manifest_without_results(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "live_sessions.jsonl"
            output_path = Path(temp_dir) / "ragas_results.jsonl"
            input_path.write_text(
                _sample(sample_id="sample-1", with_context=True).model_dump_json(),
                encoding="utf-8",
            )

            result = run_ragas_live_eval(
                input_path=input_path,
                output_path=output_path,
                dry_run=True,
                run_id="dry-run",
                evaluated_at=datetime(2026, 4, 29, 12, 0, tzinfo=UTC),
            )

            manifest = json.loads(
                ragas_run_manifest_path(output_path).read_text(encoding="utf-8")
            )

        self.assertTrue(result.dry_run)
        self.assertEqual(result.result_rows, ())
        self.assertFalse(output_path.exists())
        self.assertEqual(manifest["run_id"], "dry-run")
        self.assertTrue(manifest["dry_run"])
        self.assertIn("no evaluator", manifest["cost_warning"])

    def test_run_ragas_live_eval_requires_provider_opt_in_for_default_backend(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "live_sessions.jsonl"
            output_path = Path(temp_dir) / "ragas_results.jsonl"
            input_path.write_text(
                _sample(sample_id="sample-1", with_context=True).model_dump_json(),
                encoding="utf-8",
            )

            with self.assertRaises(RagasProviderCostBoundaryError):
                run_ragas_live_eval(
                    input_path=input_path,
                    output_path=output_path,
                    run_id="blocked-cost",
                    evaluated_at=datetime(2026, 4, 29, 12, 0, tzinfo=UTC),
                )

    def test_run_ragas_live_eval_latest_keeps_summary_counts_correct(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "live_sessions.jsonl"
            output_path = Path(temp_dir) / "ragas_results.jsonl"
            input_path.write_text(
                "\n".join(
                    (
                        _sample(
                            sample_id="sample-1", with_context=True
                        ).model_dump_json(),
                        _sample(
                            sample_id="sample-2", with_context=False
                        ).model_dump_json(),
                        _sample(
                            sample_id="sample-3", with_context=True
                        ).model_dump_json(),
                        _sample(
                            sample_id="sample-4", with_context=True
                        ).model_dump_json(),
                    )
                ),
                encoding="utf-8",
            )

            result = run_ragas_live_eval(
                input_path=input_path,
                output_path=output_path,
                latest=2,
                backend=_FakeRagasBackend(),
                run_id="run-latest",
                evaluated_at=datetime(2026, 4, 29, 12, 0, tzinfo=UTC),
            )

        self.assertEqual(result.total_samples, 4)
        self.assertEqual(result.eligible_samples, 2)
        self.assertEqual(result.skipped_samples, 2)
        self.assertEqual(
            [row.sample_id for row in result.result_rows],
            ["sample-3", "sample-4"],
        )
        self.assertEqual(
            [(skipped.sample_id, skipped.reason) for skipped in result.skipped],
            [
                ("sample-2", "missing_retrieved_contexts"),
                ("sample-1", "limit_exceeded"),
            ],
        )

    def test_run_ragas_live_eval_records_metric_failures(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "live_sessions.jsonl"
            output_path = Path(temp_dir) / "ragas_results.jsonl"
            input_path.write_text(
                _sample(sample_id="sample-1", with_context=True).model_dump_json(),
                encoding="utf-8",
            )

            result = run_ragas_live_eval(
                input_path=input_path,
                output_path=output_path,
                backend=_FailingMetricBackend(),
                run_id="run-1",
                evaluated_at=datetime(2026, 4, 29, 12, 0, tzinfo=UTC),
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(result.metric_failures, 1)
        self.assertEqual(payload["metrics"]["context_precision"], None)
        self.assertEqual(
            payload["metric_errors"],
            {"context_precision": "metric_returned_null"},
        )

    def test_default_ragas_backend_reports_missing_dependency(self) -> None:
        with patch("builtins.__import__", side_effect=_blocked_ragas_import):
            with self.assertRaises(RagasDependencyError):
                DefaultRagasEvaluationBackend().evaluate(
                    [select_ragas_live_eval_rows((_sample(),)).rows[0]],
                    DEFAULT_RAGAS_METRICS,
                )

    def test_ragas_cli_uses_settings_paths_and_reports_success(self) -> None:
        result = _run_result()

        with (
            patch(
                "creative_coding_assistant.eval.ragas_cli.load_settings",
                return_value=Settings(
                    eval_data_path=Path("custom/live.jsonl"),
                    eval_ragas_results_path=Path("custom/results.jsonl"),
                    eval_ragas_model="custom-eval-model",
                    openai_embedding_model="custom-embedding-model",
                    eval_ragas_timeout_seconds=30,
                    eval_ragas_max_retries=1,
                    eval_ragas_max_workers=1,
                    openai_api_key=None,
                ),
            ),
            patch(
                "creative_coding_assistant.eval.ragas_cli.run_ragas_live_eval",
                return_value=result,
            ) as run_mock,
        ):
            exit_code = ragas_cli_main(
                (
                    "--limit",
                    "2",
                    "--metric",
                    "faithfulness",
                    "--metric",
                    "context_precision",
                )
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            run_mock.call_args.kwargs["input_path"],
            Path("custom/live.jsonl"),
        )
        self.assertEqual(
            run_mock.call_args.kwargs["output_path"],
            Path("custom/results.jsonl"),
        )
        self.assertEqual(run_mock.call_args.kwargs["limit"], 2)
        self.assertEqual(
            run_mock.call_args.kwargs["metric_names"],
            ("faithfulness", "context_precision"),
        )
        evaluator_config = run_mock.call_args.kwargs["evaluator_config"]
        self.assertIsInstance(evaluator_config, RagasEvaluatorConfig)
        self.assertEqual(evaluator_config.model, "custom-eval-model")
        self.assertEqual(evaluator_config.embedding_model, "custom-embedding-model")
        self.assertEqual(evaluator_config.timeout_seconds, 30)
        self.assertEqual(evaluator_config.max_retries, 1)
        self.assertEqual(evaluator_config.max_workers, 1)
        self.assertFalse(run_mock.call_args.kwargs["dry_run"])
        self.assertFalse(run_mock.call_args.kwargs["allow_provider_calls"])

    def test_ragas_cli_passes_latest_selection(self) -> None:
        result = _run_result(run_id="run-latest", total_samples=4, skipped_samples=2)

        with (
            patch(
                "creative_coding_assistant.eval.ragas_cli.load_settings",
                return_value=Settings(),
            ),
            patch(
                "creative_coding_assistant.eval.ragas_cli.run_ragas_live_eval",
                return_value=result,
            ) as run_mock,
        ):
            exit_code = ragas_cli_main(("--limit", "1", "--latest", "2"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_mock.call_args.kwargs["limit"], 1)
        self.assertEqual(run_mock.call_args.kwargs["latest"], 2)

    def test_ragas_cli_passes_dry_run_and_provider_opt_in_flags(self) -> None:
        result = _run_result(run_id="run-dry", dry_run=True)

        with (
            patch(
                "creative_coding_assistant.eval.ragas_cli.load_settings",
                return_value=Settings(),
            ),
            patch(
                "creative_coding_assistant.eval.ragas_cli.run_ragas_live_eval",
                return_value=result,
            ) as run_mock,
        ):
            exit_code = ragas_cli_main(("--dry-run", "--allow-provider-calls"))

        self.assertEqual(exit_code, 0)
        self.assertTrue(run_mock.call_args.kwargs["dry_run"])
        self.assertTrue(run_mock.call_args.kwargs["allow_provider_calls"])

    def test_ragas_cli_reports_missing_optional_dependency(self) -> None:
        with (
            patch(
                "creative_coding_assistant.eval.ragas_cli.load_settings",
                return_value=Settings(),
            ),
            patch(
                "creative_coding_assistant.eval.ragas_cli.run_ragas_live_eval",
                side_effect=RagasDependencyError("install ragas"),
            ),
        ):
            exit_code = ragas_cli_main(())

        self.assertEqual(exit_code, 2)

    def test_ragas_cli_reports_provider_cost_boundary(self) -> None:
        with (
            patch(
                "creative_coding_assistant.eval.ragas_cli.load_settings",
                return_value=Settings(),
            ),
            patch(
                "creative_coding_assistant.eval.ragas_cli.run_ragas_live_eval",
                side_effect=RagasProviderCostBoundaryError("allow provider calls"),
            ),
        ):
            exit_code = ragas_cli_main(())

        self.assertEqual(exit_code, 3)


class _FakeRagasBackend:
    def evaluate(self, rows, metric_names):
        self.rows = rows
        return tuple(
            {
                "faithfulness": 0.8,
                "answer_relevancy": 0.7,
                "context_precision": 0.9,
            }
            for _ in rows
        )


class _FailingMetricBackend:
    def evaluate(self, rows, metric_names):
        self.rows = rows
        return tuple({metric_name: None for metric_name in metric_names} for _ in rows)


def _run_result(
    *,
    run_id: str = "run-1",
    total_samples: int = 3,
    eligible_samples: int = 2,
    skipped_samples: int = 1,
    dry_run: bool = False,
) -> RagasLiveEvalRunResult:
    input_path = Path("data/eval/live_sessions.jsonl")
    output_path = Path("data/eval/ragas_results.jsonl")
    manifest_path = ragas_run_manifest_path(output_path)
    evaluated_at = datetime(2026, 4, 29, 12, 0, tzinfo=UTC)
    cost_warning = (
        "Dry run only: no evaluator LLM or embedding provider calls are made."
        if dry_run
        else (
            "Provider calls disabled: pass --allow-provider-calls only after "
            "reviewing the prepared dataset and expected cost."
        )
    )
    manifest = RagasLiveEvalRunManifest(
        run_id=run_id,
        dataset=RagasLiveEvalDataset(
            dataset_id=run_id,
            created_at=evaluated_at,
            source_path=input_path,
            total_samples=total_samples,
            metrics=DEFAULT_RAGAS_METRICS,
        ),
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        metrics=DEFAULT_RAGAS_METRICS,
        dry_run=dry_run,
        provider_calls_allowed=False,
        cost_warning=cost_warning,
        result_rows=0,
        metric_failures=0,
        evaluated_at=evaluated_at,
    )
    return RagasLiveEvalRunResult(
        run_id=run_id,
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        total_samples=total_samples,
        eligible_samples=eligible_samples,
        skipped_samples=skipped_samples,
        metrics=DEFAULT_RAGAS_METRICS,
        dry_run=dry_run,
        provider_calls_allowed=False,
        cost_warning=cost_warning,
        metric_failures=0,
        result_rows=(),
        skipped=(),
        manifest=manifest,
    )


def _blocked_ragas_import(name, *args, **kwargs):
    if name == "ragas" or name.startswith("ragas."):
        raise ImportError("missing ragas")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


def _sample(
    *,
    sample_id: str = "sample-1",
    with_context: bool = True,
    ground_truth: str | None = None,
    with_provider_metadata: bool = False,
) -> LiveSessionEvalSample:
    return LiveSessionEvalSample(
        sample_id=sample_id,
        question="How does draw work in p5.js?",
        answer="draw runs repeatedly after setup.",
        conversation_id="conversation-1",
        ground_truth=ground_truth,
        route=LiveSessionRouteMetadata(
            route=RouteName.EXPLAIN,
            mode=AssistantMode.EXPLAIN,
            domains=(CreativeCodingDomain.P5_JS,),
            capabilities=(RouteCapability.OFFICIAL_DOCS,),
        ),
        retrieved_contexts=(
            (
                LiveSessionRetrievedContext(
                    source_id="p5_reference",
                    domain=CreativeCodingDomain.P5_JS,
                    source_type=OfficialSourceType.API_REFERENCE,
                    publisher="p5.js",
                    registry_title="p5.js Reference",
                    document_title="draw",
                    source_url="https://p5js.org/reference/",
                    chunk_index=0,
                    excerpt="draw() continuously executes code.",
                    score=0.91,
                ),
            )
            if with_context
            else ()
        ),
        provider_metadata=(
            LiveSessionProviderMetadata(
                provider="openai",
                model="gpt-5-mini",
                response_id="response-1",
                finish_reason="stop",
                token_usage={"total_tokens": 42},
            )
            if with_provider_metadata
            else None
        ),
        started_at=datetime(2026, 4, 29, 11, 0, tzinfo=UTC),
        completed_at=datetime(2026, 4, 29, 11, 0, 2, tzinfo=UTC),
        recorded_at=datetime(2026, 4, 29, 11, 0, 3, tzinfo=UTC),
    )


if __name__ == "__main__":
    unittest.main()
