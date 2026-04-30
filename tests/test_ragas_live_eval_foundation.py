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
    LiveSessionRetrievedContext,
    LiveSessionRouteMetadata,
)
from creative_coding_assistant.eval.ragas_cli import main as ragas_cli_main
from creative_coding_assistant.eval.ragas_models import (
    DEFAULT_RAGAS_METRICS,
    SUPPORTED_RAGAS_METRICS,
    load_live_session_samples,
    resolve_ragas_metric_names,
    select_ragas_live_eval_rows,
)
from creative_coding_assistant.eval.ragas_runner import (
    DefaultRagasEvaluationBackend,
    RagasDependencyError,
    RagasEvaluatorConfig,
    RagasLiveEvalRunResult,
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
        self.assertEqual(resolve_ragas_metric_names(None), DEFAULT_RAGAS_METRICS)
        self.assertEqual(
            resolve_ragas_metric_names(
                ("faithfulness", "context_precision", "faithfulness")
            ),
            ("faithfulness", "context_precision"),
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
                        _sample(sample_id="sample-1", with_context=True)
                        .model_dump_json(),
                        _sample(sample_id="sample-2", with_context=False)
                        .model_dump_json(),
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

        self.assertEqual(result.total_samples, 2)
        self.assertEqual(result.eligible_samples, 1)
        self.assertEqual(result.skipped_samples, 1)
        self.assertEqual(result.metrics, DEFAULT_RAGAS_METRICS)
        self.assertEqual(result.metric_failures, 0)
        self.assertEqual(payload["run_id"], "run-1")
        self.assertEqual(payload["sample_id"], "sample-1")
        self.assertEqual(payload["metrics"]["context_precision"], 0.9)
        self.assertEqual(payload["metric_errors"], {})
        self.assertEqual(payload["source_ids"], ["p5_reference"])
        self.assertNotIn("response", payload)
        self.assertNotIn("retrieved_contexts", payload)
        self.assertNotIn("embedding", json.dumps(payload))

    def test_run_ragas_live_eval_latest_keeps_summary_counts_correct(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "live_sessions.jsonl"
            output_path = Path(temp_dir) / "ragas_results.jsonl"
            input_path.write_text(
                "\n".join(
                    (
                        _sample(sample_id="sample-1", with_context=True)
                        .model_dump_json(),
                        _sample(sample_id="sample-2", with_context=False)
                        .model_dump_json(),
                        _sample(sample_id="sample-3", with_context=True)
                        .model_dump_json(),
                        _sample(sample_id="sample-4", with_context=True)
                        .model_dump_json(),
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
        result = RagasLiveEvalRunResult(
            run_id="run-1",
            input_path=Path("data/eval/live_sessions.jsonl"),
            output_path=Path("data/eval/ragas_results.jsonl"),
            total_samples=3,
            eligible_samples=2,
            skipped_samples=1,
            metrics=DEFAULT_RAGAS_METRICS,
            metric_failures=0,
            result_rows=(),
            skipped=(),
        )

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

    def test_ragas_cli_passes_latest_selection(self) -> None:
        result = RagasLiveEvalRunResult(
            run_id="run-latest",
            input_path=Path("data/eval/live_sessions.jsonl"),
            output_path=Path("data/eval/ragas_results.jsonl"),
            total_samples=4,
            eligible_samples=2,
            skipped_samples=2,
            metrics=DEFAULT_RAGAS_METRICS,
            metric_failures=0,
            result_rows=(),
            skipped=(),
        )

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


def _blocked_ragas_import(name, *args, **kwargs):
    if name == "ragas" or name.startswith("ragas."):
        raise ImportError("missing ragas")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


def _sample(
    *,
    sample_id: str = "sample-1",
    with_context: bool = True,
) -> LiveSessionEvalSample:
    return LiveSessionEvalSample(
        sample_id=sample_id,
        question="How does draw work in p5.js?",
        answer="draw runs repeatedly after setup.",
        conversation_id="conversation-1",
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
        started_at=datetime(2026, 4, 29, 11, 0, tzinfo=UTC),
        completed_at=datetime(2026, 4, 29, 11, 0, 2, tzinfo=UTC),
        recorded_at=datetime(2026, 4, 29, 11, 0, 3, tzinfo=UTC),
    )


if __name__ == "__main__":
    unittest.main()
