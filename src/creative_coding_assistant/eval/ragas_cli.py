"""CLI entrypoint for manual RAGAs evaluation of live session records."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from creative_coding_assistant.core import load_settings
from creative_coding_assistant.core.logging import configure_logging
from creative_coding_assistant.eval.ragas_models import SUPPORTED_RAGAS_METRICS
from creative_coding_assistant.eval.ragas_runner import (
    RagasDependencyError,
    RagasEvaluatorConfig,
    run_ragas_live_eval,
)


def build_ragas_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate recorded live chat samples with RAGAs.",
        epilog=(
            "This command is local/manual only. Running RAGAs may call evaluator "
            "LLM APIs and can incur provider cost."
        ),
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        help="Live-session JSONL input path. Defaults to CCA_EVAL_DATA_PATH.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "JSONL result output path. Defaults to "
            "CCA_EVAL_RAGAS_RESULTS_PATH."
        ),
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Evaluate only the first N eligible samples.",
    )
    parser.add_argument(
        "--latest",
        type=_positive_int,
        help=(
            "Evaluate the newest N eligible samples. "
            "Takes precedence over --limit."
        ),
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=SUPPORTED_RAGAS_METRICS,
        help=(
            "Metric to run. Repeat to select multiple metrics. "
            "Defaults to the safe smoke metric set."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_ragas_eval_parser()
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(settings.log_level)
    input_path = args.input_path or settings.eval_data_path
    output_path = args.output_path or settings.eval_ragas_results_path

    try:
        result = run_ragas_live_eval(
            input_path=input_path,
            output_path=output_path,
            metric_names=tuple(args.metric) if args.metric else None,
            limit=args.limit,
            latest=args.latest,
            evaluator_config=RagasEvaluatorConfig(
                model=settings.eval_ragas_model,
                embedding_model=settings.openai_embedding_model,
                openai_api_key=settings.openai_api_key,
                timeout_seconds=settings.eval_ragas_timeout_seconds,
                max_retries=settings.eval_ragas_max_retries,
                max_workers=settings.eval_ragas_max_workers,
            ),
        )
    except RagasDependencyError as exc:
        logger.error(str(exc))
        return 2
    except Exception:
        logger.exception("Live-session RAGAs evaluation failed.")
        return 1

    logger.info(
        "RAGAs live eval complete: {} total, {} eligible, {} skipped, "
        "{} result row(s), {} metric failure(s), metrics={}, output={}",
        result.total_samples,
        result.eligible_samples,
        result.skipped_samples,
        len(result.result_rows),
        result.metric_failures,
        ", ".join(result.metrics),
        result.output_path,
    )
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
