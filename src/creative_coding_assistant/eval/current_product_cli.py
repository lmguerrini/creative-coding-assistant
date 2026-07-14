"""CLI for reproducible current-product retrieval/generation/RAGAS runs."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from loguru import logger

from creative_coding_assistant.core import load_settings
from creative_coding_assistant.core.logging import configure_logging
from creative_coding_assistant.eval.current_product import (
    CurrentProductEvaluationBlockedError,
    CurrentProductEvaluationResult,
    CurrentProductEvaluationRunner,
    CurrentProductRunOptions,
    build_safe_current_product_evidence,
    write_safe_current_product_evidence,
)
from creative_coding_assistant.eval.ragas_runner import RagasDependencyError

DEFAULT_CURRENT_PRODUCT_EVIDENCE_PATH = Path("demo/evaluation/current_product_ragas_evidence.json")


def build_current_product_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the immutable seven-case pack through the current retriever, "
            "prompt renderer, generation provider, and reference-aware RAGAS metrics."
        ),
        epilog=(
            "Without --allow-provider-calls this is a preparation-only dry run. "
            "Write baseline and iteration diagnostics only under ignored data/eval/. "
            "Use --publish-canonical only for the final retained, complete seven-case run."
        ),
    )
    parser.add_argument(
        "--scope",
        choices=("full", "rag", "cases"),
        default="rag",
        help="Select all canonical RAG cases or an explicit diagnostic subset.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Canonical case ID; repeat for --scope cases.",
    )
    parser.add_argument(
        "--allow-provider-calls",
        action="store_true",
        help=("Authorize current generation, query embedding, RAGAS evaluator, and evaluator embedding calls."),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force preparation only, even when provider calls are authorized.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_CURRENT_PRODUCT_EVIDENCE_PATH,
        help="Canonical safe evidence output for a promotable full run.",
    )
    parser.add_argument(
        "--publish-canonical",
        action="store_true",
        help=("Publish the final promotable full run to --output-path. Omit this for baseline and iteration runs."),
    )
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        help=(
            "Optional redacted prepared/subset/partial output. Must resolve under the gitignored data/eval/ directory."
        ),
    )
    parser.add_argument(
        "--private-diagnostic-output",
        type=Path,
        help=(
            "Optional exact answer/context/evaluator diagnostic. Must resolve under "
            "the gitignored data/eval/ directory and must never be committed."
        ),
    )
    parser.add_argument(
        "--run-id",
        help="Optional stable run ID for controlled reproduction.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_current_product_eval_parser().parse_args(argv)
    if args.allow_provider_calls:
        _disable_secondary_observability()
    settings = load_settings()
    configure_logging(settings.log_level)
    try:
        diagnostic_output = _resolve_ignored_diagnostic_path(
            args.diagnostic_output,
            kind="Redacted diagnostics",
        )
        private_output = _resolve_private_diagnostic_path(args.private_diagnostic_output)
    except ValueError as exc:
        logger.error(str(exc))
        return 4
    dry_run = args.dry_run or not args.allow_provider_calls
    try:
        result = CurrentProductEvaluationRunner(settings=settings).run(
            run_id=args.run_id or uuid4().hex,
            options=CurrentProductRunOptions(
                scope=args.scope,
                case_ids=tuple(args.case_id),
                allow_provider_calls=args.allow_provider_calls,
                dry_run=dry_run,
            ),
        )
    except CurrentProductEvaluationBlockedError as exc:
        logger.error(str(exc))
        return 3
    except RagasDependencyError as exc:
        logger.error(str(exc))
        return 2
    except Exception:
        logger.exception("Current-product evaluation failed without publishing a score.")
        return 1

    safe_payload = build_safe_current_product_evidence(result)
    if private_output is not None:
        _write_private_diagnostic_payload(private_output, result)
        logger.info("Private exact diagnostic written to {}", private_output)
    if args.publish_canonical:
        try:
            write_safe_current_product_evidence(result, args.output_path)
        except ValueError as exc:
            logger.error(str(exc))
            return 5
        logger.info("Canonical current-product evidence written to {}", args.output_path)
    elif diagnostic_output is not None:
        _write_diagnostic_payload(diagnostic_output, safe_payload)
        logger.info("Unscored diagnostic evidence written to {}", diagnostic_output)
    print(json.dumps(safe_payload, indent=2, sort_keys=True))
    return 0


def _disable_secondary_observability() -> None:
    """Keep authorized evaluation traffic limited to declared model providers."""

    disabled = {
        "ANONYMIZED_TELEMETRY": "False",
        "DO_NOT_TRACK": "1",
        "RAGAS_DO_NOT_TRACK": "true",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGSMITH_TRACING": "false",
        "CCA_LANGSMITH_TRACING": "false",
        "OTEL_SDK_DISABLED": "true",
    }
    for name, value in disabled.items():
        os.environ[name] = value
    for name in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"):
        os.environ.pop(name, None)


def _write_diagnostic_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_private_diagnostic_path(path: Path | None) -> Path | None:
    return _resolve_ignored_diagnostic_path(path, kind="Private diagnostics")


def _resolve_ignored_diagnostic_path(
    path: Path | None,
    *,
    kind: str,
) -> Path | None:
    if path is None:
        return None
    repository_root = Path(__file__).resolve().parents[3]
    private_root = (repository_root / "data/eval").resolve()
    resolved = (repository_root / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(private_root) or resolved == private_root:
        raise ValueError(f"{kind} must use a file below the gitignored data/eval/ directory.")
    if resolved == (repository_root / DEFAULT_CURRENT_PRODUCT_EVIDENCE_PATH).resolve():
        raise ValueError("Private diagnostics cannot use the canonical public evidence path.")
    return resolved


def _write_private_diagnostic_payload(
    path: Path,
    result: CurrentProductEvaluationResult,
) -> None:
    payload = {
        "schemaVersion": "current-product-private-diagnostic.v1",
        "privacyClass": "private_local_gitignored_do_not_commit",
        "result": result.model_dump(mode="json", by_alias=True),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
