"""CLI entrypoint for official KB sync."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from loguru import logger

from creative_coding_assistant.app.sync import SyncFailureMode
from creative_coding_assistant.app.sync_service import build_official_kb_sync_service
from creative_coding_assistant.core import load_settings
from creative_coding_assistant.core.logging import configure_logging


def build_sync_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync approved official KB sources into Chroma.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        dest="source_ids",
        help="Approved official source ID to sync. Repeat to sync multiple sources.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sync all approved official sources.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue syncing remaining approved sources after one source fails.",
    )
    parser.add_argument(
        "--summary-format",
        choices=("log", "json"),
        default="log",
        help="How to emit the final sync summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_sync_parser()
    args = parser.parse_args(argv)
    if args.all and args.source_ids:
        parser.error("Use either --all or --source-id, not both.")

    settings = load_settings()
    configure_logging(settings.log_level)

    try:
        service = build_official_kb_sync_service(
            settings=settings,
            failure_mode=(
                SyncFailureMode.CONTINUE
                if args.continue_on_error
                else SyncFailureMode.FAIL_FAST
            ),
        )
        result = (
            service.sync_all_sources()
            if args.all or not args.source_ids
            else service.sync_selected_sources(args.source_ids)
        )
    except (RuntimeError, ValueError) as exc:
        logger.error(str(exc))
        return 2
    except Exception:
        logger.exception("Official KB sync failed.")
        return 1

    if args.summary_format == "json":
        print(result.summary_json())
    else:
        logger.info(
            "Synced {} source(s), {} success, {} failed, {} chunk(s), {} record(s)",
            len(result.source_ids),
            result.succeeded_count,
            result.failed_count,
            result.total_chunks,
            result.total_records,
        )
    return 1 if result.failed_count else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
