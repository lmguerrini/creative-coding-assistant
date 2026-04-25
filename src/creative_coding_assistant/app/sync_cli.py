"""CLI entrypoint for official KB sync."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from loguru import logger

from creative_coding_assistant.app.sync import sync_official_sources
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_sync_parser()
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(settings.log_level)

    try:
        result = sync_official_sources(
            source_ids=args.source_ids,
            settings=settings,
        )
    except (RuntimeError, ValueError) as exc:
        logger.error(str(exc))
        return 2
    except Exception:
        logger.exception("Official KB sync failed.")
        return 1

    logger.info(
        "Synced {} source(s), {} chunk(s), {} record(s)",
        len(result.source_ids),
        result.total_chunks,
        result.total_records,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
