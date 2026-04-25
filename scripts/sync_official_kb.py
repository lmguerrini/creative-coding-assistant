#!/usr/bin/env python3
"""Local entrypoint for syncing approved official KB sources."""


if __name__ == "__main__":
    from creative_coding_assistant.app.sync_cli import main

    raise SystemExit(main())
