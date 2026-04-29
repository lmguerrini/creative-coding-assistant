#!/usr/bin/env python3
"""Local entrypoint for evaluating recorded live sessions with RAGAs."""


if __name__ == "__main__":
    from creative_coding_assistant.eval.ragas_cli import main

    raise SystemExit(main())
