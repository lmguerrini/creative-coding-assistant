from pathlib import Path

from scripts.v7_quality_gates import (
    ROADMAP_ITEMS,
    build_quality_dashboard,
    find_backend_log_errors,
    find_docs_mermaid_errors,
    validate_quality_dashboard,
)


def test_v7_4_quality_dashboard_covers_every_roadmap_item():
    dashboard = build_quality_dashboard()

    assert dashboard["roadmap_coverage_complete"] is True
    assert dashboard["roadmap_item_count"] == 23
    assert dashboard["quality_gate_count"] == len(ROADMAP_ITEMS)
    assert validate_quality_dashboard() == []


def test_v7_4_quality_dashboard_exposes_performance_and_coverage_gates():
    dashboard = build_quality_dashboard()

    assert dashboard["performance_budgets"]["nextjs_localhost_ready_ms"] > 0
    assert dashboard["performance_budgets"]["workspace_local_storage_bytes"] == 200_000
    assert "frontend_e2e_quality" in dashboard["coverage_dashboard"]
    assert "runtime_pack_quality" in dashboard["coverage_dashboard"]


def test_backend_log_gate_flags_blocking_errors(tmp_path: Path):
    clean_log = tmp_path / "clean.log"
    clean_log.write_text(
        "INFO server ready\nWARNING retrying harmless probe\n", encoding="utf-8"
    )
    error_log = tmp_path / "error.log"
    error_log.write_text("INFO server ready\nERROR failed request\n", encoding="utf-8")

    assert find_backend_log_errors([clean_log]) == []
    assert find_backend_log_errors([error_log]) == [
        f"{error_log}:2: ERROR failed request"
    ]


def test_backend_log_gate_allows_controlled_keyboard_interrupt_shutdown(
    tmp_path: Path,
):
    shutdown_log = tmp_path / "shutdown.log"
    shutdown_log.write_text(
        "\n".join(
            [
                "Traceback (most recent call last):",
                '  File "socketserver.py", line 235, in serve_forever',
                "KeyboardInterrupt",
                "Creative Coding Assistant backend bridge listening on 127.0.0.1:8014",
            ]
        ),
        encoding="utf-8",
    )

    assert find_backend_log_errors([shutdown_log]) == []


def test_backend_log_gate_rejects_non_shutdown_tracebacks_with_interrupts(
    tmp_path: Path,
):
    error_log = tmp_path / "error.log"
    error_log.write_text(
        "\n".join(
            [
                "Traceback (most recent call last):",
                '  File "socketserver.py", line 478, in server_bind',
                "PermissionError: [Errno 1] Operation not permitted",
                "KeyboardInterrupt",
            ]
        ),
        encoding="utf-8",
    )

    assert find_backend_log_errors([error_log]) == [
        f"{error_log}:1: Traceback (most recent call last):"
    ]


def test_docs_mermaid_lint_accepts_valid_diagrams_and_rejects_bad_fences(
    tmp_path: Path,
):
    valid = tmp_path / "valid.md"
    valid.write_text(
        "```mermaid\nflowchart TD\n  A[Start] --> B[Done]\n```\n",
        encoding="utf-8",
    )
    invalid = tmp_path / "invalid.md"
    invalid.write_text("```mermaid\nA --> B\n```\n", encoding="utf-8")

    assert find_docs_mermaid_errors([valid]) == []
    assert find_docs_mermaid_errors([invalid]) == [
        f"{invalid}:1: Mermaid fence must start with a diagram directive"
    ]
