"""V7.4 quality, E2E, and CI gate helpers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class QualityGate:
    roadmap_item: str
    evidence: str
    command: str
    ci_required: bool = True


ROADMAP_ITEMS = (
    "Playwright E2E Smoke Suite",
    "Localhost Regression Test",
    "Browser Console Error Gate",
    "Backend Log Error Gate",
    "CI Smoke Test Pipeline",
    "Mermaid/Docs Lint",
    "Performance Regression CI Gate",
    "Release Candidate Checklist Automation",
    "Integration Test Expansion",
    "Behavioral Test Suite",
    "Streaming Failure Tests",
    "Provider Fallback Tests",
    "Adaptive Policy Integration Tests",
    "Complete User Journey Tests",
    "Workspace Persistence Tests",
    "Creative Session Tests",
    "Long Conversation Tests",
    "Memory Stress Tests",
    "Visual Regression Tests",
    "Prompt Regression Tests",
    "KB Regression Tests",
    "Performance Dashboard",
    "Test Coverage Dashboard",
)

QUALITY_GATES = (
    QualityGate(
        "Playwright E2E Smoke Suite",
        "clients/nextjs/e2e/workstation-smoke.spec.js",
        "npm run test:e2e:smoke",
    ),
    QualityGate(
        "Localhost Regression Test",
        "Playwright request/page smoke checks GET / and the mounted workstation.",
        "npm run test:e2e -- workstation-smoke.spec.js",
    ),
    QualityGate(
        "Browser Console Error Gate",
        "installConsoleGate fails on console errors, page errors, and failed requests.",
        "npm run test:e2e",
    ),
    QualityGate(
        "Backend Log Error Gate",
        (
            "scripts/v7_quality_gates.py backend-log scans backend logs for "
            "blocking errors."
        ),
        "python scripts/v7_quality_gates.py backend-log <log-path>",
    ),
    QualityGate(
        "CI Smoke Test Pipeline",
        ".github/workflows/ci.yml runs backend, frontend, and Playwright smoke gates.",
        "gh workflow run ci.yml",
    ),
    QualityGate(
        "Mermaid/Docs Lint",
        "scripts/v7_quality_gates.py docs-mermaid validates Mermaid files and fences.",
        "python scripts/v7_quality_gates.py docs-mermaid",
    ),
    QualityGate(
        "Performance Regression CI Gate",
        "dashboard performance budgets are validated before release.",
        "python scripts/v7_quality_gates.py dashboard",
    ),
    QualityGate(
        "Release Candidate Checklist Automation",
        "dashboard release checklist enumerates required V7.4 gates.",
        "python scripts/v7_quality_gates.py dashboard",
    ),
    QualityGate(
        "Integration Test Expansion",
        (
            "Playwright integrates page, mocked stream API, workspace "
            "persistence, and preview."
        ),
        "npm run test:e2e",
    ),
    QualityGate(
        "Behavioral Test Suite",
        "E2E assertions cover visible behavior instead of implementation-only units.",
        "npm run test:e2e",
    ),
    QualityGate(
        "Streaming Failure Tests",
        "workstation-resilience.spec.js covers failed stream recovery.",
        "npm run test:e2e -- workstation-resilience.spec.js",
    ),
    QualityGate(
        "Provider Fallback Tests",
        "workstation-resilience.spec.js covers provider-unavailable fallback.",
        "npm run test:e2e -- workstation-resilience.spec.js",
    ),
    QualityGate(
        "Adaptive Policy Integration Tests",
        "E2E flow keeps fallback recoverable and avoids failure-node regression.",
        "npm run test:e2e -- workstation-resilience.spec.js",
    ),
    QualityGate(
        "Complete User Journey Tests",
        "workstation-smoke.spec.js covers prompt to preview/artifacts/code/retrieval.",
        "npm run test:e2e -- workstation-smoke.spec.js",
    ),
    QualityGate(
        "Workspace Persistence Tests",
        "workstation-smoke.spec.js verifies persisted theme and density after reload.",
        "npm run test:e2e -- workstation-smoke.spec.js",
    ),
    QualityGate(
        "Creative Session Tests",
        (
            "workstation-smoke.spec.js runs a creative prompt through mocked "
            "stream events."
        ),
        "npm run test:e2e -- workstation-smoke.spec.js",
    ),
    QualityGate(
        "Long Conversation Tests",
        "workstation-resilience.spec.js submits repeated creative prompts.",
        "npm run test:e2e -- workstation-resilience.spec.js",
    ),
    QualityGate(
        "Memory Stress Tests",
        "workstation-resilience.spec.js asserts bounded localStorage size.",
        "npm run test:e2e -- workstation-resilience.spec.js",
    ),
    QualityGate(
        "Visual Regression Tests",
        "workstation-smoke.spec.js validates stable viewport geometry for key regions.",
        "npm run test:e2e -- workstation-smoke.spec.js",
    ),
    QualityGate(
        "Prompt Regression Tests",
        "Playwright submits real composer prompts through the stream contract.",
        "npm run test:e2e",
    ),
    QualityGate(
        "KB Regression Tests",
        "workstation-smoke.spec.js verifies mocked retrieval source visibility.",
        "npm run test:e2e -- workstation-smoke.spec.js",
    ),
    QualityGate(
        "Performance Dashboard",
        (
            "scripts/v7_quality_gates.py dashboard emits deterministic "
            "performance budgets."
        ),
        "python scripts/v7_quality_gates.py dashboard",
    ),
    QualityGate(
        "Test Coverage Dashboard",
        (
            "scripts/v7_quality_gates.py dashboard emits backend/frontend/E2E "
            "coverage gates."
        ),
        "python scripts/v7_quality_gates.py dashboard",
    ),
)

PERFORMANCE_BUDGETS = {
    "nextjs_localhost_ready_ms": 120_000,
    "e2e_default_timeout_ms": 45_000,
    "e2e_expect_timeout_ms": 8_000,
    "workspace_local_storage_bytes": 200_000,
    "ci_playwright_retries": 2,
}

COVERAGE_DASHBOARD = {
    "backend_static_quality": {
        "commands": [
            "git diff --check",
            "python -m compileall src tests scripts",
            "pytest",
        ],
        "required": True,
    },
    "frontend_unit_quality": {
        "commands": ["npm run typecheck", "npm run test"],
        "required": True,
    },
    "frontend_e2e_quality": {
        "commands": ["npm run test:e2e"],
        "required": True,
    },
    "product_documentation_quality": {
        "commands": [
            "python scripts/v7_quality_gates.py docs-mermaid",
            "python scripts/v7_quality_gates.py dashboard",
        ],
        "required": True,
    },
}

MERMAID_STARTERS = (
    "flowchart",
    "graph",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "erDiagram",
    "journey",
    "gantt",
    "pie",
    "mindmap",
    "timeline",
    "gitGraph",
)

BLOCKING_LOG_PATTERNS = (
    re.compile(r"\bCRITICAL\b", re.IGNORECASE),
    re.compile(r"\bERROR\b", re.IGNORECASE),
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"\bUnhandled exception\b", re.IGNORECASE),
    re.compile(r"\bException\b", re.IGNORECASE),
)

LOG_ALLOWLIST_PATTERNS = (
    re.compile(r"Vite CJS Node API deprecation", re.IGNORECASE),
    re.compile(r"DeprecationWarning", re.IGNORECASE),
)

def build_quality_dashboard() -> dict[str, object]:
    missing_items = sorted(
        set(ROADMAP_ITEMS) - {gate.roadmap_item for gate in QUALITY_GATES}
    )
    duplicate_items = sorted(
        item
        for item in ROADMAP_ITEMS
        if [gate.roadmap_item for gate in QUALITY_GATES].count(item) > 1
    )

    return {
        "version": "v7.4",
        "roadmap_item_count": len(ROADMAP_ITEMS),
        "quality_gate_count": len(QUALITY_GATES),
        "roadmap_coverage_complete": not missing_items and not duplicate_items,
        "missing_roadmap_items": missing_items,
        "duplicate_roadmap_items": duplicate_items,
        "gates": [asdict(gate) for gate in QUALITY_GATES],
        "performance_budgets": PERFORMANCE_BUDGETS,
        "coverage_dashboard": COVERAGE_DASHBOARD,
        "release_candidate_checklist": [
            gate.roadmap_item for gate in QUALITY_GATES if gate.ci_required
        ],
    }


def validate_quality_dashboard() -> list[str]:
    dashboard = build_quality_dashboard()
    errors: list[str] = []
    if not dashboard["roadmap_coverage_complete"]:
        errors.append(
            "roadmap coverage incomplete: "
            f"missing={dashboard['missing_roadmap_items']} "
            f"duplicates={dashboard['duplicate_roadmap_items']}"
        )
    for budget_name, budget_value in PERFORMANCE_BUDGETS.items():
        if budget_value <= 0:
            errors.append(f"performance budget must be positive: {budget_name}")
    for surface, config in COVERAGE_DASHBOARD.items():
        commands = config.get("commands") if isinstance(config, dict) else None
        if not commands:
            errors.append(f"coverage dashboard surface has no commands: {surface}")
    return errors


def find_backend_log_errors(
    paths: Iterable[Path], *, allow_missing: bool = False
) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if not path.exists():
            if not allow_missing:
                errors.append(f"missing log file: {path}")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        controlled_shutdown = any("KeyboardInterrupt" in line for line in lines)
        for line_number, line in enumerate(lines, 1):
            if any(pattern.search(line) for pattern in LOG_ALLOWLIST_PATTERNS):
                continue
            if (
                controlled_shutdown
                and "Traceback (most recent call last)" in line
                and is_controlled_keyboard_interrupt_traceback(lines, line_number - 1)
            ):
                continue
            if any(pattern.search(line) for pattern in BLOCKING_LOG_PATTERNS):
                errors.append(f"{path}:{line_number}: {line}")
    return errors


def is_controlled_keyboard_interrupt_traceback(
    lines: list[str], traceback_index: int
) -> bool:
    for line in lines[traceback_index + 1 : traceback_index + 80]:
        stripped = line.strip()
        if not stripped:
            continue
        if line.startswith("Traceback (most recent call last)"):
            return False
        if stripped == "KeyboardInterrupt":
            return True
        if not line.startswith((" ", "\t")):
            return False
    return False


def find_docs_mermaid_errors(paths: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for path in expand_doc_paths(paths):
        if path.suffix == ".mmd":
            errors.extend(validate_mermaid_file(path))
        elif path.suffix.lower() in {".md", ".mdx"}:
            errors.extend(validate_mermaid_fences(path))
    return errors


def expand_doc_paths(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.suffix.lower() in {".md", ".mdx", ".mmd"}
            )
        elif path.exists():
            expanded.append(path)
    return sorted(set(expanded))


def validate_mermaid_file(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return [f"{path}: Mermaid file is empty"]
    first_line = content.splitlines()[0].strip()
    if not first_line.startswith(MERMAID_STARTERS):
        return [f"{path}: Mermaid file must start with a Mermaid diagram directive"]
    return []


def validate_mermaid_fences(path: Path) -> list[str]:
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_mermaid = False
    first_content: str | None = None
    fence_start = 0

    for index, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("```mermaid"):
            if in_mermaid:
                errors.append(f"{path}:{index}: nested Mermaid fence")
            in_mermaid = True
            first_content = None
            fence_start = index
            continue
        if in_mermaid and stripped == "```":
            if first_content is None:
                errors.append(f"{path}:{fence_start}: empty Mermaid fence")
            elif not first_content.startswith(MERMAID_STARTERS):
                errors.append(
                    f"{path}:{fence_start}: Mermaid fence must start with a "
                    "diagram directive"
                )
            in_mermaid = False
            continue
        if in_mermaid and stripped and first_content is None:
            first_content = stripped

    if in_mermaid:
        errors.append(f"{path}:{fence_start}: unclosed Mermaid fence")

    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dashboard = subparsers.add_parser(
        "dashboard", help="Validate and print V7.4 dashboard"
    )
    dashboard.add_argument("--output", type=Path)

    backend_log = subparsers.add_parser(
        "backend-log", help="Scan backend logs for errors"
    )
    backend_log.add_argument("paths", nargs="+", type=Path)
    backend_log.add_argument("--allow-missing", action="store_true")

    docs_mermaid = subparsers.add_parser("docs-mermaid", help="Lint Mermaid docs")
    docs_mermaid.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("README.md"), Path("architecture"), Path("docs")],
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.command == "dashboard":
        errors = validate_quality_dashboard()
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        payload = json.dumps(build_quality_dashboard(), indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(f"{payload}\n", encoding="utf-8")
        else:
            print(payload)
        return 0

    if args.command == "backend-log":
        errors = find_backend_log_errors(args.paths, allow_missing=args.allow_missing)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        return 0

    if args.command == "docs-mermaid":
        errors = find_docs_mermaid_errors(args.paths)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
