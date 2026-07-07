from pathlib import Path

WORKFLOWS_DIR = Path(".github/workflows")
BACKEND_RELEASE_WORKFLOW = WORKFLOWS_DIR / "backend-release-verification.yml"
FAST_CI_WORKFLOW = WORKFLOWS_DIR / "ci.yml"


def test_backend_release_verification_is_version_freeze_only() -> None:
    workflow = BACKEND_RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert '"version-review/**"' in workflow
    assert '"version-freeze/**"' in workflow

    forbidden_triggers = (
        "schedule:",
        "release:",
        "tags:",
        '"v*"',
        "main",
        '"release/**"',
    )
    for trigger in forbidden_triggers:
        assert trigger not in workflow


def test_capability_tags_are_fast_ci_only() -> None:
    fast_ci = FAST_CI_WORKFLOW.read_text(encoding="utf-8")
    backend_release = BACKEND_RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "tags:" in fast_ci
    assert '"v*"' in fast_ci
    assert "tags:" not in backend_release
    assert '"v*"' not in backend_release
