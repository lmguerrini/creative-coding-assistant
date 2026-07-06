"""Planning state and stream payload helpers."""

from __future__ import annotations

from typing import Any

from creative_coding_assistant.orchestration.runtime.nodes.planning_contracts import (
    PLANNING_EVENT_PAYLOAD_FIELDS,
    PLANNING_RUNTIME_UPDATE_FIELDS,
    PlanningRuntimeArtifacts,
)


def _planning_runtime_updates(
    artifacts: PlanningRuntimeArtifacts,
) -> dict[str, Any]:
    return {field: getattr(artifacts, field) for field in PLANNING_RUNTIME_UPDATE_FIELDS}


def _planning_event_payload(
    artifacts: PlanningRuntimeArtifacts,
) -> dict[str, Any]:
    return {
        field: getattr(artifacts, field).model_dump(mode="json")
        for field in PLANNING_EVENT_PAYLOAD_FIELDS
    }
