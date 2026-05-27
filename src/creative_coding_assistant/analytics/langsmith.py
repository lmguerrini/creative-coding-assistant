"""Optional LangSmith observability boundaries."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from random import random
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.core import Settings

_CLIENT_UNSET = object()


class LangSmithRuntimeConfig(BaseModel):
    """Safe runtime view of optional LangSmith tracing configuration."""

    model_config = ConfigDict(frozen=True)

    requested: bool = False
    enabled: bool = False
    project_name: str = Field(default="creative-coding-assistant", min_length=1)
    environment: str = Field(default="local", min_length=1)
    endpoint: str | None = None
    timeout_ms: int = Field(default=1500, ge=100)
    sampling_rate: float = Field(default=1.0, ge=0, le=1)
    api_key: SecretStr | None = Field(default=None, exclude=True)
    status: str = Field(default="disabled", min_length=1)
    reason: str | None = "tracing_disabled"

    def get_api_key(self) -> str | None:
        if self.api_key is None:
            return None
        secret_value = self.api_key.get_secret_value().strip()
        return secret_value or None

    def summary_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "provider": "langsmith",
            "requested": self.requested,
            "enabled": self.enabled,
            "project_name": self.project_name,
            "environment": self.environment,
            "status": self.status,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.endpoint is not None:
            payload["endpoint"] = self.endpoint
        if self.sampling_rate != 1:
            payload["sampling_rate"] = self.sampling_rate
        return payload


class LangSmithRunMetadata(BaseModel):
    """Serializable trace lineage metadata safe for stream/eval artifacts."""

    model_config = ConfigDict(frozen=True)

    provider: str = "langsmith"
    trace_kind: str = Field(min_length=1)
    run_name: str = Field(min_length=1)
    trace_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    requested: bool = False
    enabled: bool = False
    project_name: str | None = None
    status: str = Field(default="disabled", min_length=1)
    reason: str | None = None
    tags: tuple[str, ...] = Field(default_factory=tuple)
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude_none=True)


class LangSmithObservability:
    """Thin optional adapter around LangSmith tracing primitives."""

    def __init__(self, config: LangSmithRuntimeConfig) -> None:
        self.config = config
        self._client: object = _CLIENT_UNSET

    def assistant_run_context(
        self,
        request: AssistantRequest,
    ) -> LangSmithRunMetadata:
        enabled, status, reason = self._run_state()
        return LangSmithRunMetadata(
            trace_kind="assistant_workflow",
            run_name="assistant.workflow",
            requested=self.config.requested,
            enabled=enabled,
            project_name=self.config.project_name,
            status=status,
            reason=reason,
            tags=("assistant", "workflow", request.mode.value),
            metadata={
                "environment": self.config.environment,
                "mode": request.mode.value,
                "domain": request.domain.value if request.domain else None,
                "domains": [domain.value for domain in request.domains],
                "conversation_id": request.conversation_id,
                "project_id": request.project_id,
                "image_reference_count": len(request.attachments),
            },
        )

    def evaluation_run_context(
        self,
        *,
        eval_run_id: str,
        dataset_id: str,
        metrics: Sequence[str],
        eligible_samples: int,
        skipped_samples: int,
        dry_run: bool,
    ) -> LangSmithRunMetadata:
        enabled, status, reason = self._run_state()
        return LangSmithRunMetadata(
            trace_kind="ragas_evaluation",
            run_name="eval.ragas.live_session",
            requested=self.config.requested,
            enabled=enabled,
            project_name=self.config.project_name,
            status=status,
            reason=reason,
            tags=("evaluation", "ragas", "live-session"),
            metadata={
                "eval_run_id": eval_run_id,
                "dataset_id": dataset_id,
                "metrics": list(metrics),
                "eligible_samples": eligible_samples,
                "skipped_samples": skipped_samples,
                "dry_run": dry_run,
            },
        )

    def event_payload(
        self,
        run: LangSmithRunMetadata,
        *,
        lineage: Mapping[str, object] | None = None,
    ) -> dict[str, object] | None:
        if not run.requested and not run.enabled:
            return None
        payload = run.payload()
        if lineage is not None:
            payload["lineage"] = _json_safe_dict(lineage)
        return payload

    @contextmanager
    def trace(
        self,
        run: LangSmithRunMetadata,
        *,
        run_type: str = "chain",
        inputs: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[None]:
        if not run.enabled:
            yield
            return

        client = self._get_client()
        trace_factory = _get_trace_factory()
        if client is None or trace_factory is None:
            yield
            return

        trace_context = None
        try:
            trace_context = trace_factory(
                name=run.run_name,
                run_type=run_type,
                inputs=dict(inputs or {}),
                project_name=run.project_name,
                client=client,
                tags=list(run.tags),
                metadata={
                    **run.metadata,
                    **dict(metadata or {}),
                    "trace_kind": run.trace_kind,
                    "trace_id": run.trace_id,
                },
            )
            trace_context.__enter__()
        except Exception as exc:
            logger.warning(
                "langsmith_trace_start_failed: {}: {}",
                type(exc).__name__,
                exc,
            )
            yield
            return

        body_exception: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            body_exception = exc
            raise
        finally:
            try:
                trace_context.__exit__(
                    type(body_exception) if body_exception else None,
                    body_exception,
                    body_exception.__traceback__ if body_exception else None,
                )
            except Exception as exc:
                logger.warning(
                    "langsmith_trace_finish_failed: {}: {}",
                    type(exc).__name__,
                    exc,
                )

    def _get_client(self) -> object | None:
        if self._client is not _CLIENT_UNSET:
            return self._client

        try:
            from langsmith import Client

            self._client = Client(
                api_key=self.config.get_api_key(),
                api_url=self.config.endpoint,
                timeout_ms=self.config.timeout_ms,
            )
        except Exception as exc:
            logger.warning(
                "langsmith_client_unavailable: {}: {}",
                type(exc).__name__,
                exc,
            )
            self._client = None
        return self._client

    def _run_state(self) -> tuple[bool, str, str | None]:
        if not self.config.enabled:
            return False, self.config.status, self.config.reason
        if self.config.sampling_rate >= 1 or random() < self.config.sampling_rate:
            return True, self.config.status, self.config.reason
        return False, "disabled", "sampled_out"


def build_langsmith_runtime_config(settings: Settings) -> LangSmithRuntimeConfig:
    """Resolve LangSmith tracing config without creating external resources."""

    requested = settings.langsmith_tracing
    api_key = settings.get_langsmith_api_key()
    if not requested:
        return LangSmithRuntimeConfig(
            requested=False,
            enabled=False,
            project_name=settings.langsmith_project,
            environment=settings.environment,
            endpoint=settings.langsmith_endpoint,
            timeout_ms=settings.langsmith_timeout_ms,
            sampling_rate=settings.langsmith_sampling_rate,
            api_key=settings.langsmith_api_key,
            status="disabled",
            reason="tracing_disabled",
        )

    if api_key is None:
        return LangSmithRuntimeConfig(
            requested=True,
            enabled=False,
            project_name=settings.langsmith_project,
            environment=settings.environment,
            endpoint=settings.langsmith_endpoint,
            timeout_ms=settings.langsmith_timeout_ms,
            sampling_rate=settings.langsmith_sampling_rate,
            api_key=settings.langsmith_api_key,
            status="disabled",
            reason="missing_api_key",
        )

    return LangSmithRuntimeConfig(
        requested=True,
        enabled=True,
        project_name=settings.langsmith_project,
        environment=settings.environment,
        endpoint=settings.langsmith_endpoint,
        timeout_ms=settings.langsmith_timeout_ms,
        sampling_rate=settings.langsmith_sampling_rate,
        api_key=settings.langsmith_api_key,
        status="enabled",
        reason=None,
    )


def build_langsmith_observability(settings: Settings) -> LangSmithObservability:
    return LangSmithObservability(build_langsmith_runtime_config(settings))


def _get_trace_factory() -> object | None:
    try:
        from langsmith.run_helpers import trace
    except Exception as exc:
        logger.warning(
            "langsmith_trace_unavailable: {}: {}",
            type(exc).__name__,
            exc,
        )
        return None
    return trace


def _json_safe_dict(values: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in values.items() if value is not None}
