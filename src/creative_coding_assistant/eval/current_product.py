"""Current-product retrieval, generation, and RAGAS evaluation pipeline."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativityProfile,
    GenerationControls,
    StreamEventType,
)
from creative_coding_assistant.core import Settings
from creative_coding_assistant.eval.live_session import (
    LiveSessionEvalSample,
    LiveSessionRetrievedContext,
)
from creative_coding_assistant.eval.ragas_models import RagasLiveEvalRow
from creative_coding_assistant.eval.ragas_runner import (
    DefaultRagasEvaluationBackend,
    RagasEvaluationBackend,
    RagasEvaluatorConfig,
    RagasProviderBoundaryError,
)
from creative_coding_assistant.eval.recorder import build_live_session_sample
from creative_coding_assistant.eval.retrieval_demo_pack import (
    CURRENT_PRODUCT_RETRIEVAL_BENCHMARK_VERSION,
    CURRENT_PRODUCT_RETRIEVAL_METRICS,
    RetrievalDemoScenario,
    build_capstone_retrieval_demo_pack,
    fingerprint_capstone_retrieval_demo_pack,
)
from creative_coding_assistant.llm.factory import build_generation_provider
from creative_coding_assistant.llm.generation import (
    GenerationInput,
    GenerationProvider,
    GenerationStreamEvent,
)
from creative_coding_assistant.orchestration.runtime.workflow_review import (
    MAX_WORKFLOW_REFINEMENT_COUNT,
)
from creative_coding_assistant.rag import get_official_source
from creative_coding_assistant.rag.sources import OFFICIAL_HOSTS_BY_DOMAIN
from creative_coding_assistant.vectorstore import (
    ChromaCollection,
    create_chroma_client,
)

CurrentProductStatus = Literal["completed", "prepared", "blocked", "failed"]
ProgressCallback = Callable[["CurrentProductEvaluationProgress"], None]

_RETRIEVAL_PIPELINE_MODULES = (
    "creative_coding_assistant.app.bootstrap",
    "creative_coding_assistant.orchestration.runtime.routing",
    "creative_coding_assistant.orchestration.runtime.retrieval",
    "creative_coding_assistant.rag.embeddings.openai",
    "creative_coding_assistant.rag.retrieval.embedder",
    "creative_coding_assistant.rag.retrieval.factory",
    "creative_coding_assistant.rag.retrieval.filters",
    "creative_coding_assistant.rag.retrieval.models",
    "creative_coding_assistant.rag.retrieval.openai_embedder",
    "creative_coding_assistant.rag.retrieval.search",
    "creative_coding_assistant.rag.retrieval.postprocess",
    "creative_coding_assistant.rag.retrieval.domain_intent",
    "creative_coding_assistant.rag.sources",
    "creative_coding_assistant.vectorstore.client",
    "creative_coding_assistant.vectorstore.collections",
    "creative_coding_assistant.vectorstore.metadata",
    "creative_coding_assistant.vectorstore.repository",
)
_PROMPT_PIPELINE_MODULES = (
    "creative_coding_assistant.orchestration.runtime.context",
    "creative_coding_assistant.orchestration.runtime.nodes.refinement",
    "creative_coding_assistant.orchestration.runtime.nodes.review",
    "creative_coding_assistant.orchestration.runtime.prompt_inputs",
    "creative_coding_assistant.orchestration.runtime.prompt_templates",
    "creative_coding_assistant.orchestration.runtime.workflow_review",
)
_GENERATION_PIPELINE_MODULES = (
    "creative_coding_assistant.orchestration.runtime.service",
    "creative_coding_assistant.orchestration.runtime.generation",
    "creative_coding_assistant.llm.openai_adapter",
)

_CURRENT_PRODUCT_PRIVACY_CLASS = (
    "public_official_contexts_with_authored_references"
)
_CURRENT_PRODUCT_METRIC_CONTRACT = "ragas-current-product-reference.v2"
_CURRENT_PRODUCT_EVIDENCE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "demo/evaluation/current_product_ragas_evidence.schema.json"
)


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class _CamelModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        alias_generator=_to_camel,
    )


class CurrentProductRunOptions(_CamelModel):
    """Bounded selection and provider authorization for one current run."""

    scope: str = Field(default="full", min_length=1)
    case_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=100)
    allow_provider_calls: bool = False
    dry_run: bool = True


class CurrentProductEvaluationProgress(_CamelModel):
    """Stable progress payload shared by the runner and async HTTP registry."""

    phase: str = Field(min_length=1)
    lane: str = Field(min_length=1)
    current_case_id: str | None = None
    current_case_label: str = ""
    completed_cases: int = Field(ge=0)
    total_cases: int = Field(ge=0)
    remaining_cases: int = Field(ge=0)
    percent: float | None = Field(default=None, ge=0, le=100)
    execution_state: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class CurrentProductCaseResult(_CamelModel):
    """Raw current answer and public official contexts for one evaluated case."""

    case_id: str
    label: str
    question: str
    reference_answer: str
    reference_context: tuple[str, ...]
    reference_source_ids: tuple[str, ...]
    generated_answer: str
    expected_domains: tuple[str, ...]
    expected_source_ids: tuple[str, ...]
    retrieved_contexts: tuple[LiveSessionRetrievedContext, ...]
    source_ids: tuple[str, ...]
    domains: tuple[str, ...]
    metrics: dict[str, float | None]
    metric_errors: dict[str, str]
    provider: str | None = None
    model: str | None = None
    prompt_fingerprint: str
    generation_fingerprint: str


class CurrentProductGenerationConfiguration(_CamelModel):
    """Non-secret generation settings bound into the pipeline fingerprint."""

    provider: str
    model: str
    max_output_tokens: int = Field(ge=1)
    timeout_seconds: int = Field(ge=1)
    creativity_profile: str


class CurrentProductEvaluationResult(_CamelModel):
    """One terminal current-product benchmark result with complete provenance."""

    benchmark_mode: Literal["current_product"] = "current_product"
    score_origin: Literal["current_product", "unscored"]
    scope: str
    run_id: str
    status: CurrentProductStatus
    benchmark_version: str
    dataset_id: str
    dataset_version: str
    privacy_class: str = _CURRENT_PRODUCT_PRIVACY_CLASS
    dataset_fingerprint: str
    retrieval_fingerprint: str | None = None
    prompt_fingerprint: str | None = None
    generation_fingerprint: str | None = None
    generation_configuration: CurrentProductGenerationConfiguration | None = None
    output_fingerprint: str | None = None
    selection_fingerprint: str | None = None
    kb_fingerprint: str | None = None
    selected_case_ids: tuple[str, ...]
    evaluated_at: datetime
    metrics: tuple[str, ...]
    metric_scores: dict[str, float]
    retrieval_score: float | None = None
    result_rows: int = Field(ge=0)
    total_samples: int = Field(ge=0)
    eligible_samples: int = Field(ge=0)
    skipped_samples: int = Field(ge=0)
    metric_failures: int = Field(ge=0)
    provider: str | None = None
    model: str | None = None
    generation_model: str | None = None
    evaluator: str | None = None
    evaluator_model: str | None = None
    embedding_model: str | None = None
    ragas_version: str | None = None
    metric_contract: str = _CURRENT_PRODUCT_METRIC_CONTRACT
    duration_ms: int = Field(ge=0)
    detail: str
    case_results: tuple[CurrentProductCaseResult, ...]


class CurrentProductEvaluationBlockedError(RuntimeError):
    """Raised when a requested live run cannot cross its provider boundary."""


class CurrentProductEvaluationProviderBoundaryError(
    CurrentProductEvaluationBlockedError
):
    """Raised when an authorized provider is systemically unavailable."""


@dataclass(frozen=True)
class CurrentProductCaseExecution:
    sample: LiveSessionEvalSample
    prompt_fingerprint: str


class CurrentProductCaseExecutor(Protocol):
    def execute(
        self,
        scenario: RetrievalDemoScenario,
    ) -> CurrentProductCaseExecution:
        """Run one case through the current assistant service."""


class _CollectingLiveSessionRecorder:
    def __init__(self) -> None:
        self.samples: list[LiveSessionEvalSample] = []

    def record(self, **kwargs: Any) -> LiveSessionEvalSample | None:
        sample = build_live_session_sample(**kwargs)
        if sample is not None:
            self.samples.append(sample)
        return sample


class _CapturingGenerationProvider:
    def __init__(
        self,
        delegate: GenerationProvider,
        *,
        allowed_public_chunks: frozenset[tuple[str, str, str]] | None = None,
    ) -> None:
        self._delegate = delegate
        self._allowed_public_chunks = allowed_public_chunks
        self.inputs: list[GenerationInput] = []

    def stream(self, request: GenerationInput) -> Iterable[GenerationStreamEvent]:
        prompt_input = request.request.rendered_prompt.request.prompt_input
        retrieval_input = prompt_input.retrieval_input
        if retrieval_input is None or not retrieval_input.chunks:
            raise ValueError("Current-product generation requires validated official retrieval context.")
        _validate_prompt_official_chunks(
            retrieval_input.chunks,
            allowed_public_chunks=self._allowed_public_chunks,
        )
        _validate_benchmark_generation_request(request)
        self.inputs.append(request)
        return self._delegate.stream(request)


class _NoopMemoryRecorder:
    def record_turns(self, **_: Any) -> None:
        return None


class AssistantServiceCaseExecutor:
    """Execute benchmark cases through retrieval, Jinja prompts, and generation."""

    def __init__(self, settings: Settings) -> None:
        from creative_coding_assistant.app.bootstrap import build_assistant_service

        self._recorder = _CollectingLiveSessionRecorder()
        self._provider = _CapturingGenerationProvider(
            build_generation_provider(settings),
            allowed_public_chunks=_validated_public_kb_chunk_allowlist(settings),
        )
        self._service = build_assistant_service(
            settings=settings,
            generation_provider=self._provider,
            eval_recorder=self._recorder,
            memory_recorder=_NoopMemoryRecorder(),
        )

    def execute(
        self,
        scenario: RetrievalDemoScenario,
    ) -> CurrentProductCaseExecution:
        sample_count = len(self._recorder.samples)
        input_count = len(self._provider.inputs)
        response = self._service.respond(
            AssistantRequest(
                query=scenario.query,
                domains=scenario.domains,
                mode=AssistantMode.EXPLAIN,
                generation_controls=GenerationControls(profile=CreativityProfile.CONTROLLED),
            )
        )
        if any(event.event_type is StreamEventType.ERROR for event in response.events):
            raise CurrentProductEvaluationProviderBoundaryError(
                "Current generation provider was unavailable; the fallback answer "
                "is not scoreable."
            )
        if len(self._recorder.samples) != sample_count + 1:
            raise RuntimeError("Current assistant did not record an evaluation sample.")
        new_inputs = self._provider.inputs[input_count:]
        max_generation_inputs = 1 + MAX_WORKFLOW_REFINEMENT_COUNT
        if not 1 <= len(new_inputs) <= max_generation_inputs:
            raise RuntimeError(
                "Current assistant must prepare one initial prompt and at most "
                f"{MAX_WORKFLOW_REFINEMENT_COUNT} bounded refinement prompts."
            )
        return CurrentProductCaseExecution(
            sample=self._recorder.samples[-1],
            prompt_fingerprint=_fingerprint([item.model_dump(mode="json") for item in new_inputs]),
        )


class CurrentProductEvaluationRunner:
    """Run the immutable seven-case pack against the current product stack."""

    def __init__(
        self,
        *,
        settings: Settings,
        case_executor_factory: Callable[[Settings], CurrentProductCaseExecutor] | None = None,
        evaluator_factory: Callable[[Settings], RagasEvaluationBackend] | None = None,
        kb_fingerprint_factory: Callable[[Settings], str] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings = settings.model_copy(
            update={
                "langsmith_tracing": False,
                "langsmith_api_key": None,
            }
        )
        self._case_executor_factory = case_executor_factory or AssistantServiceCaseExecutor
        self._evaluator_factory = evaluator_factory or _default_evaluator
        self._kb_fingerprint_factory = kb_fingerprint_factory or fingerprint_current_knowledge_base
        self._now = now or (lambda: datetime.now(UTC))

    def run(
        self,
        *,
        run_id: str,
        options: CurrentProductRunOptions,
        progress: ProgressCallback | None = None,
    ) -> CurrentProductEvaluationResult:
        started_at = perf_counter()
        pack = build_capstone_retrieval_demo_pack()
        scenarios = _select_scenarios(pack.scenarios, options)
        selected_ids = tuple(item.demo_id for item in scenarios)
        dataset_fingerprint = fingerprint_capstone_retrieval_demo_pack(pack)
        metrics = CURRENT_PRODUCT_RETRIEVAL_METRICS
        self._publish(
            progress,
            phase="preflight",
            lane="current_product_rag",
            completed=0,
            total=len(scenarios),
            execution_state="prepared" if options.dry_run else "provider_authorized",
            detail="Validated the immutable current-product benchmark selection.",
        )

        if not scenarios:
            return self._empty_result(
                run_id=run_id,
                options=options,
                dataset_fingerprint=dataset_fingerprint,
                selected_ids=selected_ids,
                metrics=metrics,
                status="completed",
                detail="RAGAS was not requested because this scope selected no canonical RAG cases.",
                started_at=started_at,
            )
        if options.dry_run:
            return self._empty_result(
                run_id=run_id,
                options=options,
                dataset_fingerprint=dataset_fingerprint,
                selected_ids=selected_ids,
                metrics=metrics,
                status="prepared",
                detail=(
                    "Current-product cases and authored references were prepared; no "
                    "retrieval, generation, embedding, or evaluator provider was called."
                ),
                started_at=started_at,
            )
        if not options.allow_provider_calls:
            raise CurrentProductEvaluationBlockedError(
                "Live current-product evaluation requires explicit provider-call authorization."
            )
        if not self._settings.has_openai_api_key:
            raise CurrentProductEvaluationBlockedError(
                "Evaluator, embedding, and generation provider credentials are unavailable."
            )

        kb_fingerprint = self._kb_fingerprint_factory(self._settings)
        executor = self._case_executor_factory(self._settings)
        executions: list[tuple[RetrievalDemoScenario, CurrentProductCaseExecution]] = []
        for index, scenario in enumerate(scenarios):
            self._publish(
                progress,
                phase="generation",
                lane="current_product_rag",
                current_case=scenario,
                completed=index,
                total=len(scenarios),
                execution_state="running",
                detail="Running current retrieval, prompt rendering, and generation.",
                percent=(index / len(scenarios)) * 70,
            )
            execution = executor.execute(scenario)
            validated_contexts = _validate_public_official_contexts(execution.sample.retrieved_contexts)
            executions.append(
                (
                    scenario,
                    CurrentProductCaseExecution(
                        sample=execution.sample.model_copy(update={"retrieved_contexts": validated_contexts}),
                        prompt_fingerprint=execution.prompt_fingerprint,
                    ),
                )
            )

        rows = tuple(_ragas_row(scenario, execution) for scenario, execution in executions)
        self._publish(
            progress,
            phase="evaluation",
            lane="ragas_reference_metrics",
            completed=len(scenarios),
            total=len(scenarios),
            execution_state="running",
            detail="Scoring current answers and contexts against authored references.",
            percent=85,
        )
        try:
            scores = self._evaluator_factory(self._settings).evaluate(rows, metrics)
        except RagasProviderBoundaryError as exc:
            raise CurrentProductEvaluationProviderBoundaryError(
                "The evaluator provider was unavailable across the current-product "
                "metric run; no completed score was produced."
            ) from exc
        if len(scores) != len(rows):
            raise ValueError("RAGAS backend returned a mismatched score row count.")
        generation_configuration = CurrentProductGenerationConfiguration(
            provider=str(self._settings.default_generation_provider),
            model=self._settings.openai_model,
            max_output_tokens=self._settings.openai_max_output_tokens,
            timeout_seconds=self._settings.openai_timeout_seconds,
            creativity_profile=CreativityProfile.CONTROLLED.value,
        )
        generation_fingerprint = _generation_pipeline_fingerprint(
            generation_configuration,
        )
        case_results = tuple(
            _case_result(
                scenario,
                execution,
                score,
                metrics,
                generation_fingerprint,
            )
            for (scenario, execution), score in zip(executions, scores, strict=True)
        )
        metric_failures = sum(len(item.metric_errors) for item in case_results)
        metric_scores = _aggregate_metric_scores(case_results, metrics)
        complete_score_contract = (
            selected_ids == tuple(item.demo_id for item in pack.scenarios)
            and len(case_results) == len(scenarios)
            and metric_failures == 0
            and all(_valid_score(item.metrics.get(metric)) for item in case_results for metric in metrics)
        )
        retrieval_score = (
            sum(metric_scores[name] for name in metrics) / len(metrics)
            if complete_score_contract and all(name in metric_scores for name in metrics)
            else None
        )
        retrieval_fingerprint = _retrieval_pipeline_fingerprint(
            self._settings.openai_embedding_model,
        )
        prompt_fingerprint = _prompt_pipeline_fingerprint(
            tuple(execution.prompt_fingerprint for _, execution in executions),
        )
        output_fingerprint = _output_fingerprint(case_results)
        selection_fingerprint = _selection_fingerprint(case_results)
        provider, generation_model = _generation_identity(case_results)
        self._publish(
            progress,
            phase="completed",
            lane="current_product_rag",
            completed=len(scenarios),
            total=len(scenarios),
            execution_state="completed",
            detail="Current-product RAGAS evaluation completed.",
            percent=100,
        )
        return CurrentProductEvaluationResult(
            score_origin=("current_product" if complete_score_contract else "unscored"),
            scope=options.scope,
            run_id=run_id,
            status="completed",
            benchmark_version=CURRENT_PRODUCT_RETRIEVAL_BENCHMARK_VERSION,
            dataset_id=pack.pack_id,
            dataset_version=CURRENT_PRODUCT_RETRIEVAL_BENCHMARK_VERSION,
            dataset_fingerprint=dataset_fingerprint,
            retrieval_fingerprint=retrieval_fingerprint,
            prompt_fingerprint=prompt_fingerprint,
            generation_fingerprint=generation_fingerprint,
            generation_configuration=generation_configuration,
            output_fingerprint=output_fingerprint,
            selection_fingerprint=selection_fingerprint,
            kb_fingerprint=kb_fingerprint,
            selected_case_ids=selected_ids,
            evaluated_at=self._now(),
            metrics=metrics,
            metric_scores=metric_scores,
            retrieval_score=retrieval_score,
            result_rows=len(case_results),
            total_samples=len(scenarios),
            eligible_samples=len(case_results),
            skipped_samples=len(scenarios) - len(case_results),
            metric_failures=metric_failures,
            provider=provider,
            model=generation_model,
            generation_model=generation_model,
            evaluator=f"RAGAS OpenAI evaluator ({self._settings.eval_ragas_model})",
            evaluator_model=self._settings.eval_ragas_model,
            embedding_model=self._settings.openai_embedding_model,
            ragas_version=_package_version("ragas"),
            duration_ms=round((perf_counter() - started_at) * 1000),
            detail=(
                "Current assistant retrieval, rendered prompts, generated answers, and "
                "reference-aware RAGAS metrics were evaluated on public official contexts."
                if complete_score_contract
                else (
                    "Evaluation completed, but at least one applicable case metric "
                    "was null; no current score was promoted."
                )
            ),
            case_results=case_results,
        )

    def _empty_result(
        self,
        *,
        run_id: str,
        options: CurrentProductRunOptions,
        dataset_fingerprint: str,
        selected_ids: tuple[str, ...],
        metrics: tuple[str, ...],
        status: Literal["completed", "prepared"],
        detail: str,
        started_at: float,
    ) -> CurrentProductEvaluationResult:
        return CurrentProductEvaluationResult(
            score_origin="unscored",
            scope=options.scope,
            run_id=run_id,
            status=status,
            benchmark_version=CURRENT_PRODUCT_RETRIEVAL_BENCHMARK_VERSION,
            dataset_id=build_capstone_retrieval_demo_pack().pack_id,
            dataset_version=CURRENT_PRODUCT_RETRIEVAL_BENCHMARK_VERSION,
            dataset_fingerprint=dataset_fingerprint,
            selected_case_ids=selected_ids,
            evaluated_at=self._now(),
            metrics=metrics,
            metric_scores={},
            result_rows=0,
            total_samples=len(selected_ids),
            eligible_samples=0,
            skipped_samples=len(selected_ids),
            metric_failures=0,
            duration_ms=round((perf_counter() - started_at) * 1000),
            detail=detail,
            case_results=(),
        )

    @staticmethod
    def _publish(
        callback: ProgressCallback | None,
        *,
        phase: str,
        lane: str,
        completed: int,
        total: int,
        execution_state: str,
        detail: str,
        current_case: RetrievalDemoScenario | None = None,
        percent: float | None = None,
    ) -> None:
        if callback is None:
            return
        callback(
            CurrentProductEvaluationProgress(
                phase=phase,
                lane=lane,
                current_case_id=current_case.demo_id if current_case else None,
                current_case_label=current_case.title if current_case else "",
                completed_cases=completed,
                total_cases=total,
                remaining_cases=max(total - completed, 0),
                percent=percent,
                execution_state=execution_state,
                detail=detail,
            )
        )


def _select_scenarios(
    scenarios: Sequence[RetrievalDemoScenario],
    options: CurrentProductRunOptions,
) -> tuple[RetrievalDemoScenario, ...]:
    if options.scope in {"full", "rag"}:
        return tuple(scenarios)
    if options.scope == "cases":
        if not options.case_ids:
            raise ValueError("Current-product case scope requires at least one case ID.")
        canonical_ids = {item.demo_id for item in scenarios}
        unknown_ids = tuple(case_id for case_id in options.case_ids if case_id not in canonical_ids)
        if unknown_ids:
            raise ValueError("Unknown current-product case ID(s): " + ", ".join(unknown_ids))
        selected = set(options.case_ids)
        return tuple(item for item in scenarios if item.demo_id in selected)
    return ()


def _ragas_row(
    scenario: RetrievalDemoScenario,
    execution: CurrentProductCaseExecution,
) -> RagasLiveEvalRow:
    sample = execution.sample
    return RagasLiveEvalRow(
        sample_id=scenario.demo_id,
        user_input=scenario.query,
        response=sample.answer,
        retrieved_contexts=tuple(item.excerpt for item in sample.retrieved_contexts),
        ground_truth=scenario.reference_answer,
        source_ids=tuple(dict.fromkeys(item.source_id for item in sample.retrieved_contexts)),
        domains=tuple(dict.fromkeys(item.domain.value for item in sample.retrieved_contexts)),
        retrieval_scores=tuple(item.score for item in sample.retrieved_contexts),
        provider_metadata=sample.provider_metadata,
        observability_metadata=sample.observability_metadata,
        recorded_at=sample.recorded_at,
    )


def _case_result(
    scenario: RetrievalDemoScenario,
    execution: CurrentProductCaseExecution,
    scores: Mapping[str, float | None],
    metrics: Sequence[str],
    generation_pipeline_fingerprint: str,
) -> CurrentProductCaseResult:
    normalized = {name: _score(scores.get(name)) for name in metrics}
    metric_errors = {name: "metric_returned_null" for name, value in normalized.items() if value is None}
    sample = execution.sample
    provider = sample.provider_metadata.provider if sample.provider_metadata else None
    model = sample.provider_metadata.model if sample.provider_metadata else None
    generation_fingerprint = _case_generation_fingerprint(
        case_id=scenario.demo_id,
        prompt_fingerprint=execution.prompt_fingerprint,
        generation_pipeline_fingerprint=generation_pipeline_fingerprint,
    )
    return CurrentProductCaseResult(
        case_id=scenario.demo_id,
        label=scenario.title,
        question=scenario.query,
        reference_answer=scenario.reference_answer,
        reference_context=scenario.reference_context,
        reference_source_ids=scenario.reference_source_ids,
        generated_answer=sample.answer,
        expected_domains=tuple(domain.value for domain in scenario.domains),
        expected_source_ids=scenario.expected_source_ids,
        retrieved_contexts=sample.retrieved_contexts,
        source_ids=tuple(dict.fromkeys(item.source_id for item in sample.retrieved_contexts)),
        domains=tuple(dict.fromkeys(item.domain.value for item in sample.retrieved_contexts)),
        metrics=normalized,
        metric_errors=metric_errors,
        provider=provider,
        model=model,
        prompt_fingerprint=execution.prompt_fingerprint,
        generation_fingerprint=generation_fingerprint,
    )


def _validate_public_official_contexts(
    contexts: Sequence[LiveSessionRetrievedContext],
) -> tuple[LiveSessionRetrievedContext, ...]:
    if not contexts:
        raise ValueError("Current-product RAGAS requires retrieved contexts.")
    validated: list[LiveSessionRetrievedContext] = []
    for context in contexts:
        source = get_official_source(context.source_id)
        if (
            context.domain != source.domain
            or context.publisher != source.publisher
            or context.registry_title != source.title
            or context.source_url != source.url
        ):
            raise ValueError("Retrieved context lineage does not match the official registry.")
        for url in (context.source_url, context.resolved_url):
            if url is None:
                continue
            parsed = urlparse(url)
            if (
                parsed.scheme != "https"
                or parsed.hostname not in OFFICIAL_HOSTS_BY_DOMAIN[source.domain]
                or not any(parsed.path.startswith(prefix) for prefix in source.allowed_path_prefixes)
            ):
                raise ValueError("Retrieved context URL is outside the approved public scope.")
        validated.append(context)
    return tuple(validated)


def _validate_benchmark_generation_request(request: GenerationInput) -> None:
    """Reject any prompt input that is not wholly benchmark-authored or derived."""

    prompt_input = request.request.rendered_prompt.request.prompt_input
    assistant_request = prompt_input.request.assistant_request
    scenario = next(
        (item for item in build_capstone_retrieval_demo_pack().scenarios if item.query == assistant_request.query),
        None,
    )
    if scenario is None:
        raise ValueError("Generation query is outside the public benchmark allowlist.")
    expected = AssistantRequest(
        query=scenario.query,
        domains=scenario.domains,
        mode=AssistantMode.EXPLAIN,
        generation_controls=GenerationControls(profile=CreativityProfile.CONTROLLED),
    )
    if assistant_request != expected:
        raise ValueError("Generation request contains state outside the public benchmark contract.")
    if (
        prompt_input.user_input.query != scenario.query
        or prompt_input.memory_input is not None
        or prompt_input.user_input.is_follow_up
        or prompt_input.user_input.image_references
        or prompt_input.user_input.artifact_refinement is not None
        or prompt_input.user_input.clarification_response is not None
        or request.image_inputs
    ):
        raise ValueError("Generation prompt contains memory, media, or non-benchmark user state.")


def _validate_prompt_official_chunks(
    chunks: Sequence[object],
    *,
    allowed_public_chunks: frozenset[tuple[str, str, str]] | None = None,
) -> None:
    """Enforce public lineage before prompt text crosses the provider boundary."""

    for chunk in chunks:
        source_id = str(chunk.source_id)
        source = get_official_source(source_id)
        if (
            chunk.domain != source.domain
            or str(chunk.publisher) != source.publisher
            or str(chunk.registry_title) != source.title
            or str(chunk.source_url) != source.url
        ):
            raise ValueError("Prompt retrieval lineage does not match the official registry.")
        parsed = urlparse(str(chunk.source_url))
        if (
            parsed.scheme != "https"
            or parsed.hostname not in OFFICIAL_HOSTS_BY_DOMAIN[source.domain]
            or not any(parsed.path.startswith(prefix) for prefix in source.allowed_path_prefixes)
        ):
            raise ValueError("Prompt retrieval URL is outside the approved public scope.")
        excerpt = str(chunk.excerpt)
        if _contains_sensitive_text(excerpt):
            raise ValueError("Prompt retrieval text contains a sensitive-data marker.")
        if (
            allowed_public_chunks is not None
            and (source_id, str(chunk.source_url), excerpt) not in allowed_public_chunks
        ):
            raise ValueError("Prompt retrieval text is absent from the validated public KB snapshot.")


_SENSITIVE_TEXT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b",
        r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
        r"\bAIza[0-9A-Za-z_-]{30,}\b",
        r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b",
    )
)


def _contains_sensitive_text(value: str) -> bool:
    return any(pattern.search(value) for pattern in _SENSITIVE_TEXT_PATTERNS)


def _validated_public_kb_chunk_allowlist(
    settings: Settings,
) -> frozenset[tuple[str, str, str]]:
    """Load and verify the exact public KB text permitted in model prompts."""

    path = Path(settings.chroma_persist_dir)
    if not path.exists():
        raise CurrentProductEvaluationBlockedError(
            "The configured current-product Knowledge Base index is unavailable."
        )
    client = create_chroma_client(settings=settings)
    try:
        collection = client.get_collection(name=ChromaCollection.KB_OFFICIAL_DOCS.value)
    except Exception as exc:
        raise CurrentProductEvaluationBlockedError(
            "The configured official-documents collection is unavailable."
        ) from exc
    result = collection.get(include=["documents", "metadatas"])
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    allowed: set[tuple[str, str, str]] = set()
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else None
        if not isinstance(document, str) or not isinstance(metadata, Mapping):
            raise CurrentProductEvaluationBlockedError("The public Knowledge Base contains an incomplete chunk record.")
        source_id = str(metadata.get("source_id", ""))
        source = get_official_source(source_id)
        source_url = str(metadata.get("source_url", ""))
        if (
            metadata.get("collection") != ChromaCollection.KB_OFFICIAL_DOCS.value
            or metadata.get("record_kind") != "official_doc_chunk"
            or str(metadata.get("domain")) != source.domain.value
            or str(metadata.get("publisher")) != source.publisher
            or str(metadata.get("registry_title")) != source.title
            or source_url != source.url
            or str(metadata.get("chunk_hash")) != hashlib.sha256(document.encode("utf-8")).hexdigest()
            or int(metadata.get("char_count", -1)) != len(document)
        ):
            raise CurrentProductEvaluationBlockedError(
                "The public Knowledge Base chunk lineage or content hash is invalid."
            )
        parsed = urlparse(source_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in OFFICIAL_HOSTS_BY_DOMAIN[source.domain]
            or not any(parsed.path.startswith(prefix) for prefix in source.allowed_path_prefixes)
            or _contains_sensitive_text(document)
        ):
            raise CurrentProductEvaluationBlockedError(
                "The public Knowledge Base failed its outbound privacy allowlist."
            )
        allowed.add((source_id, source_url, document))
    if not allowed:
        raise CurrentProductEvaluationBlockedError("The validated public Knowledge Base chunk allowlist is empty.")
    return frozenset(allowed)


def _default_evaluator(settings: Settings) -> RagasEvaluationBackend:
    return DefaultRagasEvaluationBackend(
        RagasEvaluatorConfig(
            model=settings.eval_ragas_model,
            embedding_model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
            timeout_seconds=settings.eval_ragas_timeout_seconds,
            max_retries=settings.eval_ragas_max_retries,
            max_workers=settings.eval_ragas_max_workers,
        )
    )


def fingerprint_current_knowledge_base(settings: Settings) -> str:
    """Fingerprint only non-text KB lineage metadata from the configured index."""

    path = Path(settings.chroma_persist_dir)
    if not path.exists():
        raise CurrentProductEvaluationBlockedError(
            "The configured current-product Knowledge Base index is unavailable."
        )
    client = create_chroma_client(settings=settings)
    try:
        collection = client.get_collection(name=ChromaCollection.KB_OFFICIAL_DOCS.value)
    except Exception as exc:
        raise CurrentProductEvaluationBlockedError(
            "The configured official-documents collection is unavailable."
        ) from exc
    result = collection.get(include=["metadatas"])
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    rows = []
    for index, record_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        rows.append(
            {
                "recordId": record_id,
                "sourceId": metadata.get("source_id"),
                "domain": metadata.get("domain"),
                "documentTitle": metadata.get("document_title"),
                "chunkIndex": metadata.get("chunk_index"),
                "contentHash": metadata.get("content_hash"),
                "chunkHash": metadata.get("chunk_hash"),
            }
        )
    return _fingerprint(sorted(rows, key=lambda row: str(row["recordId"])))


def build_safe_current_product_evidence(
    result: CurrentProductEvaluationResult,
) -> dict[str, object]:
    """Drop raw answers and excerpts while retaining publishable score provenance."""

    return {
        "schemaVersion": "current-product-ragas-evidence.v1",
        "benchmarkMode": result.benchmark_mode,
        "scoreOrigin": result.score_origin,
        "scope": result.scope,
        "runId": result.run_id,
        "status": result.status,
        "benchmarkVersion": result.benchmark_version,
        "datasetId": result.dataset_id,
        "datasetVersion": result.dataset_version,
        "selectedCaseIds": list(result.selected_case_ids),
        "datasetFingerprint": result.dataset_fingerprint,
        "retrievalFingerprint": result.retrieval_fingerprint,
        "promptFingerprint": result.prompt_fingerprint,
        "generationFingerprint": result.generation_fingerprint,
        "outputFingerprint": result.output_fingerprint,
        "selectionFingerprint": result.selection_fingerprint,
        "kbFingerprint": result.kb_fingerprint,
        "evaluatedAt": result.evaluated_at.isoformat(),
        "timestamp": result.evaluated_at.isoformat(),
        "metrics": list(result.metrics),
        "metricScores": dict(result.metric_scores),
        "retrievalScore": result.retrieval_score,
        "resultRows": result.result_rows,
        "totalSamples": result.total_samples,
        "eligibleSamples": result.eligible_samples,
        "skippedSamples": result.skipped_samples,
        "metricFailures": result.metric_failures,
        "provider": result.provider,
        "model": result.model,
        "generationModel": result.generation_model,
        "evaluator": result.evaluator,
        "evaluatorModel": result.evaluator_model,
        "embeddingModel": result.embedding_model,
        "ragasVersion": result.ragas_version,
        "metricContract": result.metric_contract,
        "durationMs": result.duration_ms,
        "detail": result.detail,
        "privacyClass": result.privacy_class,
        "caseResults": [
            {
                "caseId": item.case_id,
                "sourceIds": list(item.source_ids),
                "domains": list(item.domains),
                "metrics": dict(item.metrics),
                "metricErrors": dict(item.metric_errors),
                "promptFingerprint": item.prompt_fingerprint,
                "generationFingerprint": item.generation_fingerprint,
            }
            for item in result.case_results
        ],
    }


def write_safe_current_product_evidence(
    result: CurrentProductEvaluationResult,
    path: Path,
) -> None:
    """Persist the public-safe evidence projection, never raw answer/context text."""

    issues = _canonical_evidence_issues(result)
    if issues:
        raise ValueError(
            "Canonical current-product evidence requires a completed, fully scored "
            "seven-case run with complete provenance: " + ", ".join(issues)
        )

    payload = build_safe_current_product_evidence(result)
    _validate_safe_current_product_evidence_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_safe_current_product_evidence_payload(
    payload: Mapping[str, object],
) -> None:
    """Validate against the committed public schema without a runtime dependency."""

    try:
        raw_schema = json.loads(
            _CURRENT_PRODUCT_EVIDENCE_SCHEMA_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "The committed current-product evidence schema is unavailable or invalid."
        ) from exc
    if not isinstance(raw_schema, Mapping):
        raise ValueError("The committed current-product evidence schema is not an object.")
    try:
        _validate_json_schema_value(
            payload,
            raw_schema,
            root_schema=raw_schema,
            path="$",
        )
    except ValueError as exc:
        raise ValueError(
            "Public current-product evidence does not satisfy the committed JSON "
            f"schema: {exc}"
        ) from exc


def _validate_json_schema_value(
    value: object,
    schema: Mapping[str, object],
    *,
    root_schema: Mapping[str, object],
    path: str,
) -> None:
    reference = schema.get("$ref")
    if isinstance(reference, str):
        resolved = _resolve_local_json_schema_reference(root_schema, reference)
        _validate_json_schema_value(
            value,
            resolved,
            root_schema=root_schema,
            path=path,
        )
        return

    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} does not match its required constant")
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        raise ValueError(f"{path} is outside its allowed values")

    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _json_schema_type_matches(
        value,
        expected_type,
    ):
        raise ValueError(f"{path} must be a JSON {expected_type}")

    if isinstance(value, Mapping):
        required = schema.get("required")
        if isinstance(required, list):
            missing = [name for name in required if name not in value]
            if missing:
                raise ValueError(f"{path} is missing required field {missing[0]}")
        maximum_properties = schema.get("maxProperties")
        if isinstance(maximum_properties, int) and len(value) > maximum_properties:
            raise ValueError(f"{path} has too many properties")
        properties = schema.get("properties")
        declared = properties if isinstance(properties, Mapping) else {}
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(declared)
            if extras:
                raise ValueError(
                    f"{path} contains forbidden field {sorted(extras)[0]}"
                )
        for key, item in value.items():
            child_schema = declared.get(key)
            if isinstance(child_schema, Mapping):
                _validate_json_schema_value(
                    item,
                    child_schema,
                    root_schema=root_schema,
                    path=f"{path}.{key}",
                )

    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        maximum_items = schema.get("maxItems")
        if isinstance(minimum_items, int) and len(value) < minimum_items:
            raise ValueError(f"{path} has too few items")
        if isinstance(maximum_items, int) and len(value) > maximum_items:
            raise ValueError(f"{path} has too many items")
        if schema.get("uniqueItems") is True:
            tokens = [
                json.dumps(item, ensure_ascii=False, sort_keys=True)
                for item in value
            ]
            if len(tokens) != len(set(tokens)):
                raise ValueError(f"{path} must contain unique items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_json_schema_value(
                    item,
                    item_schema,
                    root_schema=root_schema,
                    path=f"{path}[{index}]",
                )

    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            raise ValueError(f"{path} is too short")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            raise ValueError(f"{path} does not match its required pattern")
        if schema.get("format") == "date-time":
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"{path} is not a date-time") from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError(f"{path} must include a timezone offset")

    if _json_schema_type_matches(value, "number"):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and float(value) < minimum:
            raise ValueError(f"{path} is below its minimum")
        if isinstance(maximum, (int, float)) and float(value) > maximum:
            raise ValueError(f"{path} is above its maximum")


def _resolve_local_json_schema_reference(
    root_schema: Mapping[str, object],
    reference: str,
) -> Mapping[str, object]:
    if not reference.startswith("#/"):
        raise ValueError("only local JSON Schema references are supported")
    resolved: object = root_schema
    for raw_token in reference[2:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if not isinstance(resolved, Mapping) or token not in resolved:
            raise ValueError(f"unresolved JSON Schema reference {reference}")
        resolved = resolved[token]
    if not isinstance(resolved, Mapping):
        raise ValueError(f"JSON Schema reference {reference} is not an object")
    return resolved


def _json_schema_type_matches(value: object, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, Mapping)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    raise ValueError(f"unsupported JSON Schema type {expected_type}")


def _canonical_evidence_issues(
    result: CurrentProductEvaluationResult,
) -> tuple[str, ...]:
    issues: list[str] = []
    pack = build_capstone_retrieval_demo_pack()
    expected_case_ids = tuple(item.demo_id for item in pack.scenarios)
    expected_dataset_fingerprint = fingerprint_capstone_retrieval_demo_pack(pack)
    if result.status != "completed" or result.score_origin != "current_product":
        issues.append("promotable_status")
    if (
        not _valid_score(result.retrieval_score)
        or result.metric_failures != 0
        or tuple(result.metrics) != CURRENT_PRODUCT_RETRIEVAL_METRICS
        or set(result.metric_scores) != set(CURRENT_PRODUCT_RETRIEVAL_METRICS)
        or any(
            not _valid_score(result.metric_scores.get(metric))
            for metric in CURRENT_PRODUCT_RETRIEVAL_METRICS
        )
    ):
        issues.append("complete_metrics")
    if (
        tuple(result.selected_case_ids) != expected_case_ids
        or result.result_rows != len(expected_case_ids)
        or result.total_samples != len(expected_case_ids)
        or result.eligible_samples != len(expected_case_ids)
        or result.skipped_samples != 0
        or len(result.case_results) != len(expected_case_ids)
        or tuple(item.case_id for item in result.case_results) != expected_case_ids
    ):
        issues.append("full_seven_case_selection")
    if (
        result.benchmark_version != CURRENT_PRODUCT_RETRIEVAL_BENCHMARK_VERSION
        or result.dataset_version != CURRENT_PRODUCT_RETRIEVAL_BENCHMARK_VERSION
        or result.dataset_id != pack.pack_id
        or result.dataset_fingerprint != expected_dataset_fingerprint
        or result.privacy_class != _CURRENT_PRODUCT_PRIVACY_CLASS
        or result.metric_contract != _CURRENT_PRODUCT_METRIC_CONTRACT
        or result.scope not in {"full", "rag"}
    ):
        issues.append("immutable_dataset_contract")

    if not all(
        _case_matches_immutable_scenario(item, scenario)
        for item, scenario in zip(result.case_results, pack.scenarios, strict=False)
    ):
        issues.append("immutable_case_contract")
    if not all(_case_has_official_lineage(item) for item in result.case_results):
        issues.append("official_source_lineage")

    expected_metric_scores = _aggregate_metric_scores(
        result.case_results,
        result.metrics,
    )
    if not _score_maps_match(
        result.metric_scores,
        expected_metric_scores,
        result.metrics,
    ):
        issues.append("aggregate_metric_arithmetic")
    expected_retrieval_score = (
        sum(expected_metric_scores[metric] for metric in result.metrics)
        / len(result.metrics)
        if result.metrics
        and all(metric in expected_metric_scores for metric in result.metrics)
        else None
    )
    if not _scores_match(result.retrieval_score, expected_retrieval_score):
        issues.append("retrieval_score_arithmetic")

    fingerprint_fields = (
        result.dataset_fingerprint,
        result.retrieval_fingerprint,
        result.prompt_fingerprint,
        result.generation_fingerprint,
        result.output_fingerprint,
        result.selection_fingerprint,
        result.kb_fingerprint,
    )
    if not all(_valid_fingerprint(value) for value in fingerprint_fields):
        issues.append("provenance_fingerprints")
    identity_fields = (
        result.provider,
        result.model,
        result.generation_model,
        result.evaluator,
        result.evaluator_model,
        result.embedding_model,
        result.ragas_version,
    )
    if not all(isinstance(value, str) and value.strip() for value in identity_fields):
        issues.append("provider_evaluator_identity")
    if any(
        not item.source_ids
        or not item.domains
        or set(item.metrics) != set(result.metrics)
        or item.metric_errors
        or not _valid_fingerprint(item.prompt_fingerprint)
        or not _valid_fingerprint(item.generation_fingerprint)
        or any(not _valid_score(item.metrics.get(metric)) for metric in result.metrics)
        for item in result.case_results
    ):
        issues.append("per_case_metric_provenance")

    if not _derived_fingerprints_match(result):
        issues.append("derivable_fingerprint_arithmetic")
    return tuple(issues)


def _case_matches_immutable_scenario(
    case: CurrentProductCaseResult,
    scenario: RetrievalDemoScenario,
) -> bool:
    return (
        case.case_id == scenario.demo_id
        and case.label == scenario.title
        and case.question == scenario.query
        and case.reference_answer == scenario.reference_answer
        and tuple(case.reference_context) == tuple(scenario.reference_context)
        and tuple(case.reference_source_ids) == tuple(scenario.reference_source_ids)
        and tuple(case.expected_domains)
        == tuple(domain.value for domain in scenario.domains)
        and tuple(case.expected_source_ids) == tuple(scenario.expected_source_ids)
    )


def _case_has_official_lineage(case: CurrentProductCaseResult) -> bool:
    try:
        contexts = _validate_public_official_contexts(case.retrieved_contexts)
        sources = tuple(get_official_source(source_id) for source_id in case.source_ids)
    except (KeyError, TypeError, ValueError):
        return False
    derived_source_ids = tuple(
        dict.fromkeys(context.source_id for context in contexts)
    )
    derived_domains = tuple(
        dict.fromkeys(context.domain.value for context in contexts)
    )
    registered_domains = tuple(
        dict.fromkeys(source.domain.value for source in sources)
    )
    return (
        tuple(case.source_ids) == derived_source_ids
        and tuple(case.domains) == derived_domains
        and derived_domains == registered_domains
    )


def _score_maps_match(
    actual: Mapping[str, float],
    expected: Mapping[str, float],
    metrics: Sequence[str],
) -> bool:
    return set(actual) == set(metrics) == set(expected) and all(
        _scores_match(actual.get(metric), expected.get(metric))
        for metric in metrics
    )


def _scores_match(left: object, right: object) -> bool:
    return (
        _valid_score(left)
        and _valid_score(right)
        and math.isclose(
            float(left),
            float(right),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    )


def _derived_fingerprints_match(
    result: CurrentProductEvaluationResult,
) -> bool:
    configuration = result.generation_configuration
    if (
        configuration is None
        or not isinstance(result.embedding_model, str)
        or not result.embedding_model.strip()
        or configuration.provider != result.provider
        or result.model != result.generation_model
    ):
        return False
    if result.retrieval_fingerprint != _retrieval_pipeline_fingerprint(
        result.embedding_model
    ):
        return False
    if result.generation_fingerprint != _generation_pipeline_fingerprint(
        configuration
    ):
        return False
    if result.prompt_fingerprint != _prompt_pipeline_fingerprint(
        tuple(item.prompt_fingerprint for item in result.case_results)
    ):
        return False
    if result.output_fingerprint != _output_fingerprint(result.case_results):
        return False
    if result.selection_fingerprint != _selection_fingerprint(result.case_results):
        return False
    return all(
        item.generation_fingerprint
        == _case_generation_fingerprint(
            case_id=item.case_id,
            prompt_fingerprint=item.prompt_fingerprint,
            generation_pipeline_fingerprint=result.generation_fingerprint,
        )
        for item in result.case_results
    )


def _valid_score(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0 <= float(value) <= 1
    )


def _valid_fingerprint(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _aggregate_metric_scores(
    cases: Sequence[CurrentProductCaseResult],
    metrics: Sequence[str],
) -> dict[str, float]:
    aggregates: dict[str, float] = {}
    for metric in metrics:
        values = [item.metrics[metric] for item in cases if item.metrics[metric] is not None]
        if values:
            aggregates[metric] = sum(value for value in values if value is not None) / len(values)
    return aggregates


def _generation_identity(
    cases: Sequence[CurrentProductCaseResult],
) -> tuple[str | None, str | None]:
    providers = tuple(dict.fromkeys(item.provider for item in cases if item.provider))
    models = tuple(dict.fromkeys(item.model for item in cases if item.model))
    return (
        providers[0] if len(providers) == 1 else None,
        models[0] if len(models) == 1 else None,
    )


def _score(value: object) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) and 0 <= score <= 1 else None


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _pipeline_fingerprint(
    module_names: Sequence[str],
    config: Mapping[str, object],
) -> str:
    return _fingerprint({"sourceHash": _module_source_hash(module_names), "config": dict(config)})


def _retrieval_pipeline_fingerprint(embedding_model: str) -> str:
    return _pipeline_fingerprint(
        _RETRIEVAL_PIPELINE_MODULES,
        {
            "collection": ChromaCollection.KB_OFFICIAL_DOCS.value,
            "embeddingModel": embedding_model,
            "retrievalLimit": 5,
        },
    )


def _generation_pipeline_fingerprint(
    configuration: CurrentProductGenerationConfiguration,
) -> str:
    return _pipeline_fingerprint(
        _GENERATION_PIPELINE_MODULES,
        configuration.model_dump(mode="json", by_alias=True),
    )


def _prompt_pipeline_fingerprint(
    case_prompt_fingerprints: Sequence[str],
) -> str:
    return _fingerprint(
        {
            "sourceHash": _module_source_hash(_PROMPT_PIPELINE_MODULES),
            "casePromptFingerprints": list(case_prompt_fingerprints),
        }
    )


def _case_generation_fingerprint(
    *,
    case_id: str,
    prompt_fingerprint: str,
    generation_pipeline_fingerprint: str,
) -> str:
    return _fingerprint(
        {
            "caseId": case_id,
            "promptFingerprint": prompt_fingerprint,
            "generationPipelineFingerprint": generation_pipeline_fingerprint,
        }
    )


def _output_fingerprint(
    cases: Sequence[CurrentProductCaseResult],
) -> str:
    return _fingerprint(
        [
            {
                "caseId": item.case_id,
                "answer": item.generated_answer,
                "contextHashes": [
                    _fingerprint(context.excerpt)
                    for context in item.retrieved_contexts
                ],
                "provider": item.provider,
                "model": item.model,
            }
            for item in cases
        ]
    )


def _selection_fingerprint(
    cases: Sequence[CurrentProductCaseResult],
) -> str:
    return _fingerprint(
        [
            {
                "caseId": item.case_id,
                "sourceIds": list(item.source_ids),
                "contexts": [
                    {
                        "sourceId": context.source_id,
                        "chunkIndex": context.chunk_index,
                        "score": context.score,
                        "excerptHash": _fingerprint(context.excerpt),
                    }
                    for context in item.retrieved_contexts
                ],
            }
            for item in cases
        ]
    )


def _module_source_hash(module_names: Sequence[str]) -> str:
    return _fingerprint([inspect.getsource(import_module(module_name)) for module_name in module_names])


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None
