import unittest
from datetime import UTC, datetime

from creative_coding_assistant.contracts import (
    AssistantMode,
    CreativeCodingDomain,
)
from creative_coding_assistant.memory import ConversationRole
from creative_coding_assistant.orchestration import (
    InMemoryExecutionCache,
    MemoryContextRequest,
    MemoryContextResponse,
    RecentConversationTurn,
    RetrievedKnowledgeChunk,
    RouteName,
    analyze_assistant_execution_graph,
    analyze_creative_complexity,
    analyze_workflow_complexity,
    analyze_workflow_cost,
    compress_prompt_text,
    compress_retrieval_chunks,
    forecast_execution_cost,
    plan_context_budget,
    plan_context_reuse,
    plan_execution_path_optimization,
    plan_exploration_budget,
    plan_workflow_pruning,
    route_context_sources,
    select_execution_strategy,
    summarize_memory_context,
)
from creative_coding_assistant.rag.sources import OfficialSourceType

EXPECTED_V5_1_SURFACES = (
    "execution_graph_analysis",
    "workflow_cost_analysis",
    "workflow_complexity_analysis",
    "creative_complexity_analysis",
    "context_budget_plan",
    "exploration_budget_plan",
    "context_routing_plan",
    "prompt_compression_result",
    "retrieval_compression_result",
    "memory_summarization_result",
    "execution_cache_lookup",
    "context_reuse_plan",
    "workflow_pruning_plan",
    "execution_cost_forecast",
    "execution_path_optimization_plan",
    "execution_strategy_selection",
)

RUNTIME_FLAGS_THAT_MUST_REMAIN_DISABLED = (
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_order_mutation_implemented",
    "graph_compilation_implemented",
    "workflow_execution_implemented",
    "node_handler_invocation_implemented",
    "execution_path_selection_implemented",
    "execution_strategy_application_implemented",
    "budget_enforcement_implemented",
    "cost_based_routing_implemented",
)

FORBIDDEN_RUNTIME_TERMS = (
    "route_provider(",
    "select_provider(",
    "control_workflow(",
    "trigger_retry(",
    "write_storage(",
    "write_persistent_storage(",
    "modify_output(",
    "mutate_graph(",
    "compile_graph(",
    "execute_workflow(",
    "invoke_node_handler(",
    "select_execution_path(",
    "apply_strategy(",
    "enforce_budget(",
)


class V51ExecutionOptimizationArchitectureConsistencyTests(unittest.TestCase):
    def test_v5_1_surfaces_preserve_architecture_boundaries(self) -> None:
        surfaces = _v5_1_surface_artifacts()

        self.assertEqual(
            tuple(surface_id for surface_id, _ in surfaces), EXPECTED_V5_1_SURFACES
        )
        for surface_id, artifact in surfaces:
            self.assertTrue(
                artifact.serialization_version.endswith(".v1"),
                surface_id,
            )
            self.assertTrue(getattr(artifact, "role", surface_id), surface_id)
            self.assertTrue(getattr(artifact, "authority_boundary", ""), surface_id)
            self.assertTrue(
                getattr(artifact, "blocked_runtime_behaviors", ()), surface_id
            )

            for flag_name in RUNTIME_FLAGS_THAT_MUST_REMAIN_DISABLED:
                if hasattr(artifact, flag_name):
                    self.assertFalse(
                        getattr(artifact, flag_name), (surface_id, flag_name)
                    )

            combined_text = _combined_surface_text(artifact)
            for forbidden_term in FORBIDDEN_RUNTIME_TERMS:
                self.assertNotIn(forbidden_term, combined_text, surface_id)

    def test_v5_1_strategy_selection_remains_advisory_not_runtime_control(self) -> None:
        strategy_selection = dict(_v5_1_surface_artifacts())[
            "execution_strategy_selection"
        ]

        self.assertEqual(
            strategy_selection.selected_strategy_id,
            "execution_strategy::cost_guarded_pruning",
        )
        self.assertTrue(strategy_selection.execution_strategy_selection_implemented)
        self.assertFalse(strategy_selection.execution_strategy_application_implemented)
        self.assertFalse(strategy_selection.execution_path_selection_implemented)
        self.assertFalse(strategy_selection.workflow_control_implemented)
        self.assertTrue(strategy_selection.selection_only)


def _v5_1_surface_artifacts() -> tuple[tuple[str, object], ...]:
    graph = analyze_assistant_execution_graph()
    workflow_cost = analyze_workflow_cost(execution_graph=graph)
    workflow_complexity = analyze_workflow_complexity(
        execution_graph=graph,
        cost_analysis=workflow_cost,
    )
    creative_complexity = analyze_creative_complexity()
    context_budget = plan_context_budget(
        creative_complexity=creative_complexity,
        workflow_cost=workflow_cost,
        user_query="Generate a calm p5.js sketch.",
    )
    exploration_budget = plan_exploration_budget(
        creative_complexity=creative_complexity,
        workflow_cost=workflow_cost,
        context_budget=context_budget,
    )
    context_routes = route_context_sources(context_budget=context_budget)
    prompt_compression = compress_prompt_text(
        "Create a calm p5.js sketch with blue light and slow particles.",
    )
    retrieval_compression = compress_retrieval_chunks((_retrieval_chunk(),))
    memory_summary = summarize_memory_context(_memory_context())
    cache_lookup = InMemoryExecutionCache().get(
        namespace="workflow",
        components={"node": "planning"},
        now=_now(),
    )
    context_reuse = plan_context_reuse(
        previous_context_budget=context_budget,
        current_context_budget=context_budget,
    )
    pruning = plan_workflow_pruning(
        execution_graph=graph,
        cost_analysis=workflow_cost,
        complexity_analysis=workflow_complexity,
    )
    cost_forecast = forecast_execution_cost(
        cost_analysis=workflow_cost,
        pruning_plan=pruning,
    )
    path_optimization = plan_execution_path_optimization(
        execution_graph=graph,
        cost_analysis=workflow_cost,
        cost_forecast=cost_forecast,
        pruning_plan=pruning,
    )
    strategy_selection = select_execution_strategy(
        path_optimization=path_optimization,
        cost_forecast=cost_forecast,
        pruning_plan=pruning,
    )

    return (
        ("execution_graph_analysis", graph),
        ("workflow_cost_analysis", workflow_cost),
        ("workflow_complexity_analysis", workflow_complexity),
        ("creative_complexity_analysis", creative_complexity),
        ("context_budget_plan", context_budget),
        ("exploration_budget_plan", exploration_budget),
        ("context_routing_plan", context_routes),
        ("prompt_compression_result", prompt_compression),
        ("retrieval_compression_result", retrieval_compression),
        ("memory_summarization_result", memory_summary),
        ("execution_cache_lookup", cache_lookup),
        ("context_reuse_plan", context_reuse),
        ("workflow_pruning_plan", pruning),
        ("execution_cost_forecast", cost_forecast),
        ("execution_path_optimization_plan", path_optimization),
        ("execution_strategy_selection", strategy_selection),
    )


def _combined_surface_text(artifact: object) -> str:
    fields: list[str] = [
        str(getattr(artifact, "authority_boundary", "")),
        *tuple(getattr(artifact, "blocked_runtime_behaviors", ())),
        *tuple(getattr(artifact, "advisory_actions", ())),
    ]
    for collection_name in (
        "candidates",
        "strategies",
        "scenarios",
        "allocations",
        "decisions",
        "sections",
        "chunks",
        "segments",
        "factors",
        "components",
        "nodes",
        "edges",
    ):
        for item in getattr(artifact, collection_name, ()):
            fields.extend(tuple(getattr(item, "blocked_runtime_behaviors", ())))
            fields.extend(tuple(getattr(item, "advisory_actions", ())))
    return " ".join(fields)


def _retrieval_chunk() -> RetrievedKnowledgeChunk:
    return RetrievedKnowledgeChunk(
        source_id="p5-reference",
        domain=CreativeCodingDomain.P5_JS,
        source_type=OfficialSourceType.GUIDE,
        publisher="p5.js",
        registry_title="p5.js Reference",
        document_title="Reference",
        source_url="https://p5js.org/reference/",
        chunk_index=1,
        excerpt="Use createCanvas in setup and draw slow particles in draw.",
        score=0.8,
        rank=1,
    )


def _memory_context() -> MemoryContextResponse:
    return MemoryContextResponse(
        request=MemoryContextRequest(route=RouteName.GENERATE),
        recent_turns=(
            RecentConversationTurn(
                turn_index=1,
                role=ConversationRole.USER,
                content="Keep the project calm with blue particles.",
                created_at=_now(),
                mode=AssistantMode.GENERATE,
            ),
        ),
    )


def _now() -> datetime:
    return datetime(2026, 6, 28, 12, 0, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
