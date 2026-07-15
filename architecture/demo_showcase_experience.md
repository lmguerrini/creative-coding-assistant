# Demo Asset Architecture

This document describes the passive metadata and committed assets behind Demo
Mode and the static local launcher. It does not define the executable request
graph; see the [System Architecture Overview](system_architecture_overview.md)
and [End-to-End Product Workflow](end_to_end_product_workflow.md).

## Scope

The demo layer catalogs scenario prompts, expected behavior, fallback states,
evidence references, static creative-code artifacts, and sanitized evaluation
fixtures. Demo Mode loads those records into the normal workstation composer.

## Runtime boundary

Demo metadata does not:

- execute providers, retrieval, or evaluator scoring;
- render or repair previews;
- route models or change the LangGraph workflow;
- mutate prompts, artifacts, memory, or persistent storage;
- execute external creative tools; or
- establish that a recorded expected result occurred in the current session.

## Source surfaces

- `src/creative_coding_assistant/orchestration/advisory/demo_showcase_experience.py`
- `demo/v9_5_golden_demo_dataset.json`
- `demo/golden_demo_dataset.json`
- `demo/golden_artifacts/`
- `demo/evaluation/`
- `docs/CAPSTONE_DEMO_SHOWCASE.md`

The older static launcher and compatibility dataset remain test-bound. Current
product behavior is defined by the in-product scenario catalog, normal request
path, streamed evidence, and explicit runtime boundaries.
