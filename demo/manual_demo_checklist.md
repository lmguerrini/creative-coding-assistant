# Product Demo Validation Guide

This protocol verifies the committed demo path without assigning a score or
directing an evaluator.

## Local services

- The API health and readiness endpoints return successfully.
- The Next.js workstation loads without console errors.
- Demo Mode scenarios populate the normal composer rather than bypassing it.

## Product evidence

- Single, Multi, and Auto publish the route that actually ran.
- Retrieval distinguishes registered, indexed, retrieved, and cited sources.
- Provider, retrieval, persistence, and preview failures remain explicit.
- A prepared preview contract is not reported as a rendered frame.
- Image references clear after submission and are not restored with sessions.

## Reproducible assets

- `demo/v9_5_golden_demo_dataset.json` matches the current scenario catalog.
- `demo/golden_artifacts/qa_manifest.json` identifies the recorded browser
  harness evidence and limitations.
- `demo/evaluation/current_product_ragas_evidence.json` validates against its
  committed schema.
- Historical sanitized fixtures remain labeled separately from current-product
  evidence.

The [Product Demo Guide](../docs/CAPSTONE_DEMO_SHOWCASE.md) describes startup and
the public evidence boundary.
