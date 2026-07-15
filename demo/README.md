# Reproducible Demo Assets

This directory contains committed scenarios, prompts, browser harnesses,
creative-code artifacts, and sanitized evaluation fixtures. Demo Mode in the
workstation remains the primary product surface; the static launcher is a local
fallback for inspecting committed assets.

## Product path

1. Start the API and Next.js workstation as described in the
   [Installation Guide](../docs/INSTALLATION_GUIDE.md).
2. Open `http://127.0.0.1:3000` and choose Demo Mode.
3. Load a scenario, then submit it through the normal assistant composer.
4. Inspect the streamed route, retrieval evidence, artifacts, preview state,
   and explicit failures.

Loading a fixture prepares inputs only. It does not prove that a provider,
retriever, or browser runtime succeeded in the current session.

## Asset map

- `v9_5_golden_demo_dataset.json` and `v9_5_exact_prompt_library.md`: current
  scenario metadata and prompts.
- `golden_demo_dataset.json` and `final_demo_suite.json`: compatibility data for
  the static launcher and passive demo metadata.
- `golden_artifacts/`: generated source plus local browser-harness evidence.
- `evaluation/`: public-safe current-product summaries and sanitized historical
  fixtures.
- `final_demo_launcher.html`: local static asset browser, not a hosted product.

## Boundaries

- Live generation, embeddings, retrieval sync, and evaluator scoring require
  explicit provider configuration.
- Browser QA evidence applies only to the recorded harness and dependency
  versions.
- Raw sessions, private evaluator payloads, local databases, Chroma data, and
  backups are excluded.
- External creative applications remain code/export continuation targets.

See the [Product Demo Guide](../docs/CAPSTONE_DEMO_SHOWCASE.md) and
[User Manual](../docs/USER_MANUAL.md).
