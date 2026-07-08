# Manual Demo Checklist

Run this checklist before the Capstone demo rehearsal and again before the
showcase session.

## Preflight

- Confirm branch: `version-review/v8` during Grand Review rehearsal.
- Confirm no merge, push, tag, or freeze action is planned without HITL.
- Keep `README.md`, `docs/CAPSTONE_DEMO_SHOWCASE.md`, and
  `docs/CAPSTONE_EVALUATION_ETHICS.md` open.
- Keep `demo/golden_demo_dataset.json` and `demo/demo_prompt_library.md` open.
- Confirm `assets/preview_current.png`, `assets/preview_v1.png`, and
  `assets/preview_v2.png` exist.

## Reliability

- Rehearse the primary flow in 7 minutes.
- Rehearse the case-alignment summary in 90 seconds.
- Rehearse the evaluation and ethics summary in 60 seconds.
- Rehearse the offline fallback in 30 seconds.
- Practice stating provider, retrieval, preview, network, and timing failures
  clearly without implying live success.

## Provider Failure Recovery

If provider access fails:

1. Say the provider path is unavailable in the current environment.
2. Switch to the primary prompt in `demo/golden_demo_dataset.json`.
3. Show the prepared preview screenshot.
4. Continue with architecture, evaluation, ethics, and limitations.

## Retrieval Failure Recovery

If retrieval is unavailable:

1. Say retrieval is not being run live.
2. Show the retrieval demo pack and evaluation workflow references.
3. Explain that recorded-session RAGAs evaluation is manual.

## Preview Failure Recovery

If preview is unavailable:

1. Use `assets/preview_current.png`.
2. Explain that V8.8 did not change preview runtime behavior.
3. Continue with code/artifact planning and critique/refinement explanation.

## HITL Gate

Stop for HITL review before showcase upload, merge, push, tag, or freeze.
