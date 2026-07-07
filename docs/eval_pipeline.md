# Retrieval Evaluation Pipeline

## Overview

This pipeline evaluates retrieval quality, not final answer quality.

The main metric in this workflow is `context_precision`. In this project we use
the no-reference RAGAs variant, which checks whether the retrieved chunks are
ranked so that the chunks most relevant to the generated answer appear first.
That makes it a retrieval-ranking metric, not a direct measure of generation
quality or UX quality. See the official RAGAs context precision docs:
[RAGAs Context Precision](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/).

We use `--latest` instead of `--limit` because `data/eval/live_sessions.jsonl`
is append-only. `--limit` walks oldest-first and can easily evaluate stale
samples from before a KB sync, source change, or retrieval fix. `--latest`
keeps the evaluation focused on the most recent live app behavior.

## Standard Workflow

1. Add or refine KB sources when retrieval is weak for a domain.
2. Sync only the affected source IDs.
3. Run fresh manual queries in the Streamlit app to create new live samples.
4. Run RAGAs over the latest samples with `--latest N`.
5. Inspect both:
   - `context_precision` scores
   - `source_ids`

### Step 1: Add New KB Sources

Only add sources when the current KB cannot support the target query style.
Typical examples:

- specs are strong for explanation queries but weak for generation queries
- examples index pages are too noisy
- domain leakage is gone, but the correct domain still lacks practical chunks

### Step 2: Sync Specific Sources

Sync only the affected source IDs so the local Chroma state stays easy to
reason about.

Example:

```bash
SSL_CERT_FILE="$(.venv/bin/python -c 'import certifi; print(certifi.where())')" \
.venv/bin/python -m dotenv run -- \
.venv/bin/python scripts/sync_official_kb.py \
  --source-id glsl_language_spec_460 \
  --source-id glsl_es_language_spec_320 \
  --source-id glsl_mdn_webgl_examples
```

### Step 3: Run Manual App Queries

After sync, open the Streamlit app and run fresh manual queries in the target
domain. This is important because the eval pipeline measures recorded live
samples, not the current KB in the abstract.

Examples:

- `Write a basic GLSL fragment shader`
- `GLSL fragment shader example with color`
- `Create a simple p5.js sketch with a moving circle`

### Step 4: Run RAGAs on the Latest Samples

Use `--latest N`, not `--limit N`.

Example:

```bash
.venv/bin/python -m dotenv run -- \
.venv/bin/python scripts/eval_live_sessions.py \
  --input-path data/eval/live_sessions.jsonl \
  --output-path data/eval/ragas_latest4_context_precision.jsonl \
  --latest 4 \
  --metric context_precision
```

### Step 5: Inspect Scores and Sources

Always inspect the scores together with the retrieved sources:

```bash
jq '{cp: .metrics.context_precision, sources: .source_ids, domains: .domains}' \
  data/eval/ragas_latest4_context_precision.jsonl
```

The score tells you how clean the ranking was. The `source_ids` tell you why.

## Interpretation Guidelines

- `cp ≈ 1.0`
  Very clean retrieval. The most relevant chunks are consistently surfaced
  first.
- `cp ~ 0.7–0.9`
  Good retrieval. Usually production-usable, though still worth checking for
  repeated chunks or mild source noise.
- `cp < 0.5`
  Poor retrieval. Ranking is weak, or the underlying chunks are not suitable
  for the query style.

Important note:
`context_precision` measures how well relevant chunks are ranked in the
retrieved context. It does not directly measure answer quality, code quality, or
overall assistant usefulness. See:
[RAGAs Context Precision](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/).

## Known Pitfalls

- stale `live_sessions.jsonl`
  Old samples can hide recent KB or retrieval improvements
- spec-heavy sources
  Good for explanation queries, weak for generation-oriented queries
- domain leakage
  Mixed-domain `source_ids` usually indicate wrong retrieval scope
- page noise
  MDN or similar pages can still surface navigation-heavy or page-chrome chunks

## Debug Checklist

- Is the domain correct?
- Are the top results example-oriented?
- Is source diversity reasonable, or is one weak source repeating?
- Is the eval using the latest samples?
- Do the `source_ids` match the domains you expected to test?

## Baseline

### Verified latest-eval snapshot

Latest verified `--latest 4` `context_precision` snapshot:

- `react_three_fiber`: `0.9167` to `1.0`
- `p5_js`: `0.0`
- `glsl`: `0.4778`

This snapshot is still influenced by stale recorded samples for `p5_js` and
`glsl`. In particular, the latest GLSL eval sample did not yet include the new
MDN GLSL examples source.

### Post-refresh target baseline

After syncing the corrected KB sources and recording fresh live samples, the
working target baseline is:

- `p5_js ≈ 0.80`
- `glsl ≈ 0.91`
- `react_three_fiber ≈ 0.9–1.0`

### Status

Retrieval pipeline: `PRODUCTION-READY`

That status refers to the workflow and operational pipeline:

- source sync works
- live sample recording works
- latest-sample eval works
- per-domain retrieval issues can be isolated and measured

It does not mean every domain is already at target score in the latest recorded
snapshot. Fresh post-sync live samples are still required before treating a
domain-specific score as current.
