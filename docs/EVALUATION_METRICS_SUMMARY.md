# Evaluation Metrics Summary

This document is the concise reviewer interpretation of Creative Coding
Assistant (CCA) evaluation evidence as of 2026-07-14. It keeps current-product
RAGAS, historical fixtures, local retrieval engineering, automated product
checks, and missing human evidence in separate lanes.

## Evidence status vocabulary

| Status | Meaning |
|---|---|
| **Current** | Reproduced against the current product, index, or checked-in implementation |
| **Approved fixture** | Provider-scored result transcribed into the product from committed synthetic/public input data; useful but not current-product output quality |
| **Historical** | Valid only for its dated code/data/runtime scope |
| **Missing evidence** | The required evidence was not collected; it is neither zero nor a pass |
| **Blocked** | A named execution, privacy, provider, or environment constraint prevents the measurement |
| **Human review required** | Automated evidence cannot establish the judgment |

## Current evidence table

| Lane | Dataset or scope | Metric/evidence | Result | Status and interpretation |
|---|---|---|---|---|
| Current-product RAGAS | Seven frozen public RAG cases, run `v9-current-product-final-retained` | Context precision | `0.5196428571169692` (51.96428571169692%) | **Current**; reference-aware, seven eligible/scored cases |
| Current-product RAGAS | Same run | Faithfulness | `0.648989898989899` (64.8989898989899%) | **Current** |
| Current-product RAGAS | Same run | Answer relevancy | `0.5662963631284655` (56.62963631284655%) | **Current** |
| Current-product RAGAS | Same run | Context relevancy | `0.8571428571428571` (85.71428571428571%) | **Current** |
| Current-product RAGAS | Same run | Context recall | `0.8095238095238094` (80.95238095238094%) | **Current**; independently authored reference answers/contexts supply the denominator |
| Current-product RAGAS | Same run | Equal-weight macro across the five measured means | `0.6803191571804` (**68.03191571804%**) | **Current Retrieval Quality**; not a project grade or artistic-quality score |
| Historical provider-scored RAGAS | Four-row committed synthetic/public fixture | Four-metric equal-weight macro | `0.6143970054089596` (61.44%) | **Historical approved fixture**; obsolete as the primary score and has no context-recall result |
| Current local retrieval | Fixed seven-query retrieval pack, top 5, 1,445-chunk indexed snapshot | Substantive expected-source overlap | 16/23 (69.57%), from 9/23 (39.13%) | **Current** local coverage indicator; not RAGAS or answer quality |
| Current local retrieval | Same report | Requested-domain coverage | 18/19 (94.74%), from 7/19 (36.84%) | **Current**; Shadertoy remains unindexed after HTTP 403 |
| Current local retrieval | Same report | Queries returning requested result count | 7/7 returned five results | **Current** availability evidence, not relevance by itself |
| Evaluation contract catalog | 35 deduplicated product-authored prompt contracts | Contract coverage | 35 stable catalog entries | **Current contract inventory**; not 35 generated/evaluator-scored cases |
| Full evaluation | Seven canonical RAG cases plus current Creative, Workflow, and Reliability snapshots | Executed/scored RAG cases | 7/7, 0 skipped, 0 metric failures | Snapshot lanes remain local and separate; they are not additional RAGAS cases |
| Human evaluation | Current creative and reviewer experience | Aesthetic quality, usefulness, clarity, accessibility judgment | no completed study | **Missing evidence / human review required** |

## Interpreting the current 68.03% result

The canonical public evidence was recorded on 2026-07-14 with RAGAS 0.4.3,
evaluator model `gpt-4o-mini`, generation model
`gpt-5-mini-2025-08-07`, and embedding model `text-embedding-3-small`. It
reports all seven eligible current-product RAG cases, zero skips, and zero
metric failures. The score contract is:

```text
(context precision + faithfulness + answer relevancy
 + context relevancy + context recall) / 5
= 0.6803191571804
```

The dataset fingerprint is
`sha256:b5fbc0e7cc9a523658eee8b0fc5cd7c417aa10540f8919e10bc2c4e10a40705f`.
The evidence JSON also records the benchmark version, run ID, evaluator and
embedding models, timestamp, and retrieval/prompt/generation/KB/selection/output
fingerprints.

It must not be described as “68.03% accurate,” a project grade, universal
assistant quality, or human artistic judgment. It is the equal-weight macro of
five RAGAS dimensions on this frozen seven-case current-product benchmark.

## Why 61.44% is historical

The earlier dashboard macro was:

```text
(context precision + faithfulness + answer relevancy + context relevancy) / 4
= 0.6143970054089596
```

That result used four synthetic/public rows and had no independently justified
context-recall denominator. It remains legitimate historical fixture evidence,
but it was disconnected from current retrieval, prompt, generation, and
benchmark state. Presenting it as primary Retrieval Quality was classified
`EVALUATION_PIPELINE_DEFECT`, not a product-quality or benchmark failure.

## Engineering iteration ledger

| Stage | Score | Decision and evidence |
|---|---:|---|
| Corrected baseline | 65.79% | Repaired the evaluator integration before comparing product work |
| Iteration 1 | 65.46% | Kept truthful retrieval and concise grounding; did not claim improvement |
| Iteration 2 | 67.54% | Retained bridge-aware domain routing and corpus-quality exclusions |
| Iteration 3 | 62.24% | Rejected because answer relevancy and context recall regressed |
| Final rollback confirmation | **68.03%** | Removed the ranking experiment and confirmed the retained fixes on all 7 cases |

Retained product/evaluator fixes include full bounded candidate headroom per
requested domain, bridge-aware routing across retrieval and prompt surfaces,
exclusion of the verified index-only Tone.js source, filtering of a
non-actionable Three.js documentation gap, concise grounded-answer guidance,
and the RAGAS collections-v2 reference-aware evaluator path.

Iteration 3's hybrid BM25 reranker, retrieval-term/CamelCase normalization,
numeric source-novelty bonus, and experimental three-source cap were reverted.
The shader case improved, but the pack-level regression showed that the change
did not generalize. The final path returns to semantic-distance selection with
the retained bounded diversity behavior.

The 85% target was not reached. The weakest retained means are context precision
(51.96%), answer relevancy (56.63%), and faithfulness (64.90%). The benchmark is
only seven cases; evaluator judgments can vary; Shadertoy remains unavailable
after an HTTP 403; and no automated score establishes aesthetic quality.

## Interpreting the current retrieval gain

The canonical report uses the same seven committed queries, expected-source
anchors, requested domains, and top-five limit across the comparison. No query,
top-k, source weight, fixture, or scoring rule was changed to raise the result.

The engineering progression reached an apparent 19/23 expected-source overlap,
but lineage inspection showed title-only and index-only chunks that were not
substantive evidence. Those false positives were removed. Bounded candidate
headroom then recovered one substantive Three.js manual result, leaving the
truthful final 16/23. This lower verified number is the current claim.

`demo/evaluation/canonical_retrieval_report.json` records non-text result
lineage and binds the report to:

- 1,445 indexed chunk metadata records;
- KB fingerprint
  `sha256:b64323bf14246d63a2294794d5948da6abe130d8dd4a0c7ad5a4b3ac3bca11ae`;
- selection fingerprint
  `sha256:74acf5d62f669eff64fd5fe4fe176bff04da4fcbdc7a7588e18b85a8a418d1c7`.

The remaining requested-domain gap is Shadertoy. Its approved source has no
local indexed chunks, and a focused sync was blocked by HTTP 403. It stays
explicit rather than being injected into results or counted as a successful
source.

## Reproduce the local, read-only report

```bash
PYTHONPATH=src .venv/bin/python scripts/report_canonical_retrieval.py --limit 5
```

The command prints JSON and does not overwrite the committed report, benchmark,
or KB data. It sends the committed public query strings to the configured
embedding provider when an embedding is required; retrieved local excerpts
remain local. Compare the new collection fingerprint and chunk count before
claiming exact reproduction.

## Prepare evaluation without provider calls

Run a dry-run manifest against the approved fixture:

```bash
LANGSMITH_TRACING=false .venv/bin/python scripts/eval_live_sessions.py \
  --input-path demo/evaluation/sanitized_ragas_live_sessions.jsonl \
  --output-path /tmp/cca-approved-ragas-results.jsonl \
  --metric context_precision \
  --metric faithfulness \
  --metric answer_relevancy \
  --metric context_relevancy \
  --dry-run
```

`--dry-run` prepares/persists the manifest but does not run RAGAS. The explicit
tracing override prevents a separately configured LangSmith client from making
this dry run an external telemetry event. Inspect the input and manifest before
any external call.

## Optional approved-fixture provider scoring

The following command is an explicit paid/external operation and can produce
non-identical model-evaluator output on a later date. Run it only with an
approved public fixture, evaluator credentials, expected-cost review, and a
noncanonical output path:

```bash
LANGSMITH_TRACING=false .venv/bin/python scripts/eval_live_sessions.py \
  --input-path demo/evaluation/sanitized_ragas_live_sessions.jsonl \
  --output-path /tmp/cca-approved-ragas-results.jsonl \
  --metric context_precision \
  --metric faithfulness \
  --metric answer_relevancy \
  --metric context_relevancy \
  --allow-provider-calls
```

Never replace the input with `data/eval/live_sessions.jsonl` or local Chroma
excerpts merely to complete a run. Those records can contain private prompts,
answers, and retrieved contexts. The approved fixture is synthetic/public; the
raw local session file is private by default.

If tracing is separately desired, authorize it as an additional external data
transfer instead of silently removing the `LANGSMITH_TRACING=false` guard.

## Other quality lanes

- Backend and frontend tests establish their asserted contracts, not artistic
  merit.
- Browser smoke establishes current interaction and preview mechanics for its
  covered scenarios, not universal browser compatibility or human usability.
- Creative-readiness metadata can organize review but does not automatically
  score generated art.
- Cost, latency, and provider telemetry apply only when a measured current run
  records them. A configured model name is not usage evidence.
- External-tool exports require validation in the target tool; CCA does not
  count export as execution.

For current runner details and report provenance, see [eval.md](eval.md) and
[DATA_AND_KB.md](DATA_AND_KB.md). Historical Streamlit-era iteration notes are
retained separately in [eval_pipeline.md](eval_pipeline.md) and are not the
current runbook. Ethical limits are summarized in
[ETHICS_PRIVACY_ASSESSMENT.md](ETHICS_PRIVACY_ASSESSMENT.md).
