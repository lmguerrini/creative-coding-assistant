# Evaluation Metrics Summary

This document is the concise reviewer interpretation of Creative Coding
Assistant (CCA) evaluation evidence as of 2026-07-13. It keeps provider-scored
fixture metrics, current local retrieval engineering, automated product checks,
and missing human evidence in separate lanes.

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
| Provider-scored RAGAS | Product transcription of a result over a four-row committed synthetic/public input fixture | Context precision | `0.999999999925` | **Approved fixture**; four eligible rows, no skips or metric failures; raw result JSONL and run manifest are not tracked |
| Provider-scored RAGAS | Same fixture | Faithfulness | `0.29583333333333334` | **Approved fixture**; weak score is retained, not edited away |
| Provider-scored RAGAS | Same fixture | Answer relevancy | `0.4742546883775048` | **Approved fixture** |
| Provider-scored RAGAS | Same fixture | Context relevancy | `0.6875` | **Approved fixture** |
| Provider-scored RAGAS | Same fixture | Equal-weight macro across the four measured means | `0.6143970054089596` (61.44%) | **Approved fixture**; not a project grade or current-product quality score |
| Provider-scored RAGAS | Same fixture | Context recall | no result | **Missing evidence**; no independently justified reference answers, so recall is not assigned zero or included in the macro |
| Current local retrieval | Fixed seven-query retrieval pack, top 5, 1,445-chunk indexed snapshot | Substantive expected-source overlap | 16/23 (69.57%), from 9/23 (39.13%) | **Current** local coverage indicator; not RAGAS or answer quality |
| Current local retrieval | Same report | Requested-domain coverage | 18/19 (94.74%), from 7/19 (36.84%) | **Current**; Shadertoy remains unindexed after HTTP 403 |
| Current local retrieval | Same report | Queries returning requested result count | 7/7 returned five results | **Current** availability evidence, not relevance by itself |
| Current-product RAGAS | Exact current generation plus local Chroma excerpts | End-to-end provider-assisted metrics | no result | **Blocked by execution environment**; local excerpts are not approved to cross the provider boundary |
| RAG golden contracts | Eight RAG-scoped contracts in the golden dataset | Exact end-to-end RAGAS coverage | 0/8 captured in an approved exact-query fixture | **Missing evidence**, not eight failures |
| Human evaluation | Current creative and reviewer experience | Aesthetic quality, usefulness, clarity, accessibility judgment | no completed study | **Missing evidence / human review required** |

## Interpreting the 61.44% result

The committed product transcription describes an approved-fixture run recorded
on 2026-07-13 with RAGAS 0.4.3, evaluator model `gpt-4o-mini`, and embedding
model `text-embedding-3-small`. It reports all four eligible synthetic/public
rows, zero skips, and zero metric failures. It is the latest defensible RAGAS
summary, but it is historical/fixture-scoped relative to current-product
behavior; the raw scored JSONL and run manifest are not tracked artifacts.

The dashboard macro is:

```text
(context precision + faithfulness + answer relevancy + context relevancy) / 4
= 0.6143970054089596
```

It is a summary of four metric means on one approved fixture. It must not be
described as “61.44% accurate,” a capstone grade, current live-session quality,
golden-case coverage, or a measured improvement over the local retrieval
report. The fixed fixture contains answer claims not fully supported by every
retrieved excerpt, which is why its low faithfulness remains useful evidence.

Context recall is unavailable because the rows do not contain independently
justified reference answers. Omitting it is the honest contract; inserting zero
would falsely turn missing evidence into measured failure.

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
