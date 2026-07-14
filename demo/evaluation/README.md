# Sanitized Evaluation Fixtures

This folder contains public, reviewer-safe evaluation material. Historical V8
fixtures remain available for comparison, while
`current_product_ragas_evidence.json` is the canonical current-product RAGAS
summary.

## Canonical current-product evidence

Run `v9-current-product-final-retained` evaluated the seven frozen public RAG
cases against benchmark `current-product-retrieval.v1`. All 7 cases were
eligible and scored; none were skipped and no metric failed. The run records
RAGAS 0.4.3, evaluator `gpt-4o-mini`, generator
`gpt-5-mini-2025-08-07`, and embedding model `text-embedding-3-small`.

| Metric | Canonical mean |
|---|---:|
| Context precision | `0.5196428571169692` (51.96428571169692%) |
| Faithfulness | `0.648989898989899` (64.8989898989899%) |
| Answer relevancy | `0.5662963631284655` (56.62963631284655%) |
| Context relevancy | `0.8571428571428571` (85.71428571428571%) |
| Context recall | `0.8095238095238094` (80.95238095238094%) |
| Equal-weight Retrieval Quality macro | `0.6803191571804` (**68.03191571804%**) |

The dataset fingerprint is
`sha256:b5fbc0e7cc9a523658eee8b0fc5cd7c417aa10540f8919e10bc2c4e10a40705f`.
The evidence also records benchmark, retrieval, prompt, generation, KB,
selection, and output fingerprints, evaluator/embedding models, timestamp, and
run ID. It is current-product RAG quality evidence, not a project grade or a
human artistic-quality score.

The evaluation engineering progression was:

| Stage | Retrieval Quality | Decision |
|---|---:|---|
| Corrected baseline | 65.79% | Restored the supported RAGAS collections contract before comparing product changes |
| Iteration 1 | 65.46% | Retained truthful retrieval and concise-grounding work; no improvement claim |
| Iteration 2 | 67.54% | Retained bridge-aware domain routing and corpus-quality filtering |
| Iteration 3 | 62.24% | Rejected; experimental hybrid ranking regressed answer relevancy and recall |
| Final rollback confirmation | **68.03%** | Reverted the ranking experiment and confirmed the retained product/evaluator fixes |

The original primary 61.44% display was an
`EVALUATION_PIPELINE_DEFECT`: it surfaced a disconnected historical four-row
fixture instead of executing and publishing the current retrieval, prompt, and
generation paths. That fixture remains historical evidence below.

`sanitized_ragas_live_sessions.jsonl` follows the same
`LiveSessionEvalSample` schema as local live-session records, but every row is
synthetic and limited to public reviewer-safe prompts plus paraphrased public
or committed documentation context. It is safe to use with external evaluator
providers because it does not contain private conversation history, user data,
workspace paths, secrets, or local Chroma content.

The canonical command for the sanitized public RAGAs run is:

```bash
.venv/bin/python scripts/eval_live_sessions.py \
  --input-path demo/evaluation/sanitized_ragas_live_sessions.jsonl \
  --output-path demo/evaluation/sanitized_ragas_context_precision_results_external.jsonl \
  --metric context_precision \
  --allow-provider-calls
```

`redacted_live_session_ragas_latest4.jsonl` is a stronger latest-live evidence
path: it preserves the latest 4 eligible live-session structure, domains,
source ids, retrieval scores, and timestamps, but replaces private question,
answer, and context text with public reviewer-safe p5.js content before
external evaluation.

The final redacted latest-live RAGAs command was:

```bash
.venv/bin/python scripts/eval_live_sessions.py \
  --input-path demo/evaluation/redacted_live_session_ragas_latest4.jsonl \
  --output-path demo/evaluation/redacted_live_session_ragas_latest4_results.jsonl \
  --metric context_precision \
  --metric faithfulness \
  --metric answer_relevancy \
  --allow-provider-calls
```

Final redacted latest-live results after the public-safe wording refresh:
4 rows, 4 eligible, 0 skipped, 0 metric failures. Averages were context
precision `0.7006944444230672`, faithfulness `0.6875`, and answer relevancy
`0.4419141765019863`.
Limitations: this is a redacted p5.js-only latest-live subset, not raw private
session scoring and not a broad RAG benchmark.

The historical complete sanitized approved-fixture run was recorded on
2026-07-13 with RAGAS 0.4.3, `gpt-4o-mini`, and
`text-embedding-3-small`. It evaluated all 4 eligible rows with no metric
failures. Averages were context precision `0.999999999925`, faithfulness
`0.29583333333333334`, answer relevancy `0.4742546883775048`, and context
relevancy `0.6875`. The equal-weight macro-average across those four measured
dimensions was `0.6143970054089596` (61.44%; exact dashboard computation uses the
unrounded values). Context recall remains missing because this fixture has no
independently justified reference answers.

This approved-fixture score is not current-product or canonical golden-case
coverage. The frozen product catalog contains seven RAG-scoped contracts, and
the current public benchmark executes those seven canonical cases as recorded in
`current_product_ragas_evidence.json`; contract inventory and provider execution
remain separate counts. The fixed historical-fixture answers also contain
claims beyond some retrieved excerpts, so weak faithfulness is retained as
evidence rather than edited away.

The current retrieval-only canonical report is reproducible with
`PYTHONPATH=src .venv/bin/python scripts/report_canonical_retrieval.py --limit 5`.
All 7 retrieval-pack queries returned 5 ranked chunks. Expected-source overlap
is 16/23 (69.57%), improved from the pre-fix 9/23 baseline; requested-domain
coverage is 18/19 (94.74%), improved from 7/19. The measured progression was
9/23 + 7/19, then 12/23 + 11/19 after intent/result diversity, 15/23 + 17/19
after balanced per-domain candidate search, 15/23 + 18/19 after retaining a
bounded requested-domain fallback, 17/23 + 18/19 after limiting repeated chunks,
and 19/23 + 18/19 after chunk-level Three.js manual filtering plus unseen-source
priority. Non-text lineage then showed that the apparent 19/23 peak included
three title-only `p5_reference` chunks and a `tone_js_docs` API-name index, not
substantive evidence. Removing those false positives produced 15/23 + 18/19;
bounded candidate headroom recovered one substantive Three.js manual result for
the final 16/23 + 18/19 report without changing the benchmark, top-k, or source
weights.

The lower final anchor overlap is intentional: it counts only substantive
selected chunks. The remaining domain gap is Shadertoy; its approved
`shadertoy_howto` source has no local indexed chunks, and a focused sync was
blocked by the source returning HTTP 403. This is retrieval-phase coverage
evidence, not end-to-end answer generation, RAGAS quality, or a project score.
Expected source IDs are coverage anchors and are never pinned into results merely
to inflate the report.

The exact final per-case lineage is stored in
`canonical_retrieval_report.json`. It contains no retrieved excerpt text and is
bound to all 1,445 indexed chunk metadata records by KB fingerprint
`sha256:b64323bf14246d63a2294794d5948da6abe130d8dd4a0c7ad5a4b3ac3bca11ae`.
The selected report fingerprint is
`sha256:74acf5d62f669eff64fd5fe4fe176bff04da4fcbdc7a7588e18b85a8a418d1c7`.

The current-product run is now represented by a committed public benchmark and
the sanitized evidence summary above. Raw local Chroma excerpts and private
session rows remain outside that provider boundary; they were not uploaded or
silently reclassified. The historical 61.44% fixture remains useful for
provenance, but it is obsolete as the primary score and is not comparable as a
before/after product run.

The product-authored 35-case catalog is frozen contract coverage, not a claim
that 35 prompts were generated or evaluator-scored. Full evaluation executes
the seven canonical RAG cases and records three current local snapshot lanes:
Creative, Workflow, and Reliability. Snapshot evidence is not folded into the
Retrieval Quality macro.

Weak-row note: `redacted_live_p5_demo_fallback_73f56121` still scored
faithfulness `0.0`. Its retrieved contexts include relevant fallback evidence,
but the top-ranked contexts mix p5 setup/canvas guidance with demo fallback
guidance, and the answer compresses multiple fallback actions into one short
statement. Treat this as a retrieval-ranking plus evaluator/answer-compression
limitation, not as hidden passing evidence.

`private_live_session_ragas_decision.json` records the concrete private-data
decision for `data/eval/live_sessions.jsonl`: latest dry-run selection found
60 total samples, 4 eligible samples, and 56 skipped samples. Raw external
provider scoring is still avoided because those rows contain recorded local
questions, answers, and retrieved contexts; use the redacted latest-live fixture
above for public reviewer evidence.
