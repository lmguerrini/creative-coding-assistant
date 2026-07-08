# Sanitized Evaluation Fixtures

This folder contains public, reviewer-safe evaluation material for V8 release
candidate evidence.

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

Final redacted latest-live results: 4 rows, 4 eligible, 0 skipped, 0 metric
failures. Averages were context precision `0.7006944444251505`,
faithfulness `0.625`, and answer relevancy `0.46063699875040387`.
Limitations: this is a redacted p5.js-only latest-live subset, not raw private
session scoring and not a broad RAG benchmark.

`private_live_session_ragas_decision.json` records the concrete private-data
decision for `data/eval/live_sessions.jsonl`: latest dry-run selection found
60 total samples, 4 eligible samples, and 56 skipped samples. Raw external
provider scoring is still avoided because those rows contain recorded local
questions, answers, and retrieved contexts; use the redacted latest-live fixture
above for public reviewer evidence.
