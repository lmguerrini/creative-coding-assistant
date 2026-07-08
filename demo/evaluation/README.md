# Sanitized Evaluation Fixtures

This folder contains public, reviewer-safe evaluation material for V8 release
candidate evidence.

`sanitized_ragas_live_sessions.jsonl` follows the same
`LiveSessionEvalSample` schema as local live-session records, but every row is
synthetic and limited to public reviewer-safe prompts plus paraphrased public
or committed documentation context. It is safe to use with external evaluator
providers because it does not contain private conversation history, user data,
workspace paths, secrets, or local Chroma content.

The canonical command for the privacy-approved RAGAs run is:

```bash
.venv/bin/python scripts/eval_live_sessions.py \
  --input-path demo/evaluation/sanitized_ragas_live_sessions.jsonl \
  --output-path demo/evaluation/sanitized_ragas_context_precision_results_external.jsonl \
  --metric context_precision \
  --allow-provider-calls
```

Only the sanitized fixture is approved for provider-backed evaluator calls in
this release-candidate pass. Recorded local live-session data remains private
and requires separate HITL/privacy approval before external scoring.

`private_live_session_ragas_decision.json` records the concrete private-data
decision for `data/eval/live_sessions.jsonl`: latest dry-run selection found
60 total samples, 4 eligible samples, and 56 skipped samples, but external
provider scoring remains blocked because the rows contain recorded local
questions, answers, and retrieved contexts. The file includes the exact command
to run only after HITL privacy approval.
