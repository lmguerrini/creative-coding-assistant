# Evaluation and Responsible-AI Overview

Creative Coding Assistant evaluates retrieval with a versioned seven-case
current-product benchmark and five RAGAS metrics. The Dashboard owns current
dynamic results; committed evidence is a public-safe projection that excludes
raw prompts, answers, references, and retrieved excerpts.

Historical synthetic fixtures validate evaluator schemas and compatibility.
They do not exercise the complete current retrieval, prompt, and generation
path and therefore are not the current product result. Recorded-session rows,
including redacted ones, remain outside the public repository boundary.

The main trust boundaries are:

- generation can send rendered prompts, selected context, and submitted images
  to the configured provider;
- embeddings can send queries, approved source text, and successful
  conversation turns to the embedding provider;
- provider-scored evaluation requires an explicit authorization flag;
- local sessions, Chroma data, raw evaluation records, and backups remain
  private local state; and
- small samples and evaluator variance limit statistical and artistic claims.

Detailed methodology and controls live in [Evaluation Methodology](eval.md) and
[Ethics and Privacy Assessment](ETHICS_PRIVACY_ASSESSMENT.md).
