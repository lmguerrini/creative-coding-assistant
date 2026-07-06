# V7 Engineering Workflow Snapshot

## Status
FROZEN

## Final Workflow
- Use Runtime Pack ledgers as source of truth.
- Run Phase -1 runtime hygiene before active continuation.
- Reconcile release state against Git tags, branch history, and CI.
- Keep capability progress and version runtime ledgers synchronized.
- Stop for HITL on merge, push, tag, freeze, Runtime Evolution, product bug,
  and explicit approval gates.
- Prefer validation reuse late in a release when fresh full validation already
  exists.
- Commit validated scoped changes before audit/HITL closure.

## Final Lightweight Gates
- `runtime-hygiene`
- `git diff --check`
- Ruff for touched Python tooling.
- compileall for touched Python tooling.
- Runtime Pack dashboard.
- Runtime Pack docs-mermaid.
- stale wording scan.

## Frozen Boundary
The workflow is frozen as V7 process evidence. Future workflow changes require
explicit post-freeze correction or a future version process.
