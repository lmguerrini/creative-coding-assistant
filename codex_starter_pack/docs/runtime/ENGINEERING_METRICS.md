# V7 Engineering Metrics

## Final Lightweight Validation Set

- `runtime-hygiene`
- `git diff --check`
- Ruff
- compileall
- Runtime Pack dashboard
- Runtime Pack docs/Mermaid
- focused runtime validation where technically relevant

## Reused Evidence

- Full backend pytest evidence is reused from capability records.
- Frontend validation evidence is reused from V7.4, V7.5, V7.7, and later smoke
  evidence.
- Release validation and deployment validation are reused from GitHub CI and
  capability records.

## Latest Focused Runtime Validation

`34 passed, 1 warning, 37 subtests passed` for planning/runtime/LangGraph/V7.9
focused tests. The warning is the carried Chroma/Python deprecation warning.
