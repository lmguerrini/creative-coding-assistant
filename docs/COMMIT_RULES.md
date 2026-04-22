# Commit Rules

## Branch policy
- One feature = one branch
- No direct work on main
- Merge only after targeted checks pass

## Commit policy
- Keep commits small and reviewable
- Prefer one logical change per commit
- Avoid large mixed commits
- Do not bundle refactor + feature + tests in one commit unless tightly coupled

## Commit message format
Use:
<Imperative summary>

- Bullet 1
- Bullet 2
- Bullet 3

Example:
Add conversation summary memory

- Added summary memory models and writer flow
- Injected summary memory into prompt assembly
- Added tests for summary rollover behavior

## File-size policy
- Target: 150-250 LOC
- Soft cap: 300 LOC
- If a file grows past 300 LOC, split it unless there is a strong reason not to

## Testing policy
Every feature branch should include:
- focused tests for new logic
- no unrelated test rewrites
- smoke check before merge
