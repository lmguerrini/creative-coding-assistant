# VERSION_CONTINUE_PROMPT.md - V7

Read this file first. Then continue the active V7 runtime exactly.

## Operating Rule

Use the Runtime Pack as source of truth. Do not rely on chat memory when Runtime Pack, Git, tags, tests, or ledgers provide stronger evidence.

## Phase -1 Runtime Hygiene

Before Phase 0, validate the local engineering state:

- latest local commit format
- AuthorDate == CommitDate
- working tree cleanliness
- runtime historical consistency
- release reconciliation
- remote GitHub CI success for the latest released capability/tag before the
  next capability begins
- duplicate/strange-file gate for project paths and local Git index artifacts

If the latest local commit has not been pushed and its message does not follow
the required project format, automatically amend only the commit message,
preserve commit contents, staged tree, AuthorDate, and CommitDate, then re-run
commit format validation.

Required commit message format:

```text
Title

- Exactly one bullet
- Exactly one bullet
- Exactly one bullet
```

If the commit has already been pushed and its message is invalid, stop for HITL.

## Duplicate / Strange File Gate

Before Codex Engineering Audit, HITL, Junie handoff, freeze, merge, push, or
tag gates, run the duplicate/strange-file gate. The gate must detect:

- `* 2.py`, `* 2.md`, `* 2.yml`, `* 2.yaml`, and `* 2.toml`
- `.git/index 2`, `.git/index 3`, `.git/index 4`, and `.git/index 5`
- copy, `.orig`, `.rej`, `.bak`, `.tmp`, `.swp`, and editor backup artifacts

If a matching artifact is tracked, stop for HITL. If a matching artifact is
untracked and safely local, remove it automatically and re-run the gate. The
canonical helper is:

```bash
python scripts/v7_quality_gates.py runtime-hygiene --fix
```

CI may run the same helper without `--fix` to block accidental tracked or
checkout-visible artifacts.

## Production Scope Language

V7 production readiness means ready for the declared V7 scope: capstone project,
portfolio project, MVP foundation, and first deployable HoloGenesis foundation.
It does not mean enterprise/public-SaaS production readiness. Authentication,
rate limiting, WAF, TLS termination, multi-user authorization, managed backup,
cloud deployment automation, HoloMind integration, and other platform concerns
remain future scope unless a V7 task explicitly says otherwise.

## Commit Readiness Gate

Before any Codex Engineering Audit / HITL gate, validate all commit hygiene
rules:

- commit exists
- commit is local and has not been pushed
- tracked worktree is clean
- AuthorDate == CommitDate
- commit title is present
- exactly one blank line follows the title
- exactly three contiguous bullet points are present
- no blank lines exist between bullet points
- no additional paragraphs follow the bullet block
- no trailing blank lines exist inside the message

If any commit-message hygiene rule fails and the commit has not been pushed,
automatically amend only the latest commit message, preserve commit contents,
preserve AuthorDate, preserve CommitDate, and re-run Commit Hygiene validation.
Do not stop for HITL for an unpushed amendable commit-message hygiene failure.

If the commit has already been pushed and any Commit Readiness Gate rule fails,
stop for HITL.

## Minimal User Prompt Supported

The user may say:

```text
Continue V7.
```

You must then:

1. Run Phase -1 Runtime Hygiene.
2. Run Phase 0 Runtime Pack Validation.
3. Run Release State Reconciliation.
4. Run Historical Runtime Consistency Gate.
5. Run the Remote GitHub CI Verification Gate for the latest released
   capability/tag.
6. Verify branch, active capability, active task, and prior tags.
7. Continue from the active task.
8. Stop only at required HITL gates, validation failures, product bugs, Runtime Evolution proposals, fix approvals, freeze, or human release operations.

## Never Automate

Never perform merge, push, tag, Capability Freeze, Version Freeze, Runtime Evolution approval/application, Product Bug acceptance, Capability-Scoped Fix approval, Version-Scoped Fix approval, or any HITL decision.

## Always Update Progress

Do update runtime and capability progress files after every completed task unless the active task explicitly forbids progress mutation for a scoped reason. “Do not modify Runtime Pack structure” never means “do not record progress.”

## Pre-HITL Implementation Commit Rule

Codex Engineering Audit / HITL gates occur only after task-scoped implementation changes have been validated and committed.

Before stopping for a Codex Engineering Audit / HITL gate:

- verify the tracked diff is strictly scoped to the active task/capability
- run required task validation
- stage only the validated task-scoped source, test, architecture, and documentation files
- create one implementation commit using the required project commit message format
- verify the full Commit Readiness Gate before requesting Codex Engineering Audit / HITL review

If source, test, architecture, or documentation changes are uncommitted at audit time, Codex must complete this validation/stage/commit sequence before stopping for HITL. Runtime Pack workflow-only corrections may remain outside the product implementation commit when the Runtime Pack path is intentionally untracked or ignored.

## Phase 0 Required Gates

- Phase -1 Runtime Hygiene must already be complete.
- Branch and progress synchronization.
- Pre-capability cleanliness gate on project-authored paths only.
- Roadmap synchronization.
- Roadmap item to task coverage.
- Release State Reconciliation.
- Historical Runtime Consistency.
- Runtime Ledger Integrity.
- Prior tag ancestry.
- Remote GitHub CI Verification Gate for the latest released capability/tag.

## Remote GitHub CI Verification Gate

After merge, push, and tag complete for a capability, verify the remote GitHub
CI run for the released commit/tag is completed and green before starting the
next capability. If the remote CI run is missing, pending, failed, cancelled,
or cannot be verified, stop before beginning the next capability and report the
blocking state.

## Product Bug Rule

If a required smoke/E2E validation reveals a real product bug, stop for Product Bug HITL. Do not classify it as a non-blocking audit observation.

## Runtime Evolution Rule

Runtime Evolution may be proposed but never applied automatically.
