# RUNTIME_PACK_SELF_VALIDATION.md - V7

## Purpose
Make the Runtime Pack validate its own state before Codex trusts it.

## Required Self Checks
- active branch matches active capability
- active task exists
- completed tasks are ordered and evidenced
- every roadmap item has explicit task coverage
- every ledger has all required capability sections
- prior tags exist and are ancestors of HEAD
- release state matches tags and progress
- no stale active task after release
