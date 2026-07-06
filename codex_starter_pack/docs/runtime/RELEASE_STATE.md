# V7 Release State

## Current State
V7 is frozen. The final released capability baseline is `v7.11.0` at
`016fe56f930112afb1085b7a1ec12dceb78d2947`.

The Runtime Pack final reconciliation and V7 Freeze commits are local
version-review commits above the release baseline. They are not new V7
capability releases and do not perform merge, push, or tag operations.

## Release Baseline Refs

- `main` points at `016fe56f930112afb1085b7a1ec12dceb78d2947`.
- `origin/main` points at `016fe56f930112afb1085b7a1ec12dceb78d2947`.
- `feature/planning-runtime-decomposition` points at `016fe56f930112afb1085b7a1ec12dceb78d2947`.
- Tag `v7.11.0` points at `016fe56f930112afb1085b7a1ec12dceb78d2947`.

## Freeze Branch

- `version-review/v7-final-3` carries the local Runtime Pack reconciliation and
  V7 Freeze state above the `v7.11.0` release baseline.
- No Git tag is created by the Runtime Pack freeze workflow.

## Release Tags

| Tag | Commit | State |
|---|---|---|
| `v7.1.0` | `bac4db7ed86a27697b8da46431f0805967884a8b` | released |
| `v7.2.0` | `fa29b6462cb3d6e3919f9312e925c90d6dd38c5f` | released |
| `v7.3.0` | `4c6f1fa5f28b47f046ffa322a976febc8af132e7` | released |
| `v7.4.0` | `91c06b9382008ad52573201f30faf9c81375f494` | superseded by `v7.4.2` |
| `v7.4.1` | `92f35b037763ecb63522281acc83d5b401195117` | superseded by `v7.4.2` |
| `v7.4.2` | `79031be1da14eb088f8ef5079e2ef13e0cf46c79` | released |
| `v7.5.0` | `ebaa52eac1ecd7cec81030a4047c318fc3e0613d` | released |
| `v7.6.0` | `97110610a21d4ed96043c6bcd66f7d6ea482af22` | released |
| `v7.7.0` | `e3eb7442924371d52068801a5b63637bacc9b2f8` | superseded by `v7.7.1` |
| `v7.7.1` | `ab49e6655ec1bbc8f14cb4af3946638c5581b142` | released |
| `v7.8.0` | `40d7eca29b0f97e146c72f1fe8b3530108a941eb` | released |
| `v7.9.0` | `44ddb3dbfc873751de7f68a1d8b84e401ab30083` | released |
| `v7.10.0` | `de093b7d5b375e54cd9032246c06a28f11581f60` | released |
| `v7.11.0` | `016fe56f930112afb1085b7a1ec12dceb78d2947` | released |

## Freeze Gate
Closed. Final Codex Audit and Final Junie Audit reported no V7 blockers. V7 is
ready for future V8 planning, but V8 has not started in this repository state.
