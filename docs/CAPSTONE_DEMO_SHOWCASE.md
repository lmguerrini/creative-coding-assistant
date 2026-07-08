# Capstone Demo And Showcase Guide

V8.8 prepares Creative Coding Assistant for the final Turing College AI
Capstone demo and showcase. The V8 Grand Engineering Review uses this guide as
release-candidate evidence on `version-review/v8`; it is still not a release
freeze.

## Purpose, Problem, Solution

Purpose: help creative coders turn symbolic, visual, geometric, audio-reactive,
and runtime-oriented intent into grounded browser creative-coding guidance.

Problem: artists and creative technologists often know the experience they want
but must translate it across runtimes, APIs, shader patterns, audio analysis,
debugging constraints, and evaluation tradeoffs.

Solution: Creative Coding Assistant combines a LangGraph-orchestrated backend,
registered creative-coding knowledge sources, Chroma-backed retrieval, a
workstation interface, preview surfaces, artifact planning, critique/refinement
metadata, and local evaluation workflows.

## Capstone Case Alignment

| Case | Alignment | Demo claim | Boundary |
|---|---|---|---|
| Case 5: AI coding assistant | Primary | Creative intent is translated into code-oriented browser runtime guidance. | Do not claim autonomous production delivery. |
| Case 1: RAG knowledge assistant | Primary | Retrieval grounds runtime, shader, and audiovisual guidance in registered sources. | Do not claim complete coverage of all external docs. |
| Case 6: Advanced LLM tools | Primary | LangGraph, Chroma, preview surfaces, and evaluation workflows are part of the architecture. | Do not claim live external DCC/MCP execution. |
| Case 2: Agent automation | Guarded support | Bounded workflow stages and advisory routing/evaluation metadata are explainable. | Do not claim autonomous agent swarms or unattended workflow control. |
| Case 3: Smart document search | Guarded support | Source-grounded search exists for the registered creative-coding KB. | Do not claim generic document search outside indexed/registered sources. |

## Golden Demo Flow

Primary path:

1. State the user problem and the project purpose.
2. Run or narrate the primary prompt from `demo/demo_prompt_library.md`.
3. Show retrieval grounding or explain the registered retrieval scenario.
4. Show creative planning, runtime guidance, and artifact direction.
5. Show the internal preview or `assets/preview_current.png`.
6. Explain critique/refinement and final output readiness.
7. Summarize evaluation, ethics, limitations, and next steps.

## 10-Minute Demo Plan

| Segment | Time | Content |
|---|---:|---|
| Problem and purpose | 1:15 | Explain the creative-coding translation problem and target user. |
| Solution architecture | 1:30 | LangGraph backend, Chroma KB, workstation, preview, evaluation. |
| Primary golden flow | 4:15 | Prompt, retrieval, planning, artifact direction, preview, refinement. |
| Evaluation and ethics | 1:30 | RAGAs workflow, readiness metrics, source grounding, limitations. |
| Challenges and next steps | 1:30 | Demo reliability, fallback mode, Grand Review, post-capstone hardening. |

## 5-Minute Q&A Prep

Likely questions:

- What data does the project use?
  Registered creative-coding sources, local KB chunks, and recorded live-session
  evaluation samples. Do not claim generic web or document coverage.
- How is it evaluated?
  Manual RAGAs workflows over recorded sessions, retrieval scenario coverage,
  demo asset readiness, creative readiness summaries, Grand Review provider
  smoke, and local Chroma retrieval smoke. Fresh live RAGAs scoring still needs
  HITL/privacy approval before local session content is sent to an external
  evaluator.
- What are the biggest limitations?
  Provider availability, retrieval freshness, preview reliability, and careful
  public claim boundaries.
- Is it agent automation?
  Only bounded workflow explanation and advisory metadata should be claimed.
- Does it run Blender, Houdini, TouchDesigner, Unity, Unreal, MCP tools,
  HoloMind, or HOLOiVERSE?
  No.

## SCR Presentation Support

Situation: creative coders need help translating expressive intent into working
technical systems.

Challenge: the translation requires runtime knowledge, source grounding,
planning, preview, debugging, and evaluation under demo time constraints.

Response: CCA demonstrates a bounded assistant workflow, a golden prompt,
retrieval grounding, preview-ready artifacts, evaluation evidence, ethical
boundaries, and rehearsed fallback paths.

## SMART Presentation Support

- Specific: demo one creative-coding workflow, not every roadmap capability.
- Measurable: show retrieval scenario count, demo asset readiness, manual eval
  support, and known guarded findings.
- Achievable: use existing product surfaces and prepared fallback assets.
- Relevant: align primarily to Capstone Cases 5, 1, and 6.
- Time-boxed: keep the main demo to 10 minutes and Q&A to 5 minutes.

## Demo Fallback Mode

Fallback mode uses `demo/golden_demo_dataset.json`, prepared prompts, and
preview screenshots. It should be introduced honestly:

```text
The live dependency is unavailable, so I am switching to the prepared fallback
dataset. This preserves the same product story without pretending this is a new
live provider, retrieval, or preview result.
```

## Next Steps

- Complete V8 Grand Engineering Review validation.
- Apply scoped fixes for claim, reliability, or demo-readiness issues.
- Run final demo rehearsal and focused validation.
- Use `docs/V8_CAPSTONE_EXCELLENCE_SCORECARD.md` as the reviewer evidence map.
- Stop for HITL before showcase upload, merge, push, tag, or final freeze.
