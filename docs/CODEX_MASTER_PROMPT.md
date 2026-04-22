# Codex Master Prompt

Use this as the first project-level instruction in Codex.

You are implementing a production-minded Creative Coding Assistant.

Project goals:
- Python-first architecture
- Streamlit V1 client
- Chroma as the only persistent database
- Future-ready for a Next.js frontend without major refactors
- Modular, testable, maintainable code
- True multi-turn conversation memory
- Streaming responses
- Official-doc-driven RAG
- Tool-assisted generation, explanation, debugging, design, review, and preview
- Live session evaluation
- Advanced analytics dashboard

Strict implementation rules:
- Never work directly on main
- One feature per branch
- Create small commits only
- Keep files small; target 150-250 LOC, soft cap 300 LOC
- Avoid giant hub files
- Separate UI from backend logic
- Prefer explicit routing over opaque agent loops
- Add tests with each feature
- If a requested task is too large, split it into sub-steps before coding
- Before implementing a complex task, produce a short plan
- After coding, summarize changed files, key decisions, and checks performed

Architecture principles:
- Frontend-agnostic backend
- Chroma collections separated by concern
- Memory = recent turns + running summary + persistent project memory
- Preview = controlled preview pipeline, not arbitrary code execution
- Official KB sync only from approved official sources

When uncertain:
- choose the simpler implementation that preserves scalability
- avoid premature abstraction
- ask for confirmation only if a decision would materially affect architecture
