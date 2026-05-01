# Project Freeze Checklist

## Completed Core Features

- Streamlit chat client wired to the composed backend service
- Real OpenAI generation with streaming support
- Session memory across chat turns
- Provider selection and settings-backed OpenAI configuration
- Official KB sync pipeline and local sync CLI
- KB indexing with explicit embeddings and local Chroma storage
- Retrieval filtering, deduplication, and domain-aware narrowing
- Query-intent aligned retrieval filters in orchestration
- Retrieval, context, prompt, generation, and trace visibility in the UI
- Live session recording for retrieval evaluation
- Manual RAGAs evaluation workflow over recorded live sessions

## Deferred Features

- Matrix/UI polish
- Advanced preview rendering
- Agentic workflow
- Copy/download code actions
- README finalization

## Known Caveats

- Retrieval quality still depends on source quality inside each domain
- Stale `data/eval/live_sessions.jsonl` samples can misrepresent current KB state
- `context_precision` evaluates retrieval ranking, not final answer quality
- Empty domain selection is allowed; ambiguous prompts then rely on broad retrieval
- Preview mode exists in routing contracts but does not ship advanced preview UX yet

## Final Validation Commands

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check src clients tests scripts
.venv/bin/python -m compileall -q src clients tests scripts
```

## Manual Smoke Tests

Run these in the Streamlit app and inspect both the answer and the collapsed
trace section under the assistant response.

1. Plain Three.js prompt
   - Query: `Create a simple rotating cube in three.js`
   - Expect:
     - answer uses Three.js APIs
     - retrieval domains stay in `three_js` unless the query explicitly spans more
     - trace shows the final retrieval domains and source IDs used

2. React Three Fiber prompt
   - Query: `What is useFrame in React Three Fiber?`
   - Expect:
     - answer stays in React Three Fiber concepts
     - retrieval domains focus on `react_three_fiber`

3. p5.js prompt while UI domains are `three_js + react_three_fiber`
   - Query: `Create a bouncing ball in p5.js`
   - Expect:
     - retrieval follows explicit query intent and uses `p5_js`
     - trace shows:
       - UI selected domains: `three_js`, `react_three_fiber`
       - detected domains: `p5_js`
       - final retrieval domains: `p5_js`

4. React Three Fiber + GLSL prompt
   - Query: `Create a shader material in React Three Fiber using GLSL`
   - Expect:
     - answer bridges both ecosystems
     - retrieval domains include `react_three_fiber` and `glsl`

5. Generic ambiguous prompt using UI-selected domains
   - UI domains: choose `three_js + react_three_fiber`
   - Query: `Create a rotating cube`
   - Expect:
     - retrieval falls back to the UI-selected domains
     - trace shows no stronger detected domain override than the selected set

6. Empty domain selection behavior
   - Clear all selected domains if the current UI session allows it
   - Query: `Create a rotating cube`
   - Expect:
     - request still runs without crashing
     - ambiguous retrieval remains unconstrained
     - explicit-domain prompts such as `Create a p5.js sketch` still detect and use
       the query domain

## Freeze Status

- Functional backend and validation workflow: ready to freeze
- Remaining work is primarily UX polish, preview depth, agentic flows, and final
  project documentation
