# Creative Coding Assistant

Creative Coding Assistant is a Python-first assistant for generative visuals,
interactive sketches, and graphics programming workflows.

The initial client is Streamlit. The backend is structured so another client,
such as a TypeScript web app, can use the same service layer later.

## V1 Scope

The first version focuses on:

- Three.js
- React Three Fiber
- p5.js
- GLSL

Core capabilities are planned around official-documentation retrieval,
multi-turn project memory, streaming responses, controlled previews, live
evaluation, and analytics.

## Architecture

Persistent application data is stored in Chroma collections separated by
concern. The Streamlit client stays thin and calls frontend-agnostic backend
services under `src/creative_coding_assistant/`.
