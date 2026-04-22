# Project Context

## Project
Creative Coding Assistant

## Goal
Build a production-minded, domain-specific assistant for creative coding and generative visuals.

## Active domains in V1
- Three.js
- React Three Fiber
- p5.js
- GLSL

## Core constraints
- Python-first architecture
- Streamlit as V1 client
- Chroma as the only persistent database
- Strong modularity
- File size target: 150-250 LOC, soft cap 300 LOC
- Feature branches only
- Small commits only
- Keep architecture ready for a future Next.js frontend without major refactors

## High-level product goals
- True multi-turn conversation memory
- RAG over official docs and curated examples
- Tool-assisted generation, explanation, debugging, design, and preview
- Live streaming responses
- Live evaluation based on current chat session
- Advanced analytics dashboard
- Controlled preview pipeline with downloadable preview images

## Non-goals for V1
- Houdini integration
- TouchDesigner integration
- Generative music integration
- Full arbitrary code sandbox
- Full GLSL playground
- Next.js frontend implementation
