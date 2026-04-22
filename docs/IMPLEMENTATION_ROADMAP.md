# Implementation Roadmap

## Phase 0 - Bootstrap
- repo skeleton
- pyproject
- config
- logging
- tests scaffold

## Phase 1 - Core backend skeleton
- backend package layout
- shared schemas
- basic orchestration pipeline
- client-independent service boundaries

## Phase 2 - Chroma foundation
- chroma client
- collections module
- id conventions
- basic persistence tests

## Phase 3 - Real conversation memory
- recent turn memory
- summary memory
- persistent project memory
- memory retrieval + injection logic

## Phase 4 - Streamlit chat client
- basic chat UI
- streaming responses
- source display
- tool result display
- memory debug panel

## Phase 5 - Official KB sync
- source registry
- fetch/normalize pipeline
- manifests and freshness tracking
- index rebuild flow

## Phase 6 - Retrieval and routing
- retriever
- metadata filters
- query rewriting
- mode-aware routing

## Phase 7 - Assistant modes
- explain
- generate
- debug
- design
- review
- preview

## Phase 8 - Tool layer
- generation tools
- explain/review tools
- debug tools
- design tools
- preview tools

## Phase 9 - Preview subsystem
- HTML bundle builder
- preview renderer integration
- image capture
- download manager

## Phase 10 - Analytics
- usage metrics
- route metrics
- retrieval metrics
- tool metrics
- memory metrics
- dashboard views

## Phase 11 - Live evaluation
- trace logging
- live session evaluation
- RAGAs integration
- evaluation dashboard

## Phase 12 - Hardening
- tests expansion
- edge cases
- docs cleanup
- final UX polish
