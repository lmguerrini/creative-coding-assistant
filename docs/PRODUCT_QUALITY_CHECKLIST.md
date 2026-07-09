# Product Quality Checklist

Creative Coding Assistant is reviewed against a public product quality checklist
before release candidate freeze. This document is intentionally outcome-focused:
it describes the quality bar reviewers should expect without exposing private
review notes, temporary evidence paths, or human-in-the-loop task state.

## Reviewer Experience

- The app opens into a calm, readable chat-first workspace.
- User Mode hides diagnostics and presents only Preview, Code, and Saved output
  surfaces.
- Developer Mode exposes implementation details without overlapping or cropped
  text.
- Demo Mode uses product-facing scenario labels and loads prompts into the normal
  assistant workflow.

## Demo Reliability

- The local demo target is documented and does not require public cloud
  deployment.
- Demo scenarios have a known fallback path when provider, retrieval, or preview
  services are unavailable.
- Generated code is routed to Code/Saved surfaces instead of flooding the chat.
- Preview states distinguish visual output, unavailable preview, and fallback
  guidance clearly.

## Retrieval And Evidence

- Knowledge Base status is visible at the app level.
- Current-run retrieval state is distinct from overall Knowledge Base
  availability.
- Public evidence separates live validation, artifact-only validation, and
  accepted boundaries.
- RAGAs and retrieval evaluation claims remain conservative and traceable to
  recorded evidence.

## Observability And Privacy

- LangSmith tracing is documented as setup-ready unless an active trace has been
  validated in the current environment.
- Public documentation avoids private reviewer process, temporary local paths,
  and internal-only product names.
- `.env.example` contains safe placeholders only; credentials remain local.

## Capstone Coverage

- README and demo documentation cover the problem, solution, data, evaluation,
  challenges, next steps, ethics, SCR/SMART framing, 10-minute demo, and
  5-minute Q&A path.
- Demo scenarios cover p5.js, Three.js, GLSL, Hydra, retrieval-grounded
  answers, concept translation, visual planning, and installation planning.
- Public claims avoid unsupported live DCC/MCP, future product, or unimplemented
  runtime promises.

## Release Hygiene

- The release candidate worktree should be clean before freeze.
- Public docs should contain only reviewer-facing product evidence.
- Internal review tracking should remain in ignored private context.
- Commit history should use descriptive product/change titles before freeze.
