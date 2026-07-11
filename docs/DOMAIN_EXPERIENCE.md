# Domain and Knowledge Experience Contract

This document describes the current workstation behavior. It takes precedence
over historical QA fixtures or planning documents when they could be read as a
claim about an active in-product preview.

## Domain delivery

`GET /api/domain-experience` is the canonical, read-only registry used by the
Dashboard and Inspector. It lists every registered creative domain, its
artifact extensions, knowledge-source coverage, validation contract, fallback,
and one of three delivery kinds:

- **Live browser preview:** p5.js, Three.js, and GLSL only. Each is limited to
  the validated browser-generation contract and exposes a runtime fallback.
- **Code/export:** source is produced for use or inspection outside that live
  preview contract. Hydra and React Three Fiber are code/export-only in the
  current product. A client adapter or historical QA artifact is not a claim
  that the active generation path has an in-product live preview.
- **External-tool handoff:** a downloadable package contains a creative brief,
  system specification, parameter schema, implementation notes, validation
  checklist, and handoff boundaries. It does not start, control, or validate
  TouchDesigner, Unreal Engine, Blender Geometry Nodes, Houdini, or another
  external tool.

The Dashboard's **Domains** category shows the complete registry. The
Inspector's **Domains** panel shows the active artifact contract, or the three
validated browser contracts when no artifact is active. Both read the same
endpoint rather than relying on demo labels or request-local retrieval data.

## Knowledge Base and retrieval

The Knowledge Base inventory is persistent local-index metadata, distinct from
the retrieval result for the current request:

- **Registered** counts come from approved official-source registry metadata.
- **Indexed** counts and chunks are read from the local Chroma inventory.
- **Last indexed** is the latest local-record timestamp. It does not prove an
  upstream page is current or unchanged.
- The interface never starts an update. Refresh approved sources explicitly
  from the project root with:

  ```bash
  .venv/bin/python scripts/sync_official_kb.py --all
  ```

The endpoint returns only counts and public source identifiers. It does not
return retrieved text, prompts, private workspace data, or conversation memory.
The Retrieval Inspector remains request-scoped: zero retrieved chunks means no
context was attached to that particular run, not that the persistent Knowledge
Base is empty.

## Multimodal and export boundaries

Image attachments are retained as metadata for the workflow and may be listed
in an export. The workstation does not perform pixel analysis, visual-feature
extraction, or image-to-prompt inference. Audio upload and audio analysis are
not available in the workstation; Tone playback is only initiated through its
explicit browser playback control.

Export packages contain the generated source, manifest, and any appropriate
handoff documents. They do not imply external tool execution, a hosted
deployment, or an independently verified runtime beyond the contract shown for
that domain.
