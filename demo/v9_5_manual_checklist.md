# V9.5 Manual Demo Checklist

## Before the rehearsal

- Start the local backend and Next.js workstation.
- Open Demo Mode and verify all ten scenario cards expose input, artifact,
  preview, interaction, validation, and fallback details.
- Confirm the four-card homepage library loads exact p5.js, Three.js, GLSL,
  and Tone.js prompts.
- Keep `demo/v9_5_golden_demo_dataset.json` and
  `demo/v9_5_presenter_notes.md` available for an honest fallback.

## Live browser checks

- Cymatics: visual mounts muted, Start audio is explicit, Stop and Mute are
  visible, and no microphone control is requested.
- p5.js: canvas is nonblank, responds to the stated interaction, and reloads.
- Three.js: renderer is visible, fullscreen opens, and reload preserves the
  preview boundary.
- GLSL: shader compiles or reports its truthful code fallback; never claim a
  blank or failed frame passed.

## Workflow checks

- Retrieval: inspect only current-run retrieval evidence and source boundary.
- Multi-agent: confirm Multi-Agent is selected before sending the prompt.
- Single-agent: confirm Single Agent is selected before sending the prompt.
- Export: use the explicit local export approval and inspect the package.
- Multimodal: attach only a suitable non-sensitive image, then verify the
  resulting preview is self-contained rather than image-fetching.
- Failure recovery: use the controlled failure fixture; ensure a provider
  fallback remains Partial and no live preview is claimed.

## Fallback language

Say: “The live dependency is unavailable, so I am showing the prepared
fallback boundary. This is not a new provider, retrieval, or browser-preview
success.”
