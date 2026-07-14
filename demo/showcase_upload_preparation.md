# Showcase Upload Preparation

This file prepares a manual submission to
[showcase.turingcollege.com](https://showcase.turingcollege.com). It does not
upload, publish, merge, push, tag, deploy, or approve the project.

## Submission status

| Item | Status | Required action |
|---|---|---|
| Project title | Ready | Use the approved copy below |
| Short description | Ready | Recheck against final README |
| Long description | Ready | Recheck metrics and limitations |
| Repository link | Pending | Add the verified public URL only after access review |
| Live product link | Not available | Do not substitute the local demo URL |
| Current screenshot | Pending final capture | Capture after final UI preflight |
| Demo video | Pending | Record the truthful ten-flow/four-fixture story |
| Slide deck | Ready when final render passes | `outputs/creative-coding-assistant-capstone.pptx` |
| Manual demo rehearsal | Pending per machine | Complete `demo/manual_demo_checklist.md` |
| Public claims review | Pending | Complete every gate in this file |
| Showcase upload | Not performed | Manual submitter action |

## Approved copy

### Title

Creative Coding Assistant

### One-line description

A local AI workstation that turns creative intent into source-grounded,
inspectable browser artifacts with visible workflow and evaluation boundaries.

### Short description

Creative Coding Assistant helps creative coders translate audiovisual ideas
into bounded p5.js, Three.js, GLSL, and Tone.js artifacts. It combines local
retrieval, observable Single/Multi/Auto workflows, controlled browser previews,
refinement, persistence, and evidence-separated evaluation in one workstation.

### Long description

Creative coding starts with expressive intent but quickly becomes a technical
translation problem: choose a runtime, find the right APIs, constrain generated
source, run it safely, refine the result, and explain what actually worked.
Creative Coding Assistant brings those steps into a local workstation. Its
Python workflow can retrieve registered technical knowledge from a local
Chroma index, route requests through bounded Single-Agent, Multi-Agent, or Auto
execution, extract artifacts, and prepare controlled browser previews. The
Next.js interface exposes sources, workflow evidence, runtime health,
fullscreen presentation, refinement, persistence, and recovery.

The current Demo Mode contains ten flows and four canonical browser-validated
fixtures: a silent-first Tone.js constellation, an interactive p5.js aurora
garden, a Three.js r176 orbit sculpture, and a WebGL 1 solar bloom. Retrieval
and RAGAS results remain separate, and partial or missing evidence—such as
configured-model image influence and Context Recall—is stated explicitly.

## Six-point showcase summary

### Problem

Creative coders need to translate ambiguous visual and audio intent across
fragmented documentation, runtime constraints, generated code, and debugging
evidence.

### Solution

A local workstation combines source-grounded guidance, bounded workflow
routing, artifact extraction, controlled previews, refinement, persistence,
and recovery.

### Data

The dated local retrieval snapshot contains 1,445 indexed records. Seven
benchmark queries each returned five results, with 18/19 requested-domain
coverage and 16/23 substantive expected-source-anchor overlap.

### Evaluation

The canonical seven-case current-product RAGAS run reports a 68.03% macro
across five measured dimensions, with no skips or metric failures. The old
61.44% four-row fixture remains historical. Current browser evidence covers
four deterministic showcase fixtures; it is not configured-provider generation
evidence or part of Retrieval Quality.

### Challenge

A real-pointer refinement race appeared after fullscreen restore. The UI
stacking and hover geometry were corrected, the ordinary click stayed in the
regression, the exact fixture gate passed 4/4, and the full Playwright suite
passed 28/28 in the dated local run.

### Next

Broaden the public-safe current-product RAG benchmark, improve context precision
and answer relevancy without changing its frozen contract, run a controlled
image-influence study, complete accessibility and presentation-machine
validation, and split the largest modules.

## Media checklist

### Screenshot

- [ ] Capture the current workstation, not an earlier release image.
- [ ] Prefer a readable 16:9 frame with one canonical visual artifact and its
      evidence context.
- [ ] Ensure prompt, artifact, runtime, and status labels agree.
- [ ] Remove personal sessions, file paths, credentials, notifications, and
      unrelated browser chrome.
- [ ] Provide alt text, for example: “Creative Coding Assistant showing the
      Recursive aurora garden p5.js preview beside artifact and workflow
      evidence.”
- [ ] Optimize file size without making UI text unreadable.

### Video

- [ ] Keep the public edit concise; use the exact problem → solution → data →
      evaluation → challenge → next story.
- [ ] State that the four showcase artifacts are deterministic browser fixtures
      unless an exact configured-provider run is separately identified.
- [ ] Keep Tone.js silent until explicit Start audio; normalize recording volume
      if playback is included.
- [ ] Show one full artifact interaction/refinement/reload flow rather than fast
      cuts that hide state.
- [ ] Include captions.
- [ ] Do not show raw image data, private prompts, local evaluation rows,
      credentials, terminal history, or unsupported runtime claims.

### Slides and supporting links

- [ ] Open and visually inspect
      `outputs/creative-coding-assistant-capstone.pptx`.
- [ ] Confirm slide metrics match the current evaluation artifacts.
- [ ] Confirm the repository URL is public and resolves in a signed-out browser.
- [ ] Do not publish a `127.0.0.1` or `localhost` URL as a live product link.
- [ ] Link to `docs/CAPSTONE_DEMO_SHOWCASE.md` as the reviewer guide.

## Claim gate

- [ ] “Ten Demo Mode flows” matches the product source.
- [ ] The four canonical names and runtime versions are exact.
- [ ] Browser fixtures are not described as fresh model outputs.
- [ ] 18/19 and 16/23 are labeled retrieval coverage, not answer accuracy.
- [ ] 68.03% is labeled a seven-case current-product RAG macro, not a project score.
- [ ] 61.44% is labeled an obsolete historical four-row fixture macro.
- [ ] Context Recall reads 80.95% for the current benchmark and remains missing
      only for the historical fixture.
- [ ] Image transport is distinguished from unverified image influence.
- [ ] Automated validation is distinguished from presentation-machine and
      independent acceptance.
- [ ] External creative tools are described as export/handoff, not live control.
- [ ] The product is described as a local workstation, not a public deployment.
- [ ] No future-work item is written in the present tense.

## Privacy and rights gate

- [ ] Repository history and current tree have received a dedicated secret scan.
- [ ] `.env`, local databases, generated private artifacts, and raw session data
      are absent from the public package.
- [ ] All images, fonts, runtime libraries, music, and source excerpts have an
      understood license or are original/authorized.
- [ ] A public-safe synthetic image is used for the reference-guided demo.
- [ ] Screenshot and video metadata have been reviewed.
- [ ] Public evaluation files contain only approved synthetic/redacted content.
- [ ] The description avoids medical, therapeutic, spiritual-authority,
      copyright-ownership, or universal-quality claims.

## Final manual upload sequence

1. Complete `demo/manual_demo_checklist.md` on the presentation machine.
2. Run the repository/public-boundary review again after the final commit.
3. Open the repository link in a signed-out browser.
4. Review screenshot, video, deck, captions, alt text, and copy together.
5. Compare every metric with the machine-readable source.
6. Ask the designated submitter to approve the final public package.
7. Upload manually to the official showcase site.
8. Open the resulting public page in a signed-out browser and verify every link
   and media asset.
9. Record the public URL and upload time in the submission record.

Stop if any claim, credential, license, privacy, link, or media check remains
unclear. “Pending” is the correct state until the responsible submitter resolves
it.
