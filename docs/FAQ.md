# Frequently Asked Questions

## What is Creative Coding Assistant?

Creative Coding Assistant (CCA) is a local AI-assisted workstation for planning,
generating, grounding, inspecting, refining, previewing, and exporting
creative-coding artifacts. Its primary capstone case is retrieval-augmented
generation, supported by agent workflow orchestration, code generation, a
bounded image-input path, and retrieval/evaluation engineering.

## Is an API key required?

Not for reading the repository, inspecting committed evidence, running most
tests, or reviewing bounded local/fallback UI paths. Live provider generation,
provider embeddings, knowledge-base refresh where embeddings are required, and
provider-scored RAGAS evaluation normally require authorized credentials and
may incur cost.

## What do Single Agent, Multi Agent, and Auto prove?

They select product workflow routes. Auto also exposes the resolved route. The
current published graph and Inspector show what the request used. A label alone
does not prove parallel execution, independent autonomy, a provider call, or
output quality.

## Which outputs run inside CCA?

p5.js, Three.js, GLSL, and Tone.js have live browser previews. Tone.js requires
an explicit user start before audio plays. Hydra and React Three Fiber are
currently code/export-only. See [DOMAIN_EXPERIENCE.md](DOMAIN_EXPERIENCE.md).

## Does CCA run TouchDesigner, Unreal, Blender, or Houdini?

No. It can prepare bounded handoff code/packages for TouchDesigner, Unreal
Engine, Blender Geometry Nodes, and Houdini, but it does not execute, control,
deploy to, or validate those external tools. A user must inspect and run the
handoff in the target environment.

## Which browser is supported?

Current automated end-to-end evidence uses Playwright Chromium. The UI targets
modern browser APIs, but this repository does not claim equivalent current
automated coverage for every Chromium variant, Firefox, or WebKit. Tone.js also
depends on the browser's user-activation/audio policy.

## Is the image attachment really multimodal?

It has a real but partial multimodal request contract when the user explicitly
submits and accepted image pixels are serialized beside the prompt in the
configured-provider payload. Current evidence proves that payload construction,
not live provider receipt, use, or image influence. Merely selecting or
previewing a file proves none of those things. CCA accepts up to four PNG, JPEG,
WebP, or GIF references, each no larger than 1 MiB, and validates the file
signature as well as declared type.

## Is an attached image stored in the session?

The selected image is browser-local until submission, clears from the composer
after submission, and is not restored through normal session persistence. A
project export created while an image is still queued can include it, so review
the bundle before sharing.

## Does CCA accept audio input?

No. Audio upload and audio analysis are not supported. Tone.js provides local
browser playback after an explicit user gesture; that is different from
uploading or analyzing audio.

## Does “registered source” mean the assistant used it?

No. Registered, indexed, retrieved, and cited are distinct states. A source can
be registered but fail to index, or be indexed without appearing in a request's
top results. Inspect request-specific retrieval evidence. See
[DATA_AND_KB.md](DATA_AND_KB.md).

## What does the 68.03% score mean?

It is the equal-weight macro of Context Precision, Faithfulness, Answer
Relevancy, Context Relevancy, and Context Recall on seven frozen current-product
public RAG cases. All seven were eligible and scored with no skips or metric
failures. It is not 68.03% accuracy, a project grade, or artistic-quality
judgment. The former 61.44% four-row fixture remains historical and has no
context-recall result.

## Did retrieval improve beyond that score?

The current local retrieval report improved substantive expected-source overlap
from 9/23 to 16/23 and requested-domain coverage from 7/19 to 18/19 on the same
fixed seven queries and top-five limit. This is a separate local retrieval
coverage result, not the same measure as the 68.03% five-metric RAGAS macro.

## What remains blocked after current-product RAGAS completed?

The canonical run uses the reviewed committed public benchmark. Raw local
session text and arbitrary local Chroma excerpts remain outside the external
generation/evaluation-provider boundary. Uploading private session or index
content is still not an acceptable way to broaden the score.

## Does Full run all 35 catalog prompts?

No. The 35 entries are stable contract coverage. Full executes the seven
canonical current-product RAG cases and records current local Creative,
Workflow, and Reliability snapshots. Those snapshots are not additional model
generations or RAGAS rows.

## Has a human usability or artistic-quality study been completed?

No completed human evaluation is claimed. Automated tests can establish
specific mechanics, state transitions, and runtime checks; they cannot establish
artistic value, broad usability, accessibility conformance, or reviewer
satisfaction. A reviewer may record new observed evidence separately.

## Is data “safe” because it is local?

Local paths reduce some network exposure but do not imply encryption, access
control, retention limits, secure deletion, or safe publication. Session
databases, artifacts, Chroma data, logs, evaluation JSONL, exports, backups, and
screenshots each need deliberate handling. See
[ETHICS_PRIVACY_ASSESSMENT.md](ETHICS_PRIVACY_ASSESSMENT.md).

## Can I enable LangSmith tracing?

Only after deliberately reviewing what will leave the workstation, provider
terms, retention, access, and cost. Application settings default tracing to off.
Keep it false or unset for ordinary local review; never put its key in a public
file.

## Does an export prove the project works?

No. Export proves that CCA produced a handoff bundle. Review its files, private
data, licenses, and attribution, then execute and validate it in the intended
environment.

## Where should I start as a reviewer or user?

Use [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md) for the evidence-focused tour,
[USER_MANUAL.md](USER_MANUAL.md) for workflows, and
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) when a path fails.
