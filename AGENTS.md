# Fly_paint — Fly Gogh

## Mission

Build a playful local web app where a virtual fly holds a pen, moves over a
canvas, and attempts to redraw an image uploaded by the user using a controller
derived from real Drosophila connectivity. Image copying is the primary MVP
experience. Circles, squares and spirals are supporting training/debug presets.
Working name: Fly Gogh.
The humor: scientists mapped a fly nervous system and we sent it to art school.

This document is project guidance, not an automatic task scheduler. The owner
will start a Codex task. Once started, follow docs/CODEX_START.md and implement
the project, rather than stopping after a plan or another scaffold.

## Owner's hardware and priorities

- MacBook Pro M1, 32 GB unified RAM. Support native ARM64 Python.
- Local CPU-first operation, no paid services required.
- Begin with 512 neurons, then benchmark 1024 and 2048. These are experiment
  sizes, not claims of biological completeness or measured hardware limits.
- Fixed sparse recurrent graph; initially train only a small action readout.
- Keep preprocessing streaming and avoid a dense whole-connectome matrix.
- Start with one environment, short episodes and bounded training runs.
- PyTorch MPS is optional; test actual operators and speed before using it.
  Sparse CPU operations may be preferable. Never assume CUDA is available.
- Target under 8 GB process memory initially, leaving room for macOS and the UI.
  Record actual peak memory, speed, graph size and software versions.

## Scientific data and honesty

Dataset: Male CNS v1.0 / neuPrint male-cns:v1.0.
Sources and file manifest: docs/DATA.md and configs/data_sources.json.
A connectome is wiring, not a ready-to-run brain. This is an artificial model
with chosen dynamics, sensory encoding, action mapping and learning rules.
Do not claim consciousness, reproduction of real fly behavior, or biological
learning from connectome data alone. Graph synapse counts are not measured
electrophysiological weights. Neurotransmitter predictions are not sufficient
to infer every synapse's sign without receptor/context assumptions.

Synthetic demo mode must say "Synthetic demo graph". Real extracted mode must
say "Male CNS v1.0-derived subgraph" and report its actual node/edge count.
Do not label a 512-node graph as a simulation of all 166,700 neurons.
Distinguish observed data, engineering assumptions and untested hypotheses.

## MVP experience

The user uploads a picture, previews the drawing target, starts the experiment,
and watches a fly recreate it stroke by stroke through its movement. Show the
original image, the simplified target and the live drawing with clear labels.
Display episode/generation, measured reward and best drawing. Provide
pause/resume, reset with seed, speed control, PNG export and a short methodology
panel. Include image upload in the first working release; do not defer it.
SVG/replay export and full-color painting can follow the working sketch mode.
Neural stimulation/suppression is an experimental control, not a mood detector.
Show actual controller activity if activity is displayed.

No authentication, cloud accounts, social feed, billing or image-generation API
is required. Do not fake learning with a pre-scripted improving animation.

## Image-to-drawing requirements (MVP)

- Accept local PNG, JPEG and WebP via file picker and drag-and-drop; keep images
  local to the browser/local trainer. No external image service is needed.
- Validate decoded format, file size (initial cap 10 MB), dimensions (initial
  cap 16 megapixels) and corrupt/unsupported inputs. Enforce limits at the local
  API too; do not rely on a filename extension. Provide understandable errors.
- Apply image orientation, composite transparency onto the selected paper color
  and fit with padding so aspect ratio is preserved. Never silently stretch.
- Start with a black-ink sketch of arbitrary uploaded images: portraits, objects,
  pets or simple artwork. Preprocess into a grayscale/contour target, with
  adjustable detail/threshold and a preview. Explain that sketch mode simplifies
  color and fine detail. Do not promise photorealistic reproduction.
- Begin with a 128 x 128 reward target, configurable to 64 or 256. Display the
  canvas at higher resolution from the same recorded strokes. This keeps M1
  training costs bounded independently of original upload resolution.
- Feed target and current canvas features into the controller, including a
  coarse global target representation plus local patches/residual ink around
  the pen. Merely adding an upload button to a target-blind controller is not
  enough. Document these inputs as artificial sensory encoding.
- Every visible ink mark must come from the agent's bounded movement and pen
  actions. Preprocessing may derive a target but must not supply a finished
  drawing, a hidden tracing path or direct next-action instructions.
- Reward target coverage and image similarity alongside off-target/repeated-ink
  penalties. Evaluate foreground precision/recall with explicit blank-target
  handling; raw background-dominated pixel accuracy is not a useful score.
- Blank/near-empty targets should show a useful message and skip training.
  Replacing an image cancels the previous run, clears incompatible state and
  rejects stale progress events. Show progress only for the active image/run.
- Initial MVP may optimize a small readout for the current uploaded image.
  Label this as per-image training and report adaptation separately from
  zero-shot performance. Keep the connectome reservoir fixed. Later train over
  diverse images if generalization is needed; do not retrain the whole brain.
- Circles and spirals remain curriculum/debug fixtures. Include small generated
  or appropriately licensed non-geometric image fixtures. Compare trained and
  untrained attempts for at least one image and document actual results.
- Test upload -> preview -> start -> progressive drawing -> pause/reset ->
  export. Also check aspect ratio, orientation, transparency, bad/oversized
  inputs, blank targets and replacing an image mid-run. Export the drawing
  alone by default, without silently including the reference.

## Controller and environment

Use a sparse graph reservoir as the first engineering baseline:

    h_next = (1-leak)*h + leak*tanh(gain*W@h + B@observation + bias)
    action = readout(h_next)

W[post, pre] encodes an incoming connection. Freeze W and document its weight
normalization. Train a small linear readout first with a seeded evolutionary
strategy; PPO is a later option if evidence warrants its complexity. Separate
controller, environment, reward and rendering modules.

Observations must include pose, local target/canvas features and remaining step
budget. Document what information is supplied artificially. Actions are bounded
turn, forward speed and pen pressure. Fixed simulation timestep; no dependence
on browser frame rate. Never provide desired actions directly as observations.

Use a small raster canvas for reward evaluation. Reward newly covered target
pixels, penalize off-target ink, repeated ink and excessive travel. Penalize
remaining still with an unfinished drawing. Do not grant unlimited reward for
retracing completed strokes. Check easy adversarial behaviors before training.

Compare the learned controller against random actions and a similarly sized
random graph. Evaluate held-out seeds/shapes and report failures. A successful
benchmark of sparse matrix multiplication is not proof of successful training.

## Data pipeline

Run scripts/download_data.py, inspect schemas with scripts/inspect_data.py,
then scripts/preprocess_connectome.py. Initial downloads are annotations and
weighted connectivity; neurotransmitter predictions are optional.
No raw images, segmentation volumes or per-synapse coordinates are needed for
the initial drawing controller. Never put raw data or trained checkpoints in Git.

The initial extractor selects high-connectivity annotated nodes and retains
actual edges between them. This is a generic engineering baseline; it does NOT
establish a biologically meaningful visual-to-motor circuit. Improve selection
using verified cell annotations, preserve direction/reachability, document
omitted connections, and never invent missing neural links.

Keep original node IDs as strings across Python/JSON/JavaScript. Record exact
source URLs, dataset release, downloaded checksums, extraction settings and
actual counts. Download failures must remain visible; do not silently replace
real data with demo data.

## Preferred architecture

- apps/web: React + TypeScript + Vite, Canvas 2D first.
- services/trainer: Python + FastAPI, simulation and bounded training jobs.
- src/flypaint: reusable sparse graph and controller utilities.
- scripts: download, inspect, extract, benchmark.
- data/demo: explicitly synthetic tiny committed graph.
- data/raw and data/processed: ignored data artifacts.
- configs: data manifest and MacBook settings.
- docs: methodology, provenance, hardware notes and implementation task.

Use WebSocket events with validated schemas for backend progress. Keep long
training outside the UI event loop. Bound queues and support cancellation.
Add PyTorch only when the selected training algorithm needs it. Do not add
unused libraries, placeholder APIs or commands that do not exist.

## Visual direction

A strange science-museum art exhibit: warm paper canvas, black ink, dark
instrument panels, restrained acid-green activity and red annotations, an
expressive fly with a tiny pen. Readable typography and handwritten accents.
Avoid purple AI gradients, excessive glass, fake charts and neon overload.
Possible copy: "Send the fly to art school", "No art school. One pen."
All controls keyboard accessible, visible focus, no color-only status, respect
reduced motion and support mobile. Target smooth rendering independently of
training throughput; do not promise full-connectome real-time performance.

## Engineering and validation

Inspect existing files and preserve user changes. Keep the project runnable.
Use deterministic seeds, clear interfaces, lock dependency versions once tested
and protect tokens with environment variables. Never commit credentials.
Run existing checks for affected code. Current checks are unittest and
compileall; add frontend checks when a frontend exists. Do not claim an
unavailable command passed. Tests should cover graph orientation, missing or
duplicate IDs, bad weights, reproducibility, reward exploits, boundaries,
pause/reset, exports and the actual UI flow when implemented.

Before handing off, state implemented behavior, commands actually run, measured
results, limitations and next steps. An untrained demo is not the completed MVP.
Attribute the dataset and indicate transformations under its CC BY license;
do not imply that the project's own code already has a license.
