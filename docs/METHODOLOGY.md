# How Fly Gogh works

This is an artificial drawing controller derived from wiring, not a simulation
of a complete fly brain or biological learning. No pretrained drawing skill is
present in the source data. Real and synthetic modes are explicit; a failed
real load raises an error, never switches to synthetic mode.

## Image and environment

PNG/JPEG/WebP are decoded locally with Pillow. Both API and browser cap uploads
at 10 MiB; the API inspects the decoded format and caps dimensions at 16 million
pixels before loading. EXIF orientation is applied, alpha is composited onto
paper RGB (246,241,225), and aspect ratio is preserved with padding. The initial
reward raster is 128×128, optionally 64 or 256. A 3×3 morphological contrast
(maximum minus minimum grayscale) threshold creates contours. This is a target,
not an instruction path. Nearly blank targets cannot train. Free drawing is a
separately labeled, untrained exploration mode.

Each fixed simulation step produces bounded turn (±0.55 radians), travel
(0..size/40 pixels), and pressure (0..1). Ink is deposited only above pressure
0.45 and only with nonzero displacement. Pressure is currently a binary pen
gate, not variable line width. Boundary clipping can reduce actual movement.
The deterministic raster samples movement segments; exports replay those same
segments at higher resolution. No target, reference or fly overlay is included
in the drawing export. Display antialiasing can differ from the reward raster.

Observation has 233 float32 values: five pose/budget values, an 8×8 global target
pool, an 8×8 global residual pool, and target-residual/ink 5×5 egocentric patches
at two spatial scales. These are artificial sensory measurements. No next
waypoint, target centroid, tracing route or desired action is supplied.

## Graph and controller

The official source columns were inspected, not inferred from filenames:
`bodyId` in annotations; `body_pre`, `body_post`, `weight` in connectivity.
All stored IDs remain strings. The generic degree extractor is retained as a
computational baseline (512 nodes / 16,748 edges in this download).

Task extraction restricts the candidate pool to the observed annotation values
`superclass = visual_projection, descending_neuron, vnc_motor` or `class = CX`.
These are candidate populations, not verified drawing modules. Available
`somaSide`, `somaNeuromere` and type labels are preserved in the processed
annotation file. Null labels are not assigned invented meanings.

The script streams 151,856,684 source edge rows, retains candidate-pool sparse
edges, ranks 16 visual inputs and 16 descending + 16 motor outputs by weighted
degree, includes complete directed shortest paths and then connected recurrent
neighbors. It never fabricates missing links. The 512-node result has 22,235
edges, one weak component, five strong components, no isolates, and all 32
selected outputs reachable from selected inputs. The induced subgraph omits
connections crossing its boundary, including connections outside the candidate
pool. See EXTRACTION.json for counts and the scope of omission measurements.

Rows are postsynaptic: W[post,pre]. Incoming counts are normalized by each
retained row sum. Counts remain nonnegative engineering weights; no synaptic
signs are inferred from neurotransmitter predictions. Dynamics:

`h' = .7 h + .3 tanh(.9 W h + B observation + bias)`

W, B and bias stay fixed. Real task mode injects B only into selected input
neurons and reads the selected 32 outputs. The generic and synthetic baselines
use a seeded encoder at all nodes and 32 seeded readout nodes, so they are not
identical sensory-adapter controls for the task graph. The matched random-graph
experiment does preserve the task input/output adapters and encoder seeds.
Only a 3×33 linear readout (99 parameters) evolves. Tanh bounds turn; shifted
and scaled tanh bounds speed and pressure. Displayed activity is measured mean
absolute state from the shown rollout, not a mood estimate.

## Objective and optimization

Unique target hits define foreground precision, recall and F1. No
background-dominated accuracy is used. Episode score is:

`100 F1 + 20 recall - 10 off_target / target_pixels`
`- .02 repeated_pixels - .03 travel_in_canvas_widths - .05 stalled_steps`

Thus retracing does not award new target coverage; repeated ink, unnecessary
travel, off-target ink and stationary unfinished episodes have costs. Tests
exercise these cases against an ideal straight stroke. The scalar objective
can trade coverage for fewer penalties: a higher score need not improve every
metric. Current coverage remains low; see the actual results.

Elitist evolution evaluates 16 candidates × 10 generations initially, with
antithetic Gaussian mutations (sigma .35 × .95^(generation-1)). An incumbent
survives each generation. Two fixed training starts share random numbers across
candidates. Three additional starts (training seed +1000..1002) are evaluated
only after selection, alongside the original readout and random actions.
Models are per-image adaptations, not trained over a large image corpus.
Checkpoints include the readout, encoder, bias, output indices and target, with
source graph fingerprint and evaluation seeds in companion JSON.

## Optimization evidence

`evaluate_ablation.py` freezes checkpoints and tests each observed group plus
all six pairs on identical cat/leaf/circle targets and start seeds. Each disabled
state is zeroed before recurrence and after every update, including its readout
contribution. Neither weights nor row normalization change. F1 is the positive
normalized quality score with a declared 5% maximum relative loss; zero-intact
F1 is reported as undefined, never divided by zero. Absolute deltas are saved.

The three checkpoints do not justify any tested removal under the conservative
all-pairs rule. No physically smaller graph is therefore deployed and no memory
saving is attributed to masks. Group timings cover CPU simulation, but peak RSS
is process-wide: it cannot establish group-specific memory savings. This is
engineering ablation, not evidence about a group's importance in a living fly.

## Local application lifecycle

FastAPI has a single training worker. Cancellation is checked on each physics
step; a condition variable implements pause/resume. The browser plays real
recorded best-evaluation strokes and labels them as replay, independent of
training rate. WebSocket events are validated with Pydantic on the server and
Zod in React. A latest-event slot bounds progress storage; intermediate updates
may be skipped by slow clients. Replacing a target cancels old work and clears
the display; target/run IDs reject stale events. Training errors are visible.

The application serves on 127.0.0.1:8000, supports one local user/session, and
keeps images in local process memory and ignored local checkpoints. Refreshing
or closing the tab does not cancel an already submitted job; reopen the page
and start a new run to replace it, or stop the server. It is not designed as a
public multi-user service.
