# Fly_paint · Fly Gogh

**A fly goes to art school.** An experimental drawing agent built around a
small network derived from real fruit-fly neural connectivity. Upload an image
and watch the fly attempt to redraw it stroke by stroke on a canvas.

Status: **Codex preparation kit**, not a finished app or trained model.
Includes project instructions, official data manifest, download/schema tools,
streaming subgraph extraction, a sparse controller primitive, CPU benchmark
and tests. The next task implements image upload, drawing, training and the
interface; image copying is now a first-release requirement.

## Start locally (MacBook M1, 32 GB)

Use a native ARM64 Python 3.11+ installation (3.12 recommended).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/benchmark.py --nodes 512 --steps 2000
```

The benchmark's default graph is synthetic and measures reservoir operations
only. It does not train a fly or predict total training time.

## Get the real data

```bash
python scripts/download_data.py
python scripts/inspect_data.py data/raw/*.feather
python scripts/preprocess_connectome.py --nodes 512
python scripts/benchmark.py --graph data/processed/m1-512/graph.npz
```

The first download is roughly 1.1 GB; allow several GB of working space.
See [official download page](https://male-cns.janelia.org/download/).
No neuPrint account is needed for these public bulk files. If a schema differs,
the extractor stops and shows available columns; supply its explicit column
flags after inspection. It preserves real edges among selected annotated IDs.
Its degree-based selection is a computational baseline, not a validated
biological sensory-to-motor circuit.

Data and model artifacts are ignored by Git. Re-download in each new execution
environment or use an explicitly configured persistent local cache.

## Give this to Codex

Choose this repository and start a task with:

> Read AGENTS.md and docs/CODEX_START.md. Implement the image-copying MVP:
> upload a picture, preview a simplified sketch target, and watch the fly draw
> it through learned movement. Include this in the first working release.
> Install dependencies, download and validate the necessary Male CNS data,
> and start with the MacBook M1 / 32 GB profile. Follow docs/OPTIMIZATION.md
> for a task-related subgraph and group-ablation experiments. Run meaningful
> checks and report real results. Do not stop at a plan or claim synthetic
> data is the biological connectome.

The files do not automatically start a background coding task. In a restricted
execution environment, allow the documented public data and package sources
through that environment's supported network settings, or download locally.

## Design and limits

Warm paper, black ink, dark laboratory panels and an expressive fly with a pen.
The main flow is **upload picture -> preview sketch target -> start the fly ->
watch strokes -> export the drawing**. Begin with black-ink sketches and
adjustable detail, including portraits or objects. Original image, simplified
target and actual drawing remain visibly distinct. Circles and spirals are
training/debug presets.

The initial version can train its small readout for each uploaded image;
instant copying of previously unseen images is a later generalization goal.
Full-color painting can follow sketch mode. Nothing should paste the reference
onto the canvas or animate a predetermined tracing path as if it were learned.
There is no biological "drawing region" encoded in the dataset. This project
adds artificial dynamics, observations, actions and learning. We will select
candidate pathways and test group removal on held-out tasks before pruning.

- [Project guidance](AGENTS.md)
- [First coding task](docs/CODEX_START.md)
- [Network selection and ablation](docs/OPTIMIZATION.md)
- [MacBook resource notes](docs/HARDWARE.md)
- [Dataset and attribution](docs/DATA.md)

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

Real data download and preprocessing were blocked during preparation by a
network restriction; do not treat this pipeline as validated on the full source
tables yet. No training results or MacBook timings have been measured.

The underlying Male CNS dataset is CC BY; see the source page for attribution
and terms. A code license has not yet been selected by the repository owner.
