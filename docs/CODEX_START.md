# First implementation task

Read AGENTS.md and implement a working Fly_paint MVP. The owner has authorized
downloading necessary public data and installing project dependencies within
the task environment. Follow environment network/access restrictions.

1. Inspect the repository and run existing graph tests. This repository is a
   preparation kit; the drawing app and trainer are not implemented yet.
2. Create a Python virtual environment, install the package, and run the
   synthetic CPU benchmark. Read configs/macbook-m1.json as initial settings.
3. Execute scripts/download_data.py for annotations and connectivity. Install
   PyArrow before downloading. Run scripts/inspect_data.py on both files.
   Adapt explicit column mappings to the actual source schema if required.
   Never guess undocumented biological attributes.
4. Extract a 512-node baseline graph, inspect its connected components and
   input-to-output reachability. Record exclusions, counts and normalization.
   The included degree-based extractor is only a smoke-test baseline. Implement
   task-related extraction using verified visual/navigation/descending/motor
   annotations before treating this as a sensorimotor experiment.
5. Implement local PNG/JPEG/WebP upload and sketch-target preprocessing first,
   following AGENTS.md's image-to-drawing requirements. Include original/target
   previews, aspect-ratio preservation, upload limits and detail controls.
   Use a 128 x 128 target initially. Build a deterministic drawing environment
   whose observations contain target and canvas features. Test stationary,
   repeated-ink, off-target, ideal-path and blank-target behaviors.
6. Implement bounded evolutionary optimization of a small readout on the fixed
   reservoir. Start with 16 candidates, 10 generations and 256-step episodes.
   Record untrained, trained and random-action scores on held-out seeds. These
   budgets are initial trials; success is not guaranteed. Support per-image
   readout optimization for the uploaded target and label it honestly. Evaluate
   adaptation on an uploaded non-geometric image; shapes alone do not satisfy
   the MVP. Never substitute a computed tracing path for the agent's actions.
7. Build the React/TypeScript canvas UI and a local Python API. Wire real
   image picker/dropzone, target previews, live strokes, progress,
   pause/cancellation, reset, speed and PNG export. Replacing the image must
   cancel the old run and prevent stale events. Add install/run
   scripts and lockfiles based on versions actually used. Keep keys unnecessary.
8. Follow docs/OPTIMIZATION.md for task-related subgraphs and ablation. Report
   actual quality/performance tradeoffs. Avoid full-network backpropagation.
9. Test the complete local flow. Update README with exact commands, observed
   results and limitations. Commit the implementation and open a PR if working
   from a task branch. Do not claim completion based on the scaffold alone.

If public downloads are blocked, keep a clearly marked synthetic mode working,
record the precise blocker and commands needed in a permitted environment.
Do not bypass access restrictions or claim the real dataset was used.

Downloading is data acquisition, not training. Uploaded target images and the simulator
generate drawing experience; there is no pre-existing fly drawing skill in
these tables. A separate pretrained checkpoint is not required to start.
