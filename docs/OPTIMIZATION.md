# Task-specific network reduction

Owner's decision: use only the network needed for drawing where experiments
support that choice. Do not simulate every neuron merely to quote a large count.

There is no known drawing region supplied by this dataset. Our chosen sensory
encoder, dynamics, output decoder and reward define the task. Removing regions
requires evidence within this artificial model; it does not establish that a
region is unimportant to a living fly.

## Selection before training

1. Inspect actual cell types, side labels and available region annotations.
2. Choose documented candidate input and output populations. Visual signals,
   navigation-related processing and descending/motor pathways are candidates,
   not already validated drawing modules.
3. Preserve directed paths between the selected endpoints, including relevant
   recurrent neighbors. Begin with a bounded subgraph and document truncated
   inputs/outputs. Do not assume functions can be cleanly separated by region.
4. Report retained nodes/edges, isolated nodes, reachable outputs and cell types.
5. Use the supplied high-degree extractor only as a generic computational
   baseline; do not call its output a biologically curated sensorimotor circuit.

## Ablation after a working baseline

Freeze a trained checkpoint. On exactly the same held-out targets and seeds,
disable one annotated group at a time; compare target coverage, off-target ink,
task score, step time and memory to the intact controller. Include a no-ablation
control. Repeat across several independently trained checkpoints.

Zero ablated neuron state and outputs on every timestep. In the immediate
ablation experiment keep the remaining weights and normalization unchanged.
Otherwise re-normalization itself can cause differences. Distinguish removal
without retraining from pruning followed by retraining.

Before evaluating, choose an acceptable quality loss, for example 5% relative
on a positive normalized score, and report absolute differences too. Do not
accept a single noisy run as evidence. Retest combinations: two groups may
compensate for each other and joint removal can fail despite individual results.

Once justified, build a physically smaller sparse matrix and input/readout
adapters, optionally retrain the readout, and re-evaluate. Masking neurons alone
does not automatically reduce allocated memory or sparse operation cost.

Preserve the original dataset and checkpoints. Save every pruning decision,
node-ID mapping, evaluation seed and result. Label this as task-specific
engineering optimization. Neither art quality nor biological fidelity is
guaranteed to improve by selecting a real connectome over a random graph.
