# Source data and provenance

Official downloads: https://male-cns.janelia.org/download/
Project: https://www.janelia.org/project-team/flyem/male-cns-connectome
Explorer: https://codex.flywire.ai/

Use Male CNS v1.0, also exposed as neuPrint `male-cns:v1.0`.
The manifest pins official URLs for annotations (~13 MB) and weighted
connectivity (~1.1 GB). Neurotransmitter predictions (~42 MB) are optional.
Counts and file sizes are approximate; check the current source and file schema.
Full connectivity tables contain segments beyond the curated neuron list;
filter against annotations instead of equating all segment IDs to neurons.

The downloader writes local SHA-256 hashes, byte counts, URLs and retrieval times
to data/raw/downloads.json. Local hashes detect later file changes; without
publisher-supplied hashes they are not independent authenticity verification.
It validates Arrow metadata and expected transfer length before accepting files.
Interrupted downloads restart; existing validated files are reused.

This is structural input for a controller. Training experience is generated in
the drawing environment; these are not image training pairs or model weights.
Raw microscopy volumes, all synapse positions and a full Neo4j database are
unnecessary for the initial scope. Do not commit them or raw tables to Git.

The dataset is published under CC BY (follow the source page's license link for
the exact version). Attribute FlyEM at HHMI Janelia, University of Cambridge,
MRC Laboratory of Molecular Biology and Google Research. Cite the Male CNS
publication linked from the project page. Clearly describe extraction and
normalization as modifications. Dataset licensing does not license this code.

## Acquisition verified on 2026-09-12

Both tables were downloaded in this implementation environment and validated
with PyArrow 25.0.1. Exact transfer sizes, retrieval times, source URLs and
SHA-256 hashes are committed in [DOWNLOADS.json](DOWNLOADS.json). Checksums are
locally recorded integrity checks, not publisher-provided authenticity proofs.

Observed annotations: `bodyId: int64`, `superclass`, `class`, `type`,
`somaSide`, `somaNeuromere` and other fields. Four record batches, 211,577
annotation rows; these must not all be equated to verified neurons.
Observed connectivity: `body_pre: int64`, `body_post: int64`, `weight: int64`,
2,318 record batches and 151,856,684 segment-pair rows. The two downloaded files
are 14,483,314 and 1,051,241,946 bytes respectively.

Verified candidate labels used by the task extractor are
`visual_projection`, `descending_neuron`, `vnc_motor` (superclass) and `CX`
(class). Type/side/neuromere values are kept without inventing biological
attributes. Their task usefulness remains an engineering hypothesis.
The real 512-node task graph and generic high-degree graph were both extracted.
See [EXTRACTION.json](EXTRACTION.json) and [METHODOLOGY.md](METHODOLOGY.md).

The source download page links to [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The original source data and generated matrices stay in ignored local folders.
No raw tables, model checkpoints, credentials or microscopy data are in Git.
