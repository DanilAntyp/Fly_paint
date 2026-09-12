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

Preparation status: URLs verified from the official download page. Actual
source tables could not be downloaded in the preparation environment because
outbound access to storage.googleapis.com timed out. Real-file preprocessing
has therefore not been verified; the next Codex task must run and inspect it.
