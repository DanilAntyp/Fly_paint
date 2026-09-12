# MacBook Pro M1 / 32 GB

Initial design judgment: a modest sparse fixed reservoir and small trainable
readout are a practical starting point. No timing on the owner's laptop has
been measured. Do not quote training completion times before a local trial.

Start with 512 neurons and benchmark 1024/2048 only after the small model works.
Use CPU and float32, one environment, small batches and a bounded training
budget. The machine's unified RAM is shared with the system and GPU.

A dense 166,700 x 166,700 float32 matrix alone occupies about 111 GB (103.5 GiB),
before states, gradients and optimizer buffers. Sparse storage dramatically
changes that calculation, but full-size simulation/training still needs a
separate memory and throughput evaluation. Source file size is not peak RAM.

PyTorch can use Apple GPU through MPS, but the exact selected operators must be
tested for support and speed. Do not assume sparse SciPy CPU operations can
simply be moved to MPS. CPU reservoir + small readout is the initial path.

The provided benchmark measures only reservoir updates, not rendering, reward
rasterization, episodes or optimizer work. Its Linux results cannot predict
M1 performance. Add a bounded end-to-end trial and report completed environment
steps per second, evaluation quality and peak process memory.

Sources: https://docs.pytorch.org/docs/stable/notes/mps.html and
https://docs.pytorch.org/docs/stable/sparse.html .
