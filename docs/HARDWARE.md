# MacBook Pro M1, 32 GB

The implementation uses CPU NumPy/SciPy float32 sparse operations, one training
worker and a 99-parameter readout at the default graph size. No CUDA, PyTorch,
MPS, full-network backpropagation or paid service is needed. The shell launchers
set OpenBLAS, OpenMP and Accelerate thread limits to one.

Use native ARM64 Python 3.12 and Node 22.12+. The locked requirements were
successfully resolved/downloaded as macOS 14+ ARM64 or universal Python wheels.
In particular the downloaded SciPy wheel targets macOS 14 ARM64. Older macOS
versions have not been checked with this lock. This is a dependency compatibility
check, **not an actual execution test on a MacBook**.

Observed Linux x86_64 values (Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0):

| Task subgraph | Edges | Full CPU step | Inference peak RSS |
|---|---:|---:|---:|
| 512 | 22,235 | 0.193 ms | 47.0 MiB |
| 1024 | 87,149 | 0.265 ms | 48.8 MiB |
| 2048 | 234,095 | 0.382 ms | 53.5 MiB |

These measurements include target/canvas observations, reservoir, pen physics
and rasterization; they exclude browser and optimization. Larger graphs were
benchmarked for inference, not trained or validated for better drawing quality.
The 512-node sparse arrays occupy about 176 KiB. The source-data extractor peaked
at about 1.4 GiB. The application remains bounded independently of source image
resolution after decoding. Browser memory and macOS total unified-memory use
were not measured, and no exact M1 speed is promised.

Three default real-graph training jobs took 16.2–16.7 seconds each on this host.
Timing is observational, not a guaranteed latency budget; a random-graph run
under concurrent extraction took 46.3 seconds. Cooling, background work and
CPU/library choices can change timing on your MacBook substantially.

To measure your actual M1:

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/measure.py --graph data/processed/task-512
bash scripts/train-m1.sh --graph data/processed/task-512 \
  --preset cat --output runs/m1-cat.npz
```

Compare `elapsed_seconds`, process peak RSS and held-out drawing metrics, not
just reservoir multiplication speed. Start with 128-pixel targets, 512 nodes,
16 candidates, 10 generations, 256 steps. Prefer 64 pixels for quick trials;
increasing generations can overfit the two training starts. Keep the initial
8 GB process-memory goal and measure before increasing any budget.
