"""Measure sparse reservoir steps only, NOT complete training speed."""
import argparse
import json
import platform
import resource
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from flypaint.graph import build_matrix, reservoir_step

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--graph', type=Path)
parser.add_argument('--nodes', type=int, default=512)
parser.add_argument('--steps', type=int, default=2000)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
if args.nodes < 2 or args.steps < 1:
    parser.error('Require nodes >= 2 and steps >= 1')
rng = np.random.default_rng(args.seed)
if args.graph:
    matrix = sparse.load_npz(args.graph).astype(np.float32)
    mode = 'provided graph; check companion metadata for provenance'
else:
    n = args.nodes
    pre = np.repeat(np.arange(n), 8)
    post = rng.integers(0, n, len(pre))
    matrix = build_matrix(range(n), pre, post, np.ones(len(pre)))
    mode = 'SYNTHETIC DEMO GRAPH'
state = np.zeros(matrix.shape[0], dtype=np.float32)
drive = rng.normal(0, 0.1, state.shape).astype(np.float32)
for _ in range(100):
    state = reservoir_step(matrix, state, drive)
start = time.perf_counter()
for _ in range(args.steps):
    state = reservoir_step(matrix, state, drive)
elapsed = time.perf_counter() - start
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mib = peak / (1024**2 if platform.system() == 'Darwin' else 1024)
print(json.dumps({'mode': mode, 'platform': platform.platform(),
                  'nodes': matrix.shape[0], 'edges': matrix.nnz,
                  'steps_per_second': args.steps / elapsed,
                  'elapsed_seconds': elapsed, 'process_peak_rss_mib': peak_mib,
                  'finite': bool(np.isfinite(state).all()),
                  'scope': 'CPU reservoir only; excludes environment, rewards, training and UI'},
                 indent=2))
