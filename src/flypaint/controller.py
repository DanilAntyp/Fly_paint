"""Frozen CPU sparse reservoir and small, evolvable linear readout."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from .graph import build_matrix, reservoir_step


def load_graph(path=None, nodes=512, seed=42):
    if path:
        path = Path(path)
        matrix = sparse.load_npz(path / 'graph.npz').astype(np.float32).tocsr()
        metadata = json.loads((path / 'metadata.json').read_text())
        ids = json.loads((path / 'node_ids.json').read_text())
        if matrix.shape != (len(ids), len(ids)) or len(ids) != len(set(ids)):
            raise ValueError('Graph and node IDs do not agree.')
        if not all(isinstance(v, str) for v in ids):
            raise ValueError('Node IDs must remain strings.')
        if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
            raise ValueError('Graph weights must be finite and nonnegative.')
        if metadata.get('mode') != 'real-derived' or metadata.get('dataset') != 'male-cns:v1.0':
            raise ValueError('Expected explicit Male CNS v1.0 provenance.')
        if matrix.shape[0] > 8192 or matrix.nnz == 0:
            raise ValueError('Graph must have edges and at most 8192 nodes.')
        metadata = {**metadata, 'label': 'Male CNS v1.0-derived subgraph'}
    else:
        rng = np.random.default_rng(seed)
        pre = np.repeat(np.arange(nodes), 8)
        post = rng.integers(0, nodes, len(pre))
        matrix = build_matrix(range(nodes), pre, post, np.ones(len(pre)))
        ids = [f'synthetic-{i}' for i in range(nodes)]
        metadata = {'label': 'Synthetic demo graph', 'mode': 'synthetic', 'seed': seed}
    matrix.input_indices = metadata.get('input_indices')
    matrix.output_indices = metadata.get('output_indices')
    return matrix, {**metadata, 'nodes': matrix.shape[0], 'edges': matrix.nnz, 'node_ids': ids}


def fingerprint(matrix):
    h = hashlib.sha256()
    for array in (matrix.data, matrix.indices, matrix.indptr):
        h.update(array.tobytes())
    return h.hexdigest()


class Reservoir:
    def __init__(self, matrix, observations, seed=42, readout_size=32):
        self.matrix = matrix
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.encoder = rng.normal(0, .28, (matrix.shape[0], observations)).astype(np.float32)
        inputs = getattr(matrix, 'input_indices', None)
        if inputs is not None:
            disabled = np.ones(matrix.shape[0], dtype=bool)
            disabled[inputs] = False
            self.encoder[disabled] = 0
        self.bias = rng.normal(0, .1, matrix.shape[0]).astype(np.float32)
        self.outputs = np.sort(rng.choice(matrix.shape[0], min(readout_size, matrix.shape[0]), replace=False))
        outputs = getattr(matrix, 'output_indices', None)
        if outputs is not None:
            if not outputs:
                raise ValueError('No retained output neurons.')
            self.outputs = np.array(outputs, dtype=int)
        self.state = np.zeros(matrix.shape[0], dtype=np.float32)
        self.mask = np.ones(matrix.shape[0], dtype=np.float32)
        self.shape = (3, len(self.outputs)+1)

    def reset(self):
        self.state.fill(0)

    def action(self, observation, weights):
        self.state *= self.mask
        drive = self.encoder @ observation + self.bias
        self.state = reservoir_step(self.matrix, self.state, drive) * self.mask
        features = np.append(self.state[self.outputs], np.float32(1))
        raw = weights @ features
        return np.array([np.tanh(raw[0]), (np.tanh(raw[1])+1)/2, (np.tanh(raw[2])+1)/2])
