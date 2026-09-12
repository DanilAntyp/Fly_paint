"""Small graph utilities. Rows are postsynaptic, columns presynaptic."""
import numpy as np
from scipy import sparse


def build_matrix(node_ids, pre_ids, post_ids, weights):
    ids = [str(value) for value in node_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Node IDs must be nonempty and unique')
    if not (len(pre_ids) == len(post_ids) == len(weights)):
        raise ValueError('Edge arrays have different lengths')
    index = {value: i for i, value in enumerate(ids)}
    values = np.asarray(weights, dtype=np.float32)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Synapse counts must be finite and nonnegative')
    try:
        rows = [index[str(value)] for value in post_ids]
        cols = [index[str(value)] for value in pre_ids]
    except KeyError as error:
        raise ValueError(f'Unknown endpoint: {error}') from error
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(len(ids), len(ids))).tocsr()
    matrix.eliminate_zeros()
    incoming = np.asarray(matrix.sum(axis=1)).ravel()
    if not np.isfinite(incoming).all():
        raise ValueError('Weight sum overflow')
    scale = np.divide(1, incoming, out=np.zeros_like(incoming), where=incoming > 0)
    return (sparse.diags(scale) @ matrix).astype(np.float32).tocsr()


def reservoir_step(matrix, state, drive, leak=0.3, gain=0.9):
    if not 0 < leak <= 1 or not 0 <= gain < 1:
        raise ValueError('Require 0 < leak <= 1 and 0 <= gain < 1')
    return (1 - leak) * state + leak * np.tanh(gain * (matrix @ state) + drive)
