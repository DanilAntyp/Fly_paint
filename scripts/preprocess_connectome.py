"""Extract a bounded annotated induced subgraph in two streaming passes.

Selection is by weighted degree, not a validated biological circuit.
Unknown actual schemas require explicit column flags rather than guesses.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow as pa
from scipy import sparse
from flypaint.graph import build_matrix

ROOT = Path(__file__).resolve().parents[1]


def pick(schema, explicit, candidates):
    if explicit:
        if explicit not in schema.names:
            raise ValueError(f'Missing column {explicit}; available: {schema.names}')
        return explicit
    matches = [name for name in candidates if name in schema.names]
    if len(matches) != 1:
        raise ValueError(f'Choose column explicitly; available: {schema.names}')
    return matches[0]


def batches(path, columns):
    with pa.memory_map(str(path), 'r') as source:
        reader = pa.ipc.open_file(source)
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i).select(columns).to_pydict()


def schema(path):
    with pa.memory_map(str(path), 'r') as source:
        return pa.ipc.open_file(source).schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/m1-512')
    parser.add_argument('--nodes', type=int, default=512)
    parser.add_argument('--pre-column')
    parser.add_argument('--post-column')
    parser.add_argument('--weight-column')
    parser.add_argument('--id-column')
    args = parser.parse_args()
    if not 2 <= args.nodes <= 8192:
        parser.error('--nodes must be between 2 and 8192 for this extractor')
    manifest = json.loads((ROOT / 'configs/data_sources.json').read_text())
    ann = args.raw / manifest['files']['annotations']['filename']
    edge = args.raw / manifest['files']['weights']['filename']
    aid = pick(schema(ann), args.id_column, ['body', 'bodyId', 'body_id'])
    es = schema(edge)
    pre = pick(es, args.pre_column, ['body_pre', 'bodyId_pre', 'pre'])
    post = pick(es, args.post_column, ['body_post', 'bodyId_post', 'post'])
    weight = pick(es, args.weight_column, ['weight', 'syn_count', 'count'])
    allowed = {str(v) for batch in batches(ann, [aid]) for v in batch[aid] if v is not None}
    degree = defaultdict(float)
    for batch in batches(edge, [pre, post, weight]):
        for a, b, w in zip(batch[pre], batch[post], batch[weight]):
            if a is None or b is None or w is None:
                raise ValueError('Null edge endpoint or weight')
            a, b, w = str(a), str(b), float(w)
            if not np.isfinite(w) or w < 0:
                raise ValueError('Invalid weight')
            if a in allowed and b in allowed and w > 0:
                degree[a] += w
                degree[b] += w
    ids = sorted(degree, key=lambda key: (-degree[key], key))[:args.nodes]
    selected = set(ids)
    # Aggregate duplicate directed pairs to bound memory by the selected graph.
    retained = defaultdict(float)
    for batch in batches(edge, [pre, post, weight]):
        for a, b, w in zip(batch[pre], batch[post], batch[weight]):
            a, b = str(a), str(b)
            if a in selected and b in selected and w > 0:
                retained[(a, b)] += float(w)
    pairs = list(retained)
    matrix = build_matrix(ids, [a for a, b in pairs], [b for a, b in pairs],
                          list(retained.values()))
    if matrix.nnz == 0:
        raise ValueError('Selection has no edges; revise extraction')
    args.output.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(args.output / 'graph.npz', matrix)
    (args.output / 'node_ids.json').write_text(json.dumps(ids) + '\n')
    record = args.raw / 'downloads.json'
    info = {'dataset': manifest['dataset'], 'mode': 'real-derived',
            'nodes': len(ids), 'edges': matrix.nnz, 'requested_nodes': args.nodes,
            'selection': 'weighted degree among annotated nodes; induced subgraph',
            'normalization': 'incoming row sum; nonnegative engineering baseline',
            'orientation': 'W[post,pre]', 'columns': [aid, pre, post, weight],
            'sources': manifest['files'],
            'downloads': json.loads(record.read_text()) if record.exists() else None,
            'limitation': 'Not a validated sensory-to-motor biological circuit'}
    (args.output / 'metadata.json').write_text(json.dumps(info, indent=2) + '\n')
    print(f'Saved {len(ids)} nodes / {matrix.nnz} edges to {args.output}')


if __name__ == '__main__':
    main()
