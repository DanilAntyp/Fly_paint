"""Bounded directed-path extraction from verified Male CNS annotation values.

Candidate pool is an engineering hypothesis, not a proven drawing circuit.
No edge is added. All source edges are streamed, never made dense.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import resource
import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order, connected_components
from flypaint.graph import build_matrix

ROOT = Path(__file__).resolve().parents[1]


def select_paths(adjacency, inputs, outputs, budget):
    """Whole shortest paths only; never truncate the middle of a chosen path."""
    selected = set()
    for start in inputs:
        _, predecessor = breadth_first_order(adjacency, int(start), directed=True)
        for end in outputs:
            if end != start and predecessor[end] < 0:
                continue
            path = {int(end)}
            node = int(end)
            while node != start:
                node = int(predecessor[node])
                path.add(node)
            if len(selected | path) <= budget:
                selected.update(path)
    if not selected:
        raise ValueError('No directed input-output paths in candidate pool.')
    # Add connected recurrent neighbors, ranked by weighted degree.
    degree = np.asarray(adjacency.sum(0)).ravel() + np.asarray(adjacency.sum(1)).ravel()
    undirected = (adjacency + adjacency.T).tocsr()
    while len(selected) < budget:
        neighbors = set()
        for node in sorted(selected):
            neighbors.update(undirected.indices[undirected.indptr[node]:undirected.indptr[node+1]])
        candidates = neighbors-selected
        if not candidates:
            break
        selected.add(min(candidates, key=lambda i: (-degree[i], i)))
    return np.array(sorted(selected), dtype=int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT/'data/raw')
    parser.add_argument('--output', type=Path, default=ROOT/'data/processed/task-512')
    parser.add_argument('--nodes', type=int, default=512)
    args = parser.parse_args()
    if not 64 <= args.nodes <= 2048:
        parser.error('Use 64..2048 nodes for bounded task extraction.')
    start = time.perf_counter()
    manifest = json.loads((ROOT/'configs/data_sources.json').read_text())
    with pa.memory_map(str(args.raw/manifest['files']['annotations']['filename'])) as source:
        table = pa.ipc.open_file(source).read_all().select(['bodyId','superclass','class','type','somaSide','somaNeuromere'])
    records = table.to_pylist()
    original = {str(r['bodyId']): r for r in records if r['bodyId'] is not None}
    if len(original) != len(records):
        raise ValueError('Missing or duplicate annotation IDs.')
    pool = [r for r in records if r['superclass'] in ('visual_projection','descending_neuron','vnc_motor') or r['class']=='CX']
    pool.sort(key=lambda r: r['bodyId'])
    ids = np.array([r['bodyId'] for r in pool], dtype=np.int64)
    ids_arrow = pa.array(ids)
    rows, cols, values = [], [], []
    scanned = 0
    # Only selected columns are decoded and candidate edges retained per record batch.
    with pa.memory_map(str(args.raw/manifest['files']['weights']['filename'])) as source:
        reader = pa.ipc.open_file(source)
        if reader.schema.names != ['body_pre','body_post','weight']:
            raise ValueError('Unexpected weights schema; inspect data before adapting mappings.')
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            scanned += batch.num_rows
            if any(batch.column(c).null_count for c in range(3)):
                raise ValueError('Null source edge.')
            if pc.any(pc.less(batch['weight'], 0)).as_py():
                raise ValueError('Negative source weight.')
            keep = pc.and_(pc.is_in(batch['body_pre'], value_set=ids_arrow),pc.is_in(batch['body_post'],value_set=ids_arrow))
            part = batch.filter(keep)
            cols.append(np.searchsorted(ids,part['body_post'].to_numpy()).astype(np.int32))
            rows.append(np.searchsorted(ids,part['body_pre'].to_numpy()).astype(np.int32))
            values.append(part['weight'].to_numpy().astype(np.float32))
    adjacency = sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(len(ids),len(ids))).tocsr()
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(0)).ravel()+np.asarray(adjacency.sum(1)).ravel()
    def top(predicate, count):
        return sorted([i for i,r in enumerate(pool) if predicate(r)],key=lambda i:(-degree[i],int(ids[i])))[:count]
    inputs = top(lambda r:r['superclass']=='visual_projection',16)
    outputs = top(lambda r:r['superclass']=='descending_neuron',16)+top(lambda r:r['superclass']=='vnc_motor',16)
    selection = select_paths(adjacency,inputs,outputs,args.nodes)
    retained = adjacency[selection][:,selection].tocoo()
    node_ids = [str(ids[i]) for i in selection]
    matrix = build_matrix(node_ids,[node_ids[i] for i in retained.row],[node_ids[i] for i in retained.col],retained.data)
    index = {int(old):i for i,old in enumerate(selection)}
    input_indices = [index[i] for i in inputs if i in index]
    output_indices = [index[i] for i in outputs if i in index]
    outgoing = matrix.T.tocsr()
    reachable = set()
    for source in input_indices:
        order,_ = breadth_first_order(outgoing,source,directed=True)
        reachable.update(order.tolist())
    groups = {}
    for i,old in enumerate(selection):
        record = pool[old]
        label = 'CX' if record['class']=='CX' else record['superclass']
        groups.setdefault(label,[]).append(i)
    annotation_subset = [{k:(str(r[k]) if k=='bodyId' else r[k]) for k in r} for r in (pool[i] for i in selection)]
    crossing_out = float(adjacency[selection].sum()-retained.sum())
    crossing_in = float(adjacency[:,selection].sum()-retained.sum())
    metadata = {'dataset':manifest['dataset'],'mode':'real-derived','nodes':len(selection),'edges':matrix.nnz,
                'selection':'whole directed shortest paths between top weighted-degree verified visual_projection inputs and descending_neuron/vnc_motor outputs; connected recurrent neighbors',
                'candidate_rule':"superclass in visual_projection,descending_neuron,vnc_motor OR class == CX",
                'candidate_nodes':len(pool),'candidate_edges':adjacency.nnz,'source_edge_rows':scanned,
                'input_indices':input_indices,'output_indices':output_indices,'groups':groups,
                'requested_inputs':len(inputs),'retained_inputs':len(input_indices),
                'requested_outputs':len(outputs),'retained_outputs':len(output_indices),
                'reachable_outputs':sum(i in reachable for i in output_indices),
                'weak_components':connected_components(matrix,directed=True,connection='weak')[0],
                'strong_components':connected_components(matrix,directed=True,connection='strong')[0],
                'isolated_nodes':int(np.sum((matrix.getnnz(0)+matrix.getnnz(1))==0)),
                'excluded_annotated_nodes':len(original)-len(selection),
                'omitted_candidate_outgoing_synapses':crossing_out,'omitted_candidate_incoming_synapses':crossing_in,
                'omission_scope':'Crossing weights are within candidate pool only; connections to excluded populations also omitted, not quantified here.',
                'cell_types':dict(Counter(r['type'] or 'untyped' for r in annotation_subset)),
                'sides':dict(Counter(r['somaSide'] or 'unknown' for r in annotation_subset)),
                'normalization':'incoming row sum; nonnegative counts treated as engineering weights',
                'orientation':'W[post,pre]', 'sources':manifest['files'],
                'downloads':json.loads((args.raw/'downloads.json').read_text()),
                'elapsed_seconds':time.perf_counter()-start,'platform':platform.platform(),
                'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=='Darwin' else 1024),
                'limitation':'Candidate functions are hypotheses; this is not a validated biological drawing circuit. Sensory drive and readout are artificial.'}
    args.output.mkdir(parents=True,exist_ok=True)
    sparse.save_npz(args.output/'graph.npz',matrix)
    for name,data in [('node_ids',node_ids),('metadata',metadata),('annotations',annotation_subset)]:
        (args.output/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('groups','cell_types','sources','downloads','input_indices','output_indices')},indent=2))


if __name__=='__main__':
    main()
