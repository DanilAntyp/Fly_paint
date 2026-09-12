"""Matched real/random-graph training, independent seeds, held-out evaluation."""
import argparse
import json
from pathlib import Path
import platform
import resource
import time
import numpy as np
from flypaint.controller import load_graph, fingerprint
from flypaint.graph import build_matrix
from flypaint.targets import fixture, prepare
from flypaint.training import train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--generations',type=int,default=10)
    p.add_argument('--output',type=Path,default=Path('runs/experiments.json'))
    args=p.parse_args()
    matrix,meta=load_graph(args.graph)
    n=matrix.shape[0];rng=np.random.default_rng(2026)
    flat=rng.choice(n*n,matrix.nnz,replace=False)
    random=build_matrix(range(n),flat%n,flat//n,np.ones(matrix.nnz))
    random.input_indices=matrix.input_indices;random.output_indices=matrix.output_indices
    target=prepare(fixture('cat'))
    records=[]
    for label,graph in [('real',matrix),('random',random)]:
        for seed in args.seeds:
            path=args.output.parent/f'{label}-cat-{seed}.npz'
            if path.exists():
                result=json.loads(path.with_suffix('.json').read_text())
                if result['graph_sha256']!=fingerprint(graph) or len(result['history'])!=args.generations+1:
                    raise ValueError(f'Incompatible cached checkpoint: {path}')
            else:
                print('training',label,seed,flush=True)
                _,result=train(graph,target,seed,generations=args.generations,checkpoint=path,
                               progress=lambda e:print(e['generation'],round(e['score'],3),flush=True))
            records.append({'graph':label,'seed':seed,'training_initial':result['history'][0]['score'],
                            'training_final':result['history'][-1]['score'],'elapsed_seconds':result['elapsed_seconds'],
                            'held_out':result['held_out'],'checkpoint':str(path)})
    report={'platform':platform.platform(),'graph_nodes':n,'graph_edges':matrix.nnz,
            'random_graph_seed':2026,'random_graph_control':'same nodes, edge count, input/output adapters and encoder seeds; independent uniform random directed edges and normalized unit counts',
            'target':'procedurally authored cat outline; per-image adaptation, not zero-shot',
            'population':16,'generations':args.generations,'episode_steps':256,
            'records':records,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=='Darwin' else 1024)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
