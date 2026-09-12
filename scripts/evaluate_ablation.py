"""Paired intact/masked/combinations/random-graph evaluation of frozen readouts.

Masks never renormalize W. Physical pruning is deliberately a separate operation.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import platform
import resource
import time
import numpy as np
from flypaint.controller import load_graph, Reservoir, fingerprint
from flypaint.environment import DrawingEnv
from flypaint.targets import fixture, prepare
from flypaint.training import rollout


def evaluate(graph, checkpoint):
    matrix,meta=load_graph(graph)
    record=json.loads(checkpoint.with_suffix('.json').read_text())
    if fingerprint(matrix)!=record['graph_sha256']:
        raise ValueError('Checkpoint belongs to a different graph.')
    data=np.load(checkpoint,allow_pickle=False)
    target=data['target']
    controller=Reservoir(matrix,len(DrawingEnv(target).observation()),record['seed'])
    controller.encoder=data['encoder'];controller.bias=data['bias'];controller.outputs=data['outputs']
    groups=meta.get('groups',{})
    masks={'intact':[]}
    masks.update(groups)
    for a,b in combinations(groups,2):
        masks[f'{a}+{b}']=sorted(set(groups[a])|set(groups[b]))
    rows=[]
    for label,indices in masks.items():
        controller.mask.fill(1);controller.mask[indices]=0
        for preset,case_target in [('adapted_image',target),('held_out_leaf',prepare(fixture('leaf'),len(target))),('held_out_circle',prepare(fixture('circle'),len(target)))]:
            for seed in record['evaluation_seeds']:
                start=time.perf_counter()
                env=rollout(controller,data['weights'],case_target,seed,record['steps'])
                rows.append({'mask':label,'disabled_nodes':len(indices),'target':preset,'seed':seed,
                             **env.metrics(),'step_ms':1000*(time.perf_counter()-start)/record['steps']})
    # Equal node AND edge count random wiring control; same endpoint adapters and weights.
    rng=np.random.default_rng(record['seed']+9000)
    count=matrix.nnz;n=matrix.shape[0]
    flat=rng.choice(n*n,count,replace=False)
    from flypaint.graph import build_matrix
    random_matrix=build_matrix(range(n),flat%n,flat//n,np.ones(count))
    controller.matrix=random_matrix;controller.mask.fill(1)
    for label,weights,random_actions in [('random_graph_frozen_readout',data['weights'],False),('untrained',data['initial'],False),('random_actions',data['initial'],True)]:
        controller.matrix=random_matrix if label.startswith('random_graph') else matrix
        for seed in record['evaluation_seeds']:
            start=time.perf_counter()
            env=rollout(controller,weights,target,seed,record['steps'],random=random_actions)
            rows.append({'mask':label,'disabled_nodes':0,'target':'adapted_image','seed':seed,
                         **env.metrics(),'step_ms':1000*(time.perf_counter()-start)/record['steps']})
    return {'checkpoint':str(checkpoint),'training_seed':record['seed'],'rows':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,default=Path('runs/ablation.json'))
    args=p.parse_args()
    report={'acceptable_relative_f1_loss':.05,'comparison':'paired target and seed; frozen W, normalization, encoder and readout',
            'memory_note':'Masks retain matrix allocation. Peak RSS is process-wide, not group-specific.',
            'runs':[evaluate(args.graph,path) for path in args.checkpoints],
            'platform':platform.platform(),
            'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=='Darwin' else 1024)}
    decisions={}
    for run in report['runs']:
        intact={(r['target'],r['seed']):r for r in run['rows'] if r['mask']=='intact'}
        for row in run['rows']:
            if row['mask'] in ('intact','random_graph_frozen_readout','untrained','random_actions'):continue
            base=intact[row['target'],row['seed']]
            row['f1_absolute_delta']=row['f1']-base['f1']
            row['f1_relative_loss']=(base['f1']-row['f1'])/base['f1'] if base['f1']>0 else None
            decisions.setdefault(row['mask'],[]).append(row['f1_relative_loss'])
    report['pruning_decisions']={k:{'all_paired_runs_within_5_percent':all(x is not None and x<=.05 for x in v),
                                         'independent_checkpoints':len(report['runs']),
                                         'action':'No automatic pruning. Low drawing quality and small evaluation suite do not justify removal.'} for k,v in decisions.items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
