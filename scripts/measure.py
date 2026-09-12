"""Isolated graph-size timing including environment/sensory encoding (no learning)."""
import argparse
import json
import platform
import resource
import time
from importlib.metadata import version
import numpy as np
from flypaint.controller import load_graph, Reservoir
from flypaint.environment import DrawingEnv
from flypaint.targets import fixture, prepare
from flypaint.training import rollout

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph');p.add_argument('--nodes',type=int,default=512);args=p.parse_args()
m,meta=load_graph(args.graph,args.nodes)
target=prepare(fixture('cat'));c=Reservoir(m,len(DrawingEnv(target).observation()))
w=np.zeros(c.shape,dtype=np.float32)
rollout(c,w,target,42)
start=time.perf_counter()
for seed in range(10):env=rollout(c,w,target,seed)
elapsed=time.perf_counter()-start
print(json.dumps({'label':meta['label'],'nodes':m.shape[0],'edges':m.nnz,
                  'episodes':10,'steps_per_episode':256,'steps_per_second':2560/elapsed,
                  'step_ms':elapsed/2560*1000,'elapsed_seconds':elapsed,
                  'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=='Darwin' else 1024),
                  'sparse_array_bytes':m.data.nbytes+m.indices.nbytes+m.indptr.nbytes,
                  'platform':platform.platform(),'python':platform.python_version(),
                  'versions':{p:version(p) for p in ('numpy','scipy','pyarrow','pillow')},
                  'scope':'CPU inference with observation, physics and rasterization; no training or browser'},indent=2))
