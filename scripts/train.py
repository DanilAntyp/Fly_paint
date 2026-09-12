"""CPU per-image training and paired held-out evaluation."""
import argparse
import json
from pathlib import Path
from flypaint.controller import load_graph
from flypaint.targets import decode_image, fixture, prepare
from flypaint.training import train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path)
    p.add_argument('--image',type=Path)
    p.add_argument('--preset',default='cat',choices=['cat','leaf','circle','square','spiral'])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--generations',type=int,default=10)
    p.add_argument('--population',type=int,default=16)
    p.add_argument('--steps',type=int,default=256)
    p.add_argument('--size',type=int,choices=[64,128,256],default=128)
    p.add_argument('--threshold',type=int,default=35)
    p.add_argument('--output',type=Path,default=Path('runs/readout.npz'))
    args=p.parse_args()
    matrix,meta=load_graph(args.graph)
    image=decode_image(args.image.read_bytes()) if args.image else fixture(args.preset)
    print(meta['label'],meta['nodes'],meta['edges'],flush=True)
    _,result=train(matrix,prepare(image,args.size,args.threshold),args.seed,args.generations,
                   args.population,args.steps,
                   progress=lambda e:print(f"generation {e['generation']}: {e['score']:.4f}",flush=True),checkpoint=args.output)
    print(json.dumps(result['held_out'],indent=2))


if __name__=='__main__':
    main()
