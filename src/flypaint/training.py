"""Seeded bounded elitist evolution; real simulation rollouts, no tracing teacher."""
import json
import time
from pathlib import Path
import numpy as np
from .environment import DrawingEnv
from .controller import Reservoir, fingerprint


class Cancelled(Exception):
    pass


def rollout(controller, weights, target, seed, steps=256, gate=lambda: None, random=False, free=False):
    env = DrawingEnv(target, seed, steps, free=free)
    controller.reset()
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        gate()
        action = rng.uniform([-1,0,0],[1,1,1]) if random else controller.action(env.observation(), weights)
        env.step(action)
    env.activity = float(np.mean(np.abs(controller.state)))
    return env


def train(matrix, target, seed=42, generations=10, population=16, steps=256,
          gate=lambda: None, progress=lambda event: None, checkpoint=None, free=False):
    if not 1 <= generations <= 100 or not 4 <= population <= 32 or not 32 <= steps <= 2048:
        raise ValueError('Training budget outside supported bounds.')
    if free:
        raise ValueError('Free drawing is an untrained exploration mode, not an optimization target.')
    env = DrawingEnv(target, seed, steps)
    controller = Reservoir(matrix, len(env.observation()), seed)
    rng = np.random.default_rng(seed)
    initial = rng.normal(0, .12, controller.shape).astype(np.float32)
    initial[1,-1] = .5
    initial[2,-1] = .6
    # Fixed training seeds shared by all candidates; held-out seeds never select weights.
    train_seeds = [seed, seed+1]
    def evaluate(weights):
        episodes = [rollout(controller, weights, target, s, steps, gate) for s in train_seeds]
        return float(np.mean([e.metrics()['score'] for e in episodes])), episodes[0]
    start = time.perf_counter()
    best = initial.copy()
    best_score, episode = evaluate(best)
    history = []
    def emit(generation, ep):
        event = {'generation': generation, 'generations': generations, 'score': best_score,
                 'metrics': ep.metrics(), 'strokes': ep.strokes,
                 'elapsed_seconds': time.perf_counter()-start,
                 'activity': ep.activity}
        history.append({k:v for k,v in event.items() if k not in ('strokes','activity')})
        progress(event)
    emit(0, episode)
    for generation in range(1, generations+1):
        sigma = .35 * (.95 ** (generation-1))
        # Antithetic perturbations plus elitism retain the measured incumbent.
        candidates = [best]
        for _ in range((population+1)//2):
            noise = rng.normal(0, sigma, best.shape).astype(np.float32)
            candidates.extend([best+noise, best-noise])
        scored = [(evaluate(w), w) for w in candidates[:population]]
        (best_score, episode), best = max(scored, key=lambda item: item[0][0])
        best = best.copy()
        emit(generation, episode)
    held_out = {}
    for label, weights, random in [('untrained', initial, False), ('trained', best, False), ('random_actions', initial, True)]:
        held_out[label] = [rollout(controller, weights, target, s, steps, gate, random).metrics()
                           for s in (seed+1000, seed+1001, seed+1002)]
    result = {'history': history, 'held_out': held_out, 'seed': seed,
              'train_seeds': train_seeds, 'evaluation_seeds': [seed+1000,seed+1001,seed+1002],
              'elapsed_seconds': time.perf_counter()-start, 'parameters': int(best.size),
              'graph_sha256': fingerprint(matrix), 'steps': steps}
    if checkpoint:
        checkpoint = Path(checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(checkpoint, weights=best, initial=initial, encoder=controller.encoder,
                            bias=controller.bias, outputs=controller.outputs, target=target)
        checkpoint.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    return best, result
