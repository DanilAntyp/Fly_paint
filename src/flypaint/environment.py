"""Fixed-step pen physics; every mark is a bounded agent movement."""
import numpy as np
from PIL import Image, ImageDraw
from .targets import PAPER


class DrawingEnv:
    def __init__(self, target, seed=42, steps=256, free=False):
        self.target = np.asarray(target, dtype=bool)
        self.size = len(target)
        if self.target.shape != (self.size, self.size):
            raise ValueError('Target must be square.')
        self.free = free
        self.total = int(self.target.sum())
        if self.total < 4 and not free:
            raise ValueError('This target is blank or has too little detail. Lower the threshold or choose another image.')
        self.budget = steps
        self.ink = np.zeros_like(self.target)
        rng = np.random.default_rng(seed)
        # Independent of target: never start from a selected foreground pixel.
        self.x, self.y = rng.uniform(.35, .65, 2) * (self.size - 1)
        self.heading = float(rng.uniform(-np.pi, np.pi))
        self.t = self.repeated = self.stalled = 0
        self.travel = 0.
        self.strokes = []
        block = self.size // 8
        self.global_target = self.target.reshape(8, block, 8, block).mean((1, 3)).ravel()

    def observation(self):
        residual = self.target & ~self.ink
        # Egocentric patches at two scales; no target centroid, route or desired action.
        offsets = np.arange(-2, 3)
        xx, yy = np.meshgrid(offsets, offsets)
        patches = []
        c, s = np.cos(self.heading), np.sin(self.heading)
        for spacing in (self.size/64, self.size/16):
            x = np.rint(self.x + spacing*(xx*c-yy*s)).astype(int).clip(0, self.size-1)
            y = np.rint(self.y + spacing*(xx*s+yy*c)).astype(int).clip(0, self.size-1)
            patches.extend((residual[y, x].ravel(), self.ink[y, x].ravel()))
        block = self.size // 8
        global_residual = residual.reshape(8, block, 8, block).mean((1, 3)).ravel()
        pose = [self.x/(self.size-1)*2-1, self.y/(self.size-1)*2-1, c, s, 1-self.t/self.budget]
        return np.concatenate([pose, self.global_target, global_residual, *patches]).astype(np.float32)

    def step(self, action):
        if self.t >= self.budget:
            raise ValueError('Episode already finished.')
        a = np.asarray(action, dtype=float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError('Expected three finite actions.')
        turn, speed, pressure = np.clip(a, [-1, 0, 0], [1, 1, 1])
        self.heading = float((self.heading + turn * .55 + np.pi) % (2*np.pi) - np.pi)
        distance = speed * self.size / 40
        old_x, old_y = self.x, self.y
        self.x = float(np.clip(self.x + np.cos(self.heading)*distance, 0, self.size-1))
        self.y = float(np.clip(self.y + np.sin(self.heading)*distance, 0, self.size-1))
        moved = float(np.hypot(self.x-old_x, self.y-old_y))
        self.travel += moved/self.size
        if moved < self.size/1000 and (self.free or (self.target & ~self.ink).any()):
            self.stalled += 1
        if pressure > .45 and moved > 1e-6:
            samples = max(2, int(np.ceil(moved*2))+1)
            x = np.rint(np.linspace(old_x, self.x, samples)).astype(int)
            y = np.rint(np.linspace(old_y, self.y, samples)).astype(int)
            indices = np.unique(y*self.size+x)
            self.repeated += int(self.ink.ravel()[indices].sum())
            self.ink.ravel()[indices] = True
        stroke = [old_x/self.size, old_y/self.size, self.x/self.size, self.y/self.size,
                  float(pressure if moved > 1e-6 else 0), self.heading]
        self.strokes.append(stroke)
        self.t += 1
        return stroke

    def metrics(self):
        hit = int((self.ink & self.target).sum())
        off = int((self.ink & ~self.target).sum())
        precision = hit / max(1, hit+off)
        recall = hit / max(1, self.total)
        f1 = 2*precision*recall/max(1e-12, precision+recall)
        score = (100*f1 + 20*recall - 10*off/max(1,self.total)
                 - .02*self.repeated - .03*self.travel - .05*self.stalled)
        return dict(score=float(score), precision=precision, recall=recall, f1=f1,
                    off_target=off, repeated=self.repeated, steps=self.t)


def render_strokes(strokes, size=768, target_size=128):
    image = Image.new('RGB', (size, size), PAPER)
    draw = ImageDraw.Draw(image)
    for x0,y0,x1,y1,p,_ in strokes:
        if p > .45:
            draw.line((x0*size,y0*size,x1*size,y1*size), fill=(25,28,23), width=max(1,round(size/target_size)))
    return image
