"""Single-user local API. One bounded worker; latest event only; cancellable runs."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path
import threading
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from flypaint.controller import load_graph, Reservoir
from flypaint.environment import DrawingEnv, render_strokes
from flypaint.targets import MAX_BYTES, decode_image, fixture, prepare, png_url, raster_image
from flypaint.training import Cancelled, train, rollout

ROOT = Path(__file__).resolve().parents[2]


class Start(BaseModel):
    model_config = ConfigDict(extra='forbid')
    target_id: str
    seed: int = Field(default=42, ge=0, le=2**31-2000)
    generations: int = Field(default=10, ge=1, le=100)
    population: int = Field(default=16, ge=4, le=32)
    steps: int = Field(default=256, ge=32, le=2048)


class Progress(BaseModel):
    model_config = ConfigDict(extra='forbid')
    run_id: str
    target_id: str
    revision: int
    status: Literal['running', 'paused', 'done', 'cancelled', 'error']
    generation: int = 0
    generations: int = 10
    score: float = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    strokes: list[list[float]] = Field(default_factory=list)
    activity: float = 0
    elapsed_seconds: float = 0
    held_out: dict | None = None
    error: str | None = None


class Job:
    def __init__(self, target_id, generations):
        self.id = uuid.uuid4().hex
        self.condition = threading.Condition()
        self.cancelled = False
        self.paused = False
        self.event = Progress(run_id=self.id, target_id=target_id, revision=0,
                              status='running', generations=generations)

    def publish(self, **update):
        with self.condition:
            values = self.event.model_dump()
            values.update(update)
            values['revision'] += 1
            self.event = Progress.model_validate(values)

    def control(self, command):
        with self.condition:
            if self.event.status in ('done','error','cancelled'):
                return
            if command == 'cancel':
                self.cancelled = True
            self.paused = command == 'pause'
            self.publish(status='cancelled' if self.cancelled else ('paused' if self.paused else 'running'))
            self.condition.notify_all()

    def gate(self):
        with self.condition:
            while self.paused and not self.cancelled:
                self.condition.wait(.1)
            if self.cancelled:
                raise Cancelled()


def create_app(graph_path=None):
    matrix, graph = load_graph(graph_path)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fly-training')
    submissions = asyncio.Lock()
    state = {'image': None, 'target': None, 'target_id': None, 'job': None,
             'future': None, 'free': False}

    @asynccontextmanager
    async def lifespan(app):
        yield
        if state['job']:
            state['job'].control('cancel')
        pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title='Fly Gogh local trainer', lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1','localhost','testserver','[::1]'])

    @app.middleware('http')
    async def local_origin(request, call_next):
        origin = request.headers.get('origin')
        if origin and origin not in {'http://localhost:8000','http://127.0.0.1:8000',
                                     'http://localhost:5173','http://127.0.0.1:5173'}:
            return Response('Use the local Fly Gogh page.', status_code=403)
        return await call_next(request)

    def cancel():
        if state['job']:
            state['job'].control('cancel')
        if state['future']:
            state['future'].cancel()

    def replace(image, size, threshold, free=False):
        try:
            target = prepare(image, size, threshold)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        cancel()
        target_id = uuid.uuid4().hex
        state.update(image=image, target=target, target_id=target_id, job=None, free=free)
        preview = image.copy()
        preview.thumbnail((512,512))
        blank = int(target.sum()) < 4
        return {'target_id': target_id, 'original': png_url(preview),
                'target': png_url(raster_image(target)), 'size': size,
                'foreground': int(target.sum()), 'blank': blank and not free, 'free': free,
                'message': 'Target is nearly blank. Lower the threshold or choose another image.' if blank and not free else None}

    @app.get('/api/health')
    def health():
        return {'status':'ok', 'graph': {**{k:v for k,v in graph.items() if k != 'node_ids'},
                           'readout_parameters':3*(len(matrix.output_indices) if matrix.output_indices is not None else min(32,matrix.shape[0]))+3},
                'device':'cpu', 'training':'per-image readout evolution', 'max_workers':1}

    @app.post('/api/upload')
    async def upload(request: Request, size: int=128, threshold: int=35):
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise HTTPException(413, 'Image exceeds the 10 MB limit.')
        try:
            image = decode_image(bytes(data))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return replace(image, size, threshold)

    @app.post('/api/preset/{name}')
    async def preset(name: str, size: int=128, threshold: int=35):
        try:
            image = fixture(name)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return replace(image, size, threshold, name == 'free')

    @app.post('/api/target')
    async def target(size: int=128, threshold: int=35):
        if state['image'] is None:
            raise HTTPException(409, 'Upload an image first.')
        return replace(state['image'], size, threshold, state['free'])

    def worker(job, target, options, free):
        try:
            if free:
                import numpy as np
                controller = Reservoir(matrix, len(DrawingEnv(target, free=True).observation()), options.seed)
                weights = np.random.default_rng(options.seed).normal(0,.2,controller.shape)
                env = rollout(controller, weights, target, options.seed, options.steps, job.gate, free=True)
                job.gate()
                job.publish(status='done', strokes=env.strokes, metrics=env.metrics(), generations=0)
            else:
                _, result = train(matrix, target, options.seed, options.generations,
                                  options.population, options.steps, gate=job.gate,
                                  progress=lambda event: job.publish(**event),
                                  checkpoint=ROOT/'runs'/f'{job.id}.npz')
                job.gate()
                job.publish(status='done', held_out=result['held_out'], elapsed_seconds=result['elapsed_seconds'])
        except Cancelled:
            job.publish(status='cancelled')
        except Exception as error:
            job.publish(status='error', error=str(error))

    @app.post('/api/start')
    async def start(options: Start):
        async with submissions:
            if state['target_id'] != options.target_id or state['target'] is None:
                raise HTTPException(409, 'Target changed; use its current preview.')
            if not state['free'] and state['target'].sum() < 4:
                raise HTTPException(422, 'Cannot train a blank target.')
            cancel()
            # Wait for the single previous worker to exit before submitting again.
            # There is no accumulating executor work queue, even under repeated starts.
            previous = state['future']
            if previous is not None and not previous.done():
                try:
                    await asyncio.wrap_future(previous)
                except asyncio.CancelledError:
                    if not previous.cancelled():
                        raise
            if state['target_id'] != options.target_id:
                raise HTTPException(409, 'Target changed while cancelling the previous run.')
            job = Job(options.target_id, options.generations)
            state['job'] = job
            state['future'] = pool.submit(worker, job, state['target'].copy(), options, state['free'])
            return job.event.model_dump()

    def current(run_id):
        job = state['job']
        if job is None or job.id != run_id:
            raise HTTPException(404, 'Run expired or replaced.')
        return job

    @app.post('/api/runs/{run_id}/{command}')
    async def control(run_id: str, command: Literal['pause','resume','cancel']):
        job = current(run_id)
        job.control(command)
        return job.event.model_dump()

    @app.get('/api/runs/{run_id}')
    def status(run_id: str):
        return current(run_id).event.model_dump()

    @app.get('/api/runs/{run_id}/drawing.png')
    def export(run_id: str):
        job = current(run_id)
        stream = BytesIO()
        render_strokes(job.event.strokes, target_size=len(state['target'])).save(stream, 'PNG')
        return Response(stream.getvalue(), media_type='image/png',
                        headers={'Content-Disposition':'attachment; filename="fly-gogh.png"'})

    @app.websocket('/api/events/{run_id}')
    async def events(ws: WebSocket, run_id: str):
        if ws.headers.get('origin') not in {None,'http://localhost:8000','http://127.0.0.1:8000',
                                          'http://localhost:5173','http://127.0.0.1:5173'}:
            await ws.close(code=1008)
            return
        job = state['job']
        if job is None or job.id != run_id:
            await ws.close(code=1008)
            return
        await ws.accept()
        revision = -1
        try:
            while True:
                event = job.event
                if event.revision != revision:
                    await ws.send_json(event.model_dump())
                    revision = event.revision
                if event.status in ('done','cancelled','error'):
                    await ws.close()
                    return
                await asyncio.sleep(.1)
        except WebSocketDisconnect:
            pass

    dist = ROOT/'apps/web/dist'
    if dist.exists():
        app.mount('/', StaticFiles(directory=dist, html=True), name='web')
    return app


def app_factory():
    return create_app(os.environ.get('FLYPAINT_GRAPH') or None)
