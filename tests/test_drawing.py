from io import BytesIO
import threading
import unittest
import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from flypaint.targets import decode_image, prepare, fixture, MAX_BYTES
from flypaint.environment import DrawingEnv, render_strokes
from flypaint.controller import load_graph, Reservoir
from flypaint.training import train, rollout, Cancelled
from services.trainer.app import create_app, Job


def encoded(image, format='PNG', **kwargs):
    stream=BytesIO();image.save(stream,format,**kwargs);return stream.getvalue()


class TargetTests(unittest.TestCase):
    def test_formats_corruption_and_limits(self):
        for fmt in ('PNG','JPEG','WEBP'):
            self.assertEqual(decode_image(encoded(fixture('cat'),fmt)).size,(128,128))
        for data in (b'not an image',b'x'*(MAX_BYTES+1),encoded(fixture('cat'),'GIF')):
            with self.assertRaises(ValueError):decode_image(data)
        with self.assertRaises(ValueError):
            decode_image(encoded(Image.new('RGB',(4001,4000))))

    def test_orientation_transparency_and_aspect(self):
        image=Image.new('RGB',(80,40),'black');exif=Image.Exif();exif[274]=6
        self.assertEqual(decode_image(encoded(image,'JPEG',exif=exif)).size,(40,80))
        transparent=Image.new('RGBA',(100,20),(0,0,0,0))
        self.assertFalse(prepare(decode_image(encoded(transparent))).any())
        wide=Image.new('RGB',(200,50),'black');target=prepare(wide)
        y,x=np.where(target)
        self.assertAlmostEqual((x.max()-x.min())/(y.max()-y.min()),4,delta=.4)

    def test_blank_and_threshold(self):
        with self.assertRaises(ValueError):DrawingEnv(prepare(Image.new('RGB',(30,30),'white')))
        with self.assertRaises(ValueError):prepare(fixture('cat'),65)
        self.assertGreaterEqual(prepare(fixture('cat'),threshold=10).sum(),prepare(fixture('cat'),threshold=80).sum())


class EnvironmentTests(unittest.TestCase):
    def test_ideal_offtarget_stationary_and_retracing(self):
        target=np.zeros((64,64),bool);target[32,10:55]=True
        ideal=DrawingEnv(target,steps=128);ideal.x=10;ideal.y=32;ideal.heading=0
        for _ in range(27):ideal.step([0,1,1])
        self.assertGreater(ideal.metrics()['f1'],.9)
        before=ideal.metrics()['score']
        ideal.heading=np.pi
        for _ in range(27):ideal.step([0,1,1])
        self.assertLess(ideal.metrics()['score'],before)
        still=DrawingEnv(target)
        for _ in range(100):still.step([0,0,1])
        self.assertFalse(still.ink.any());self.assertLess(still.metrics()['score'],0)
        off=DrawingEnv(target);off.x=10;off.y=10;off.heading=0
        for _ in range(20):off.step([0,1,1])
        self.assertLess(off.metrics()['score'],0)

    def test_bounds_reproducibility_and_target_features(self):
        a=DrawingEnv(prepare(fixture('cat')),seed=4)
        b=DrawingEnv(prepare(fixture('cat')),seed=4)
        self.assertFalse(np.array_equal(a.observation(),DrawingEnv(prepare(fixture('leaf')),seed=4).observation()))
        for _ in range(256):
            self.assertEqual(a.step([10,3,2]),b.step([10,3,2]))
        self.assertTrue(0<=a.x<128 and 0<=a.y<128)
        with self.assertRaises(ValueError):a.step([0,1,1])
        with self.assertRaises(ValueError):DrawingEnv(prepare(fixture('cat'))).step([float('nan'),1,1])

    def test_export_only_recorded_ink(self):
        blank=render_strokes([])
        raised=render_strokes([[0,0,1,1,.1,0]])
        self.assertEqual(blank.tobytes(),raised.tobytes())
        drawn=render_strokes([[.1,.2,.8,.2,1,0]])
        self.assertNotEqual(blank.tobytes(),drawn.tobytes())

    def test_training_reproducibility_and_mask(self):
        matrix,_=load_graph(nodes=32)
        target=prepare(fixture('cat'),64)
        a,r=train(matrix,target,generations=1,population=4,steps=32)
        b,s=train(matrix,target,generations=1,population=4,steps=32)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(r['held_out'],s['held_out'])
        self.assertGreaterEqual(r['history'][-1]['score'],r['history'][0]['score'])
        controller=Reservoir(matrix,len(DrawingEnv(target).observation()))
        controller.mask[:10]=0
        before=matrix.copy()
        for _ in range(10):controller.action(DrawingEnv(target).observation(),a)
        self.assertFalse(controller.state[:10].any())
        self.assertEqual((matrix!=before).nnz,0)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.context=TestClient(create_app());self.client=self.context.__enter__()
    def tearDown(self):self.context.__exit__(None,None,None)
    def test_upload_start_websocket_pause_reset_export_replace(self):
        target=self.client.post('/api/upload',content=encoded(fixture('cat'))).json()
        run=self.client.post('/api/start',json={'target_id':target['target_id'],'generations':1,'population':4,'steps':32}).json()
        rid=run['run_id']
        self.assertEqual(self.client.post(f'/api/runs/{rid}/pause').status_code,200)
        self.assertEqual(self.client.post(f'/api/runs/{rid}/resume').status_code,200)
        with self.client.websocket_connect(f'/api/events/{rid}') as ws:
            while True:
                event=ws.receive_json()
                self.assertEqual(event['target_id'],target['target_id'])
                if event['status'] in ('done','error'):break
        self.assertEqual(event['status'],'done');self.assertEqual(len(event['strokes']),32)
        export=self.client.get(f'/api/runs/{rid}/drawing.png')
        self.assertEqual(Image.open(BytesIO(export.content)).size,(768,768))
        newer=self.client.post('/api/preset/leaf').json()
        self.assertNotEqual(newer['target_id'],target['target_id'])
        self.assertEqual(self.client.get(f'/api/runs/{rid}').status_code,404)
        self.assertEqual(self.client.post('/api/start',json={'target_id':target['target_id']}).status_code,409)
    def test_bad_upload_blank_and_budgets(self):
        self.assertEqual(self.client.post('/api/upload',content=b'bad').status_code,422)
        self.assertEqual(self.client.post('/api/upload',content=b'x'*(MAX_BYTES+1)).status_code,413)
        target=self.client.post('/api/upload',content=encoded(Image.new('RGB',(32,32),'white'))).json()
        self.assertTrue(target['blank'])
        self.assertEqual(self.client.post('/api/start',json={'target_id':target['target_id']}).status_code,422)
        self.assertEqual(self.client.post('/api/start',json={'target_id':target['target_id'],'generations':1000}).status_code,422)
        self.assertEqual(self.client.post('/api/preset/cat',headers={'Origin':'https://example.com'}).status_code,403)
    def test_replacement_cancels_running_job(self):
        target=self.client.post('/api/preset/cat').json()
        run=self.client.post('/api/start',json={'target_id':target['target_id'],'generations':100}).json()
        self.client.post(f"/api/runs/{run['run_id']}/pause")
        self.client.post('/api/preset/leaf')
        self.assertEqual(self.client.get(f"/api/runs/{run['run_id']}").status_code,404)
    def test_pause_gate_and_cancel(self):
        job=Job('t',1);job.control('pause');done=threading.Event();cancelled=[]
        def work():
            try:job.gate()
            except Cancelled:cancelled.append(True)
            finally:done.set()
        thread=threading.Thread(target=work);thread.start()
        self.assertFalse(done.wait(.03));job.control('cancel')
        self.assertTrue(done.wait(1));thread.join();self.assertEqual(cancelled,[True])


if __name__=='__main__':unittest.main()
