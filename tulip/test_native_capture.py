import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from xvideo.serving.native_capture import NativeCaptureTap


class NativeCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.settings=self.root/'command.json'
        self.tap=NativeCaptureTap(self.settings,self.root/'captures')
        self.frames=[np.full((12,16,3),i,dtype=np.uint8) for i in range(8)]
        self.metas=[{'seq':i,'t_capture_ms':i*1000/12} for i in range(8)]

    def tearDown(self): self.tap.close();self.tmp.cleanup()

    def arm(self, count=8, name='test'):
        self.settings.write_text(json.dumps({'id':name,'max_frames':count}))

    def wait(self):
        self.tap.job.thread.join(timeout=3)
        self.assertFalse(self.tap.job.thread.is_alive())
        return json.loads((self.tap.job.directory/'manifest.json').read_text())

    def test_off_does_not_create_files_or_modify_profile(self):
        p={};self.tap.observe(self.frames,self.metas,p)
        self.assertEqual(p,{});self.assertFalse((self.root/'captures').exists())

    def test_exact_owned_frames_limit_and_no_repeat(self):
        self.arm(3);p={};self.tap.observe(self.frames,self.metas,p)
        for f in self.frames:f[:]=99
        m=self.wait();saved=np.load(self.tap.job.directory/m['chunks'][0]['file'])
        self.assertTrue(m['complete']);self.assertEqual(m['written'],3)
        self.assertEqual(saved[:,0,0,0].tolist(),[0,1,2])
        self.tap.observe(self.frames,self.metas,p)
        self.assertEqual(self.tap.job.written,3)

    def test_scene_boundaries_are_retained(self):
        self.arm(16);self.tap.observe(self.frames,self.metas,{})
        self.tap.reset();self.tap.observe(self.frames,self.metas,{})
        m=self.wait();self.assertEqual([c['segment'] for c in m['chunks']],[0,1])

    def test_bad_command_cannot_escape_capture_root(self):
        self.arm(name='../escape');p={};self.tap.observe(self.frames,self.metas,p)
        self.assertIn('Invalid capture ID',p['native_capture']['error'])
        self.assertFalse((self.root/'escape').exists())

    def test_queue_pressure_invalidates_capture_without_raising(self):
        self.arm(600)
        with patch('xvideo.serving.native_capture.np.save',side_effect=lambda *a,**k:time.sleep(.2)):
            for _ in range(10): self.tap.observe(self.frames,self.metas,{})
            m=self.wait()
        self.assertFalse(m['complete']);self.assertIn('queue full',m['error'])

    def test_existing_capture_never_overwritten(self):
        self.arm();self.tap.observe(self.frames,self.metas,{});m=self.wait()
        other=NativeCaptureTap(self.settings,self.root/'captures');p={}
        other.observe(self.frames,self.metas,p)
        self.assertIn('FileExistsError',p['native_capture']['error'])
        self.assertEqual(json.loads((self.tap.job.directory/'manifest.json').read_text()),m)

    def test_disarm_records_incomplete_not_success(self):
        self.arm(16);self.tap.observe(self.frames,self.metas,{})
        self.settings.write_text('{}');self.tap.observe(self.frames,self.metas,{})
        m=self.wait();self.assertFalse(m['complete']);self.assertEqual(m['written'],8)


if __name__=='__main__':unittest.main()
