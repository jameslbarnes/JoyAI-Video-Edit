import struct
import sys
import unittest
import time
import subprocess
import json
import threading
from contextlib import nullcontext
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deploy'))
from xvideo.serving.gpu_video_output import FlvH264,annexb_nalus,GpuVideoPublisher,GpuFrame
from xvideo.serving.upscale_output import OutputUpscale

class GpuVideoContract(unittest.TestCase):
    def test_uhd_geometry_requires_matching_rgb_tensors(self):
        p=GpuVideoPublisher('cpu',sys.executable,'rtmp://127.0.0.1:1935/test')
        good=GpuFrame(torch.empty((2160,3840,3),dtype=torch.uint8),None)
        self.assertEqual(tuple(p._validate_frames([good,good])),(2160,3840,3))
        for bad in (GpuFrame(torch.empty((2160,3888,3),dtype=torch.uint8),None),
                    GpuFrame(torch.empty((2160,3840,3),dtype=torch.float32),None),
                    GpuFrame(torch.empty((1440,2592,3),dtype=torch.uint8),None)):
            with self.assertRaises(ValueError): p._validate_frames([good,bad])
        with self.assertRaises(ValueError):
            p._validate_frames([GpuFrame(torch.empty((2160,3888,3),dtype=torch.uint8),None)])

    def test_close_survives_worker_releasing_pacer_during_poll(self):
        p=GpuVideoPublisher('cpu',sys.executable,'rtmp://127.0.0.1:1935/test')
        terminated=[]
        def poll():
            p.pacer=None  # The worker finishes concurrently with close().
            return None
        p.pacer=SimpleNamespace(poll=poll,terminate=lambda:terminated.append(True))
        p.close()
        self.assertEqual(terminated,[True])
        self.assertIsNone(p.pacer)
        self.assertIsNone(p.thread)
        p.close()

    def test_encoded_prefetch_preserves_order_and_stops_both_workers(self):
        class Publisher(GpuVideoPublisher):
            def __init__(self):
                super().__init__('cpu',sys.executable,'rtmp://127.0.0.1:1935/test')
                self.received=[];self.produced=0;self.max_encoded=0
            def _validate_frames(self,frames):return (360,648,3)
            def _open(self,key):self.geometry=key
            def _release_encoder(self):pass
            def _encode(self,frame,fps):
                # GPU stalls exceed two display intervals, but remain smaller
                # than the encoded reserve. Transport must remain evenly paced.
                delay=.12 if self.produced in (10,20) else .001
                time.sleep(delay);self.produced+=1;self.stage_ms={'prepare':delay*1000}
                return [bytes([int(frame[0])])]
            def _write(self,packet):
                self.received.append((packet[0],time.perf_counter()))
                self.max_encoded=max(self.max_encoded,len(self.encoded))
        with patch('torch.cuda.device',lambda *a,**k:nullcontext()),patch('torch.cuda.stream',lambda *a,**k:nullcontext()),patch('torch.cuda.Stream',lambda **k:SimpleNamespace(priority=-1)):
            p=Publisher()
            try:
                p.submit([np.array([i],np.uint8) for i in range(32)],24)
                deadline=time.monotonic()+4
                while len(p.received)<32 and time.monotonic()<deadline:time.sleep(.005)
                self.assertEqual([v for v,_ in p.received],list(range(32)))
                self.assertLessEqual(p.max_encoded,p.encode_ahead)
                self.assertEqual(p.underruns,0)
                self.assertIsNone(p.error)
            finally:p.close()
            self.assertIsNone(p.thread)
            self.assertEqual(len(p.encoded),0)

    def test_isolated_pacer_hides_encoder_stalls(self):
        script=Path(__file__).resolve().parents[1]/'deploy/xvideo/serving/encoded_pacer.py'
        p=subprocess.Popen([sys.executable,'-u',str(script),'24'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
        def send(payload):p.stdin.write(struct.pack('<I',len(payload))+payload)
        try:
            send(b'H')
            def produce():
                for i in range(32):
                    time.sleep(.12 if i in (10,20) else .001)
                    send(bytes([i]))
            worker=threading.Thread(target=produce);worker.start();worker.join(3)
            self.assertFalse(worker.is_alive())
            p.stdin.close();p.stdin=None
            out,err=p.communicate(timeout=5)
            self.assertEqual(p.returncode,0,err)
            self.assertEqual(out,b'H'+bytes(range(32)))
            states=[json.loads(line) for line in err.splitlines()]
            if sys.platform == 'win32':
                self.assertTrue(states[-1]['delivery_priority']['ok'])
                self.assertEqual(states[-1]['delivery_priority']['value'], 0x8000)
            self.assertEqual(states[-1]['sent'],32)
            self.assertEqual(states[-1]['underruns'],0)
            self.assertLess(states[-1]['max_gap_ms'],75)
            self.assertLessEqual(max(s['queued'] for s in states),12)
            self.assertGreater(states[-1]['last_send_at']-states[0]['last_send_at'],1.2)
        finally:
            if p.poll() is None:p.kill();p.wait()

    def test_annexb_mux_keeps_sps_pps_and_exact_frame_timestamp(self):
        mux=FlvH264();sps=b'\x67\x64\x00\x32\xaa';pps=b'\x68\xbb';idr=b'\x65\xcc'
        data=mux.packet(b'\x00\x00\x00\x01'+sps+b'\x00\x00\x01'+pps+b'\x00\x00\x01'+idr,0x12345678)
        tags=[]
        while data:
            size=int.from_bytes(data[1:4],'big');stamp=int.from_bytes(data[4:7],'big')|(data[7]<<24)
            self.assertEqual(stamp,0x12345678)
            self.assertEqual(int.from_bytes(data[11+size:15+size],'big'),11+size)
            tags.append(data[11:11+size]);data=data[15+size:]
        self.assertEqual(tags[0][:5],b'\x17\x00\x00\x00\x00')
        self.assertEqual(tags[1],b'\x17\x01\x00\x00\x00'+len(idr).to_bytes(4,'big')+idr)
        following=mux.packet(b'\x00\x00\x01\x41\xdd',33)
        self.assertEqual(following[11:16],b'\x27\x01\x00\x00\x00')

    def test_rejects_video_without_parameter_sets(self):
        with self.assertRaises(ValueError): FlvH264().packet(b'\x00\x00\x01\x65\xcc',0)

    def test_both_publishers_receive_reset_and_close_independently(self):
        calls=[]
        def publisher(name,preserve):
            return SimpleNamespace(begin_transition=lambda source,**kw:calls.append((name,'begin',source(),kw['moving'])),
                cancel_transition=lambda:calls.append((name,'cancel')),
                reset_scene=lambda:(calls.append((name,'reset')) or preserve),
                close=lambda:calls.append((name,'close')))
        output=OutputUpscale.__new__(OutputUpscale)
        output.direct_publisher=publisher('ndi',True);output.gpu_publisher=publisher('gpu',False)
        output.begin_transition(lambda:42,moving=True)
        output.close_presentation(preserve=True)
        output.cancel_transition()
        self.assertEqual(calls,[('ndi','begin',42,True),('gpu','begin',42,True),('ndi','reset'),('gpu','reset'),('gpu','close'),('ndi','cancel'),('gpu','cancel')])

if __name__=='__main__':unittest.main()
