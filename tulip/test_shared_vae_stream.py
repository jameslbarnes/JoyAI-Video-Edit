"""Exercise the actual worker setup without loading model weights."""
import ast
from contextlib import nullcontext
import os
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

source = ast.parse((Path(__file__).parents[1]/"deploy/xvideo/serving/joyomni_streaming.py").read_text(encoding="utf-8"))
cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "JoyOmniV2VStreamingSession")
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("_start_async_workers", "_vae_submission")]
class Module:
    def __init__(self, device): self.device = device
class Device(str):
    @property
    def type(self): return self.split(":")[0]
class Thread:
    def __init__(self, **kw): pass
    def start(self): pass
class SharedVaeTests(unittest.TestCase):
    def session(self, shared, decode="cuda:0", pseudo=None, runtime=None):
        ns=dict(os=os, queue=queue, nullcontext=nullcontext,
                threading=SimpleNamespace(Thread=Thread,RLock=threading.RLock),
                _module_device=lambda m: m.device, _canonical_device=str,
                torch=SimpleNamespace(nn=SimpleNamespace(Module=Module),device=Device,
                cuda=SimpleNamespace(is_available=lambda:True,Stream=lambda **kw:object())))
        exec(compile(ast.Module(body=methods,type_ignores=[]),"worker-setup","exec"),ns)
        s=SimpleNamespace(_encode_queue=None,pipeline=SimpleNamespace(vae=Module("cuda:0")),decode_vae=Module(decode),
            pseudo_encode_vae=Module(pseudo or decode), device="cuda:0",postprocess_device="cuda:1",
            runtime=runtime or SimpleNamespace(vae_worker_streams={},vae_submission_locks={}))
        for name in ("encode","dit","decode","pseudo","postprocess"): setattr(s,"_"+name+"_worker",lambda:None)
        with patch.dict(os.environ,JOYAI_SHARED_VAE_STREAM=str(int(shared))): ns["_start_async_workers"](s)
        return s,ns["_vae_submission"]
    def test_same_device_reuses_stream_and_submission_lock(self):
        s,lock=self.session(True)
        self.assertIs(s._encode_stream,s._decode_stream)
        self.assertIs(s._decode_stream,s._pseudo_stream)
        self.assertIsNot(s._dit_stream,s._decode_stream)
        self.assertIs(lock(s,s.pipeline.vae),lock(s,s.decode_vae))
    def test_cross_device_work_remains_independent(self):
        s,lock=self.session(True,"cuda:1")
        self.assertIsNot(s._encode_stream,s._decode_stream)
        self.assertIs(s._decode_stream,s._pseudo_stream)
        self.assertIsNot(lock(s,s.pipeline.vae),lock(s,s.decode_vae))
    def test_disabled_preserves_original_streams(self):
        s,lock=self.session(False)
        self.assertIsNot(s._encode_stream,s._decode_stream)
        with lock(s,s.decode_vae): pass
    def test_pseudo_offload_keeps_encode_decode_shared(self):
        s,lock=self.session(True,pseudo="cuda:1")
        self.assertIs(s._encode_stream,s._decode_stream)
        self.assertIsNot(s._decode_stream,s._pseudo_stream)
        self.assertIsNot(lock(s,s.decode_vae),lock(s,s.pseudo_encode_vae))
    def test_replacing_session_reuses_runtime_vae_streams_and_locks(self):
        first,lock=self.session(True,pseudo="cuda:1")
        second,_=self.session(True,pseudo="cuda:1",runtime=first.runtime)
        self.assertIs(first._encode_stream,second._encode_stream)
        self.assertIs(first._decode_stream,second._decode_stream)
        self.assertIs(first._pseudo_stream,second._pseudo_stream)
        self.assertIs(lock(first,first.decode_vae),lock(second,second.decode_vae))
    def test_startup_does_not_need_worker_lock(self):
        _,lock=self.session(True)
        with lock(SimpleNamespace(),Module("cuda:0")): pass

if __name__ == "__main__": unittest.main()
