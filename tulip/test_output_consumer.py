"""Input submission must not steal results from the dedicated output consumer."""
import ast
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

source=Path(__file__).resolve().parents[1]/'deploy/xvideo/serving/joyomni_streaming.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JoyOmniV2VStreamingSession')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='push_frame')
namespace={'torch':torch,'Image':Image,'Any':object,'StreamingChunkResult':object}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)

class OutputConsumer(unittest.TestCase):
    def session(self):
        self.results=[object(),object()]
        self.submitted=[]
        def drain():
            result=self.results[:];self.results.clear();return result
        return SimpleNamespace(_submit_lock=threading.RLock(),initialized=True,
            _raise_worker_error_if_needed=lambda:None,_resize_frame=lambda f:f,
            pending_frames=[],pending_metas=[],chunk_idx=1,frames_per_next_chunk=1,
            _submit_or_process_chunk=lambda frames,metas:self.submitted.append((frames,metas)) or [],
            _drain_async_results=drain)

    def test_dedicated_consumer_keeps_results_in_order(self):
        session=self.session();expected=self.results[:]
        self.assertEqual(namespace['push_frame'](session,'frame',{'seq':8},drain_results=False),[])
        self.assertEqual(len(self.submitted),1)
        self.assertEqual(session._drain_async_results(),expected)

    def test_offline_default_still_returns_results(self):
        session=self.session();expected=self.results[:]
        self.assertEqual(namespace['push_frame'](session,'frame',{'seq':8}),expected)
        self.assertEqual(self.results,[])

if __name__=='__main__': unittest.main()
