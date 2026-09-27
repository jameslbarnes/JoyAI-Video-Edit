"""Exercise the real output pump's watchdog accounting without loading weights."""
import ast
import asyncio
import queue
import unittest
from pathlib import Path
from types import SimpleNamespace

source=Path(__file__).resolve().parents[1]/'deploy/xvideo/serving/serve_joyomni_streaming.py'
tree=ast.parse(source.read_text())
pump=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=='_output_pump')

class PresentationActivityTests(unittest.IsolatedAsyncioTestCase):
    async def run_pump(self,multiplier):
        work={'pending':0,'completed':0}
        stopped=asyncio.Event(); sending=asyncio.Event(); release=asyncio.Event()
        results=queue.Queue()
        results.put(SimpleNamespace(jpegs=[b'pixel'],source_metas=[{}],valid_count=None,profile={},elapsed=0.1))
        def wait_result(timeout):
            try: return results.get(timeout=timeout)
            except queue.Empty: return None
        async def send(*args,**kwargs):
            sending.set()
            await release.wait()
            stopped.set()
        rife=SimpleNamespace(process=lambda pixels,metas,*args,**kwargs:(pixels,metas),transition_pending=False)
        ns={'presentation_work':work,'asyncio':asyncio,'rife_multiplier':multiplier,'rife':rife,
            'app':SimpleNamespace(state=SimpleNamespace(upscaler=SimpleNamespace(interpolation_after=lambda:False))),
            'runtime':SimpleNamespace(run_interpolation=lambda fn,*args,**kwargs:fn(*args,**kwargs)),
            'stop_output_pump':stopped,'output_quality':95,'_send_encoded_frames':send}
        exec(compile(ast.Module(body=[pump],type_ignores=[]),str(source),'exec'),ns)
        task=asyncio.create_task(ns['_output_pump'](SimpleNamespace(wait_async_result=wait_result)))
        try:
            await asyncio.wait_for(sending.wait(),2)
            self.assertEqual(work,{'pending':1,'completed':0})
            release.set()
            await asyncio.wait_for(task,2)
            self.assertEqual(work,{'pending':0,'completed':1})
        finally:
            stopped.set();release.set()
            if not task.done(): await task
    async def test_without_interpolation(self): await self.run_pump(1)
    async def test_with_interpolation_prefetch(self): await self.run_pump(2)

if __name__=='__main__': unittest.main()
