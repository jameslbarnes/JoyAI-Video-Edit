import asyncio
import threading
import unittest
from xvideo.serving.presentation_pipeline import finish_thread, ordered_prefetch


class PresentationPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_overlap_is_bounded_and_ordered(self):
        stopped=asyncio.Event()
        release=asyncio.Event()
        next_ready=asyncio.Event()
        prepared=[]; consumed=[]
        async def prepare():
            index=len(prepared)
            if index==4:return None
            prepared.append(index)
            if index==1:next_ready.set()
            return index
        async def consume(item):
            if item==0:await release.wait()
            consumed.append(item)
        work=asyncio.create_task(ordered_prefetch(prepare,consume,stopped))
        await asyncio.wait_for(next_ready.wait(),1)
        self.assertEqual(prepared,[0,1])
        self.assertEqual(consumed,[])
        release.set();await work
        self.assertEqual(consumed,[0,1,2,3])

    async def test_stop_drains_interpolation_before_returning(self):
        stopped=asyncio.Event();started=threading.Event();release=threading.Event()
        finished=threading.Event();consumed=[]
        def gpu():
            started.set();release.wait(2);finished.set();return 'frame'
        async def prepare():
            if stopped.is_set():return None
            return await finish_thread(gpu)
        async def consume(item):consumed.append(item);stopped.set()
        work=asyncio.create_task(ordered_prefetch(prepare,consume,stopped))
        await asyncio.to_thread(started.wait,1)
        work.cancel()
        await asyncio.sleep(.02)
        self.assertFalse(work.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):await work
        self.assertTrue(finished.is_set())
        self.assertEqual(consumed,[])

if __name__=='__main__':unittest.main()
