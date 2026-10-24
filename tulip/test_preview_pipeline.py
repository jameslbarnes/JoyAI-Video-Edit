import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest

from xvideo.serving.preview_pipeline import DeferredPreview, LatestPreview


class PreviewPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_preview_does_not_block_later_chunks_and_pending_memory_is_bounded(self):
        entered, release = threading.Event(), threading.Event()
        delivered, published = [], []
        def encode(frame, quality):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Test failed to release encoder")
            return bytes([frame])
        async def send(frames, chunk):
            delivered.append((chunk, frames))
        with ThreadPoolExecutor(max_workers=2) as executor:
            consumer = LatestPreview(send)
            def publish(chunk):
                published.extend(range(chunk*4, chunk*4+4))
                consumer.offer(DeferredPreview(
                    [(encode, frame, 95) for frame in published[-4:]], executor), chunk)
            try:
                publish(0)
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                # Eight subsequent chunks complete while every JPEG worker is held.
                for chunk in range(1, 9):
                    publish(chunk)
                self.assertEqual(published, list(range(36)))
                self.assertEqual(consumer.status()["pending_frames"], 4)
                self.assertEqual(consumer.status()["superseded_frames"], 28)
                self.assertEqual(delivered, [])
                release.set()
                for _ in range(100):
                    if len(delivered) == 2:
                        break
                    await asyncio.sleep(.01)
                self.assertEqual(delivered, [(0, [bytes([i]) for i in range(4)]),
                                              (8, [bytes([i]) for i in range(32, 36)])])
            finally:
                release.set()
                await consumer.close()

    async def test_close_drains_workers_and_never_delivers_retired_scene(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        delivered = []
        def encode(frame, quality):
            entered.set()
            release.wait(3)
            finished.set()
            return b"old scene"
        async def send(*frames):
            delivered.append(frames)
        with ThreadPoolExecutor(max_workers=1) as executor:
            consumer = LatestPreview(send)
            consumer.offer(DeferredPreview([(encode, object(), 95)], executor))
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            closing = asyncio.create_task(consumer.close())
            await asyncio.sleep(.02)
            self.assertFalse(closing.done())
            release.set()
            await closing
            self.assertTrue(finished.is_set())
            self.assertEqual(delivered, [])
            with self.assertRaises(RuntimeError):
                consumer.offer(DeferredPreview([], executor))

    async def test_encoder_failure_is_surfaced_and_owned_jobs_are_released(self):
        def broken(*args):
            raise ValueError("encode failed")
        async def send(*args):
            self.fail("Failed batch must not be sent")
        with ThreadPoolExecutor(max_workers=1) as executor:
            consumer = LatestPreview(send)
            batch = DeferredPreview([(broken, object(), 95)], executor)
            consumer.offer(batch)
            for _ in range(100):
                if consumer.task.done():
                    break
                await asyncio.sleep(.01)
            with self.assertRaisesRegex(ValueError, "encode failed"):
                consumer.offer(batch)
            self.assertEqual(batch.jobs, [])
            with self.assertRaisesRegex(ValueError, "encode failed"):
                await consumer.close()


if __name__ == "__main__":
    unittest.main()
