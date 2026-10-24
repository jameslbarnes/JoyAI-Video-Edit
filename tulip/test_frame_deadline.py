import asyncio
import threading
import unittest

from xvideo.serving.frame_deadline import FrameDeadlineExceeded, run_frame_with_deadline


class FrameDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_timeout_keeps_real_input_failure(self):
        failure = TimeoutError('No fresh Spout frame within 0.5s')
        def capture():
            raise failure
        with self.assertRaises(TimeoutError) as raised:
            await run_frame_with_deadline(capture, 1)
        # asyncio's concurrent-future adapter may recreate built-in timeout
        # exceptions; type/message, rather than object identity, are the contract.
        self.assertEqual(type(raised.exception), type(failure))
        self.assertEqual(raised.exception.args, failure.args)
        self.assertNotIsInstance(raised.exception, FrameDeadlineExceeded)

    async def test_outer_deadline_is_identified_without_waiting_for_worker(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def work():
            entered.set()
            try:
                release.wait(2)
            finally:
                finished.set()
        try:
            with self.assertRaises(FrameDeadlineExceeded):
                await run_frame_with_deadline(work, .1)
            self.assertTrue(entered.is_set())
            self.assertFalse(finished.is_set())
        finally:
            release.set()
            await asyncio.to_thread(finished.wait, 2)

    async def test_success_and_other_worker_errors_are_preserved(self):
        result = object()
        self.assertIs(await run_frame_with_deadline(lambda: result, 1), result)
        def failed():
            raise RuntimeError('native reader failed')
        with self.assertRaisesRegex(RuntimeError, 'native reader failed'):
            await run_frame_with_deadline(failed, 1)

    async def test_caller_cancellation_is_not_reported_as_deadline(self):
        release = threading.Event()
        task = asyncio.create_task(run_frame_with_deadline(lambda: release.wait(2), 1))
        await asyncio.sleep(.02)
        try:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            release.set()


if __name__ == '__main__':
    unittest.main()
