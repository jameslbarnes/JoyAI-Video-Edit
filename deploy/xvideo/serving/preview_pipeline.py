"""Optional browser previews cannot backpressure the paced presentation outputs."""
import asyncio
from dataclasses import dataclass

from .presentation_pipeline import finish_thread


@dataclass
class DeferredPreview:
    # Owned frame references: never views into a producer's reusable storage.
    jobs: list
    executor: object

    def __len__(self):
        return len(self.jobs)

    def encode(self):
        futures = []
        try:
            for function, pixels, quality in self.jobs:
                futures.append(self.executor.submit(function, pixels, quality))
            return [future.result() for future in futures]
        finally:
            # Cancellation/error cannot release CUDA inputs while workers use them.
            for future in futures:
                if not future.cancelled():
                    future.exception()
            self.jobs.clear()


class LatestPreview:
    """One active batch and one replaceable pending batch per websocket.

    Only optional JPEG previews are superseded. Every generated presentation
    frame still goes through the local and encoded-output publishers in order.
    """
    def __init__(self, send):
        self.send = send
        self.pending = None
        self.changed = asyncio.Event()
        self.closed = False
        self.superseded = 0
        self.completed = 0
        self.task = None

    def offer(self, batch, *metadata):
        if self.closed:
            raise RuntimeError("Preview consumer is closed")
        if self.task is not None and self.task.done():
            self.task.result()  # Surface failures; do not silently lose previews.
        if self.pending is not None:
            self.superseded += len(self.pending[0])
        self.pending = (batch, metadata)
        self.changed.set()
        if self.task is None:
            self.task = asyncio.create_task(self._run())

    async def _run(self):
        while not self.closed:
            await self.changed.wait()
            item, self.pending = self.pending, None
            self.changed.clear()
            if item is None:
                continue
            batch, metadata = item
            encoded = await finish_thread(batch.encode)
            if not self.closed:
                await self.send(encoded, *metadata)
                self.completed += len(encoded)

    async def close(self):
        self.closed = True
        self.pending = None
        self.changed.set()
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            finally:
                self.task = None

    def status(self):
        return dict(pending_frames=len(self.pending[0]) if self.pending else 0,
                    superseded_frames=self.superseded, completed_frames=self.completed)
