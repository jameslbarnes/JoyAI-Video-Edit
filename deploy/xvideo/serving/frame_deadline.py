"""Keep a worker's own TimeoutError distinct from the outer submission deadline."""
import asyncio


class FrameDeadlineExceeded(TimeoutError):
    pass


async def run_frame_with_deadline(function, seconds):
    deadline = asyncio.timeout(seconds)
    try:
        async with deadline:
            return await asyncio.to_thread(function)
    except TimeoutError as exc:
        # asyncio.TimeoutError aliases built-in TimeoutError. A Spout fresh-frame
        # timeout or inference-lock timeout must retain its real message/source.
        if deadline.expired():
            raise FrameDeadlineExceeded(f'frame submission exceeded {seconds:.1f}s') from exc
        raise
