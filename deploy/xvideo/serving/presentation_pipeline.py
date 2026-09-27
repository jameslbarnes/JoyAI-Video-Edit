"""Ordered two-stage presentation with exactly one item of lookahead."""
import asyncio


async def finish_thread(function, *args, **kwargs):
    """Do not let cancellation leave GPU work mutating a reused session."""
    work = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        await work
        raise


async def ordered_prefetch(prepare_next, consume, stopped):
    pending = asyncio.create_task(prepare_next())
    try:
        while not stopped.is_set():
            item = await asyncio.shield(pending)
            pending = None
            if item is None:
                break
            pending = asyncio.create_task(prepare_next())
            await consume(item)
    finally:
        stopped.set()
        if pending is not None:
            # prepare_next observes stopped while polling. If interpolation is
            # already running, drain it before a reset can reuse its state.
            await pending
