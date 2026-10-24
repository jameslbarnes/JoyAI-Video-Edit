"""Allow a short source interruption without discarding the loaded scene.

Only an explicit no-fresh-frame result is retryable. Driver/GPU failures and
closed receivers propagate immediately. This never manufactures an input frame.
"""
import time


class FreshInputTimeout(TimeoutError):
    pass


def capture_model_input(reader, width, height, grace_s=2.0):
    if not .5 <= grace_s <= 5:
        raise ValueError('Fresh-input grace must be bounded to 0.5..5 seconds')
    reader.metrics['fresh_frame_grace_s'] = grace_s
    start = time.monotonic()
    retries = 0
    while True:
        remaining = grace_s-(time.monotonic()-start)
        try:
            frame = reader.capture(width, height, timeout=min(.5, max(.001, remaining)))
        except FreshInputTimeout as error:
            retries += 1
            reader.metrics['fresh_wait_timeouts'] = reader.metrics.get('fresh_wait_timeouts', 0)+1
            reader.metrics['last_fresh_wait_at'] = time.time()
            if time.monotonic()-start >= grace_s:
                raise FreshInputTimeout(
                    f'No fresh input after {grace_s:.1f}s grace ({retries} attempts): {error}') from error
        else:
            if retries:
                reader.metrics['fresh_wait_recoveries'] = reader.metrics.get('fresh_wait_recoveries', 0)+1
                reader.metrics['last_fresh_recovery_ms'] = (time.monotonic()-start)*1000
            return frame
