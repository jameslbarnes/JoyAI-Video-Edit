"""Small CPU-only metrics for performance tests; throughput alone is not cadence."""
import math
import statistics


def interval_metrics(intervals_ms, expected_ms):
    values = list(intervals_ms)
    if not values:
        raise ValueError("No intervals measured")
    if expected_ms <= 0 or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid timing samples")
    ordered = sorted(values)
    return dict(count=len(values), expectedMs=expected_ms, meanMs=statistics.mean(values),
                p50Ms=statistics.median(values), p95Ms=ordered[int((len(values)-1)*.95)],
                p99Ms=ordered[int((len(values)-1)*.99)], maxMs=ordered[-1],
                stddevMs=statistics.pstdev(values),
                lateIntervals=sum(v > expected_ms*1.5 for v in values),
                overBudget=sum(v > expected_ms for v in values))


def native_sender_metrics(state, fps):
    """Native Spout aggregates are exact; do not invent unmeasured percentiles."""
    count = max(0, state["sent"]-1)
    if not count:
        raise ValueError("No native sender intervals")
    mean = state["gap_sum_ms"]/count
    variance = max(0, state["gap_square_sum"]/count - mean*mean)
    return dict(intervals=count, fps=1000/mean if mean > 0 else 0,
                meanMs=mean, stddevMs=math.sqrt(variance), maxMs=state["max_gap_ms"],
                expectedMs=1000/fps, underruns=state["underruns"],
                missedTicks=state["missed_ticks"],
                percentiles="Measured at the receiver/projector by the required live acceptance probe")


def require_native_cadence(state, fps, max_gap_frames=2.5):
    metrics = native_sender_metrics(state, fps)
    if (not fps*.98 <= metrics["fps"] <= fps*1.02 or
            metrics["maxMs"] > 1000/fps*max_gap_frames or
            metrics["stddevMs"] > 1000/fps*.20):
        raise AssertionError(f"Native frame cadence failed: {metrics}")
    if metrics["underruns"]:
        raise AssertionError(f"Native playback underruns: {metrics}")
    return metrics
