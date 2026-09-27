"""CPU-only reproduction of session-age-dependent window allocation."""
import gc
import json
from pathlib import Path
import statistics
import time
import tracemalloc

from test_streaming_window_lookup import Pipeline


def measure(fn, total, repeats):
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn(total, 1, 2, True)
        timings.append((time.perf_counter() - started) * 1000)
        del result
    gc.collect()
    tracemalloc.start()
    result = fn(total, 1, 2, True)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return {"median_ms": statistics.median(timings), "peak_bytes": peak}


if __name__ == "__main__":
    results = []
    for total in [100, 50_000, 100_000, 1_000_000_000]:
        row = {"chunks": total, "direct": measure(Pipeline._get_last_chunk_window, total, 100)}
        if total <= 100_000:
            row["historical"] = measure(Pipeline._get_chunk_windows, total, 5)
        results.append(row)
    path = Path(__file__).with_name("streaming-window-benchmark-20260916.json")
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
