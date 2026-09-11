"""Endpoint-based throughput and latency summary; no playback-FPS substitution."""

import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("metrics")
a = p.parse_args()
path = Path(a.metrics)
m = json.loads(path.read_text())
chunks = m["chunks"]
assert len(chunks) > 2 and "generated_frames" in m
warm = chunks[2:]
warm_seconds = chunks[-1]["received_s"] - chunks[1]["received_s"]
latencies = [x for c in warm for x in c["submission_to_jpeg_ms"]]
dit_seconds = sum(c["profile"]["dit_denoise_s"] for c in warm)
summary = {
    "generated_frames": m["generated_frames"],
    "source_duration_s": m["generated_frames"] / m["source_fps"],
    "load_including_vae_warmup_s": m["load_including_upstream_vae_warmup_s"],
    "time_to_first_generated_jpeg_s": m["time_to_first_generated_jpeg_s"],
    "warmup_excluded_chunks": 2,
    "warm_generated_frames": sum(c["valid_frames"] for c in warm),
    "warm_delivery_window_s": warm_seconds,
    "warm_delivery_generated_fps": sum(c["valid_frames"] for c in warm) / warm_seconds,
    "warm_dit_stage_fps_cuda_events": sum(c["valid_frames"] for c in warm)
    / dit_seconds,
    "dit_stage_scope": "Graph replay includes denoising, scheduler arithmetic, clean-KV store and commit; NOT pure DiT-only"
    if any(c["profile"].get("graph_path") for c in warm)
    else "Eager denoising transformer forwards only; excludes separately timed KV store",
    "warm_submission_to_jpeg_ms_p50": float(np.percentile(latencies, 50)),
    "warm_submission_to_jpeg_ms_p95": float(np.percentile(latencies, 95)),
    "latency_endpoints": "Unpaced local frame submission to generated JPEG result received; includes bounded queue. Excludes live capture, network and display.",
    "chunk_input_frames": 8,
    "max_inflight_chunks": 2,
    "chunk_accumulation_at_16fps_oldest_frame_ms": 7 / 16 * 1000,
    "accumulation_note": "Calculated for paced 16 FPS input, not measured in this unpaced test; do not add directly to unpaced latency.",
    "interpolation": False,
    "repetition": False,
    "peak_torch_allocated_gib": m["peak_torch_allocated_gib"],
    "peak_torch_reserved_gib": m["peak_torch_reserved_gib"],
    "stages_mean_s": {
        key: float(np.mean([c["profile"][key] for c in warm]))
        for key in [
            "vae_encode_s",
            "dit_denoise_s",
            "kv_store_s",
            "vae_decode_s",
            "jpeg_encode_s",
            "total_server_chunk_s",
        ]
    },
    "timing_note": "Server stages overlap across chunks. Server elapsed time and queue waits must not be summed as disjoint stages.",
}
(path.parent / "benchmark-summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
