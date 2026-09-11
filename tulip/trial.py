"""Bounded real-model V2V test through the upstream persistent session API."""

# ruff: noqa: E402 -- validate arguments before importing CUDA/model libraries.
import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    default=r"C:\Users\Admin\src\sana-streaming\tulip\evidence\shader-ab\input.mp4",
)
parser.add_argument("--output-dir", default=str(ROOT / "tulip/evidence/shader-valid"))
parser.add_argument("--width", type=int, default=648)
parser.add_argument("--height", type=int, default=360)
parser.add_argument("--frames", type=int, default=33)
parser.add_argument("--steps", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--prompt",
    default="Transform the flowing neon shapes into an intricate bioluminescent alien coral reef, with translucent organic tendrils, rich texture and cinematic depth. Preserve the source movement and composition while completely replacing the flat synthetic surfaces with detailed living forms.",
)
args = parser.parse_args()
if args.width <= 0 or args.height <= 0 or args.width % 24 or args.height % 24:
    parser.error("width and height must be positive multiples of 24")
if args.frames < 1 or (args.frames - 1) % 8:
    parser.error("use one initial frame plus complete eight-frame chunks")
if (Path(args.output_dir) / "metrics.json").exists():
    parser.error("output directory already contains evidence; choose a new directory")

import cv2
import numpy as np
import torch
from PIL import Image
from xvideo.serving.joyomni_streaming import JoyOmniRuntime, StreamingSettings

out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
report = {
    "arguments": vars(args),
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(),
    "environment": {
        k: v
        for k, v in os.environ.items()
        if k.startswith("JOYOMNI_")
        or k in ["CUDA_VISIBLE_DEVICES", "TORCH_COMPILE_DISABLE"]
    },
    "precision": "FP8 image/text DiT linears, BF16 activations and VAE; Triton fast accumulation",
    "interpolation": False,
    "frame_repetition": False,
    "chunks": [],
    "status": "loading",
}
# Fail explicitly if the trial cannot fit; leave headroom for existing desktop/services.
free_bytes, total_bytes = torch.cuda.mem_get_info()
# WDDM can report CUDA free memory above physical headroom. Bound against NVML's
# physical-device report as well; never silently fall back if that query fails.
physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
physical_free_mib = float(
    subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            physical_gpu,
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
)
budget_bytes = max(0, min(free_bytes, physical_free_mib * 2**20) - int(0.75 * 2**30))
torch.cuda.set_per_process_memory_fraction(budget_bytes / total_bytes)
report["free_vram_before_gib"] = free_bytes / 2**30
report["physical_free_vram_before_gib"] = physical_free_mib / 1024
report["allocator_budget_gib"] = budget_bytes / 2**30


def save():
    (out / "metrics.json").write_text(json.dumps(report, indent=2))


save()
runtime = None
session = None
writer = None
cap = cv2.VideoCapture(args.input)
if not cap.isOpened():
    raise RuntimeError(f"Cannot read input: {args.input}")
fps = cap.get(cv2.CAP_PROP_FPS)
report["source_fps"] = fps
models = ROOT / "tulip/models"
started = time.perf_counter()
try:
    runtime = JoyOmniRuntime.load(
        str(models / "JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth"),
        vae_ckpt=str(models / "JoyAI-Video-Edit/vae"),
        text_encoder_ckpt=str(models / "MiMo-VL-7B-RL-2508"),
        device="cuda:0",
        vae_encode_device="cuda:0",
        vae_decode_device="cuda:0",
        vae_pseudo_device="cuda:0",
        postprocess_device="cuda:0",
        seed=args.seed,
        warmup_height=args.height,
        warmup_width=args.width,
    )
    torch.cuda.synchronize()
    report["load_including_upstream_vae_warmup_s"] = time.perf_counter() - started
    runtime.output_quality = 95
    session = runtime.create_v2v_session(
        args.prompt,
        settings=StreamingSettings(
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            seed=args.seed,
            freeze_kv_on_static=False,
            profile_timings=True,
            max_temporal_ids=128,
            output_codec="mjpeg",
        ),
    )
    writer = cv2.VideoWriter(
        str(out / "output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError("Output video writer could not be opened")
    received = 0
    submitted = {}
    run_started = time.perf_counter()

    def consume(result):
        global received
        now = time.perf_counter()
        if received == 0:
            report["time_to_first_generated_jpeg_s"] = now - run_started
        pictures = result.jpegs or []
        count = result.valid_count if result.valid_count is not None else len(pictures)
        record = {
            "profile": result.profile,
            "upstream_elapsed_s": result.elapsed,
            "received_s": now - run_started,
            "source_metas": result.source_metas,
            "valid_frames": count,
            "submission_to_jpeg_ms": [
                (now - submitted[m["seq"]]) * 1000
                for m in result.source_metas
                if m["seq"] in submitted
            ],
        }
        report["chunks"].append(record)
        for data in pictures[:count]:
            rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
            if received in {0, 1, 8, 9, 16, 17, 24, 25, 32}:
                Image.fromarray(rgb).save(out / f"frame-{received:04d}.png")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            received += 1
        print(json.dumps({"received": received, "chunk": record}), flush=True)
        save()

    for seq in range(args.frames):
        # At most two in-flight chunks; no unbounded offline queue.
        wait_deadline = time.perf_counter() + 300
        while session.inflight_chunks() >= 2:
            result = session.wait_async_result(timeout=0.1)
            if result is not None:
                consume(result)
            if time.perf_counter() > wait_deadline:
                raise TimeoutError("No progress while waiting for an in-flight chunk")
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Input ended at {seq}, expected {args.frames} frames")
        submitted[seq] = time.perf_counter()
        results = session.push_frame(
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
            {"seq": seq, "t_capture_ms": seq * 1000 / fps},
        )
        for result in results:
            consume(result)
    session.flush_pending()
    deadline = time.perf_counter() + 300
    while received < args.frames:
        result = session.wait_async_result(timeout=0.1)
        if result is not None:
            consume(result)
        if time.perf_counter() > deadline:
            raise TimeoutError(f"Only {received}/{args.frames} frames returned")
    torch.cuda.synchronize()
    report["total_inference_delivery_s"] = time.perf_counter() - run_started
    report["generated_frames"] = received
    report["full_pipeline_generated_fps_including_first_chunk"] = (
        received / report["total_inference_delivery_s"]
    )
    report["status"] = "generated_not_yet_visually_inspected"
except BaseException as exc:
    report["status"] = "failed"
    report["error"] = repr(exc)
    raise
finally:
    report["peak_torch_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["peak_torch_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    report["elapsed_total_s"] = time.perf_counter() - started
    save()
    if session is not None:
        session.close()
    cap.release()
    if writer is not None:
        writer.release()
