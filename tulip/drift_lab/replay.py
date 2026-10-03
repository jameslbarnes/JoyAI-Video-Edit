"""Deterministic offline drift replay: production-faithful JoyAI sessions on a
looped recorded shader input, one arm per hypothesis, scored continuously.

Usage (one GPU, isolated from the live service's settings files):
  python tulip/drift_lab/replay.py --out DIR --input CLIP.mp4 --prompt "..." \
      --frames 6001 --arms baseline,no_decay,clamp_feedback,...

Writes per arm: output.mp4 (native 1152x648 @12fps), metrics.jsonl (one line
per second of output, aligned to the exact input frame), chunks.jsonl (per
chunk internal stats). Process-wide arms (precision/attention) need their
own process: pass --process-arm fp8_off|sage_off|graph_off.
"""
import argparse
import io
import json
import os
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--input", required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--frames", type=int, default=6001)
ap.add_argument("--arms", default="baseline")
ap.add_argument("--strength", type=float, default=0.34, help="combined-mode dial for arms that don't set one")
ap.add_argument("--decay", type=float, default=60.0, help="sink half-life s for arms that don't set one")
ap.add_argument("--process-arm", default="", choices=["", "fp8_off", "sage_off", "graph_off"])
ap.add_argument("--width", type=int, default=1152)
ap.add_argument("--height", type=int, default=648)
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--text-device", default="cuda:1", help="production: resident text encoder on the second visible GPU")
args = ap.parse_args()
if args.frames % 8 != 1:
    ap.error("frames must be 8n+1")

# Production flags (tulip/run-service.ps1), set before any xvideo import.
os.environ.update({
    "JOYOMNI_LOW_VRAM": "1", "JOYOMNI_TE_OFFLOAD": "0" if args.text_device != "cuda:0" else "1",
    "JOYOMNI_TEXT_DEVICE": args.text_device,
    "JOYOMNI_FP8_IMG": "1", "JOYOMNI_FP8_TXT": "1", "JOYOMNI_FP8_FAST_ACCUM": "1",
    "JOYOMNI_SAGE_ATTN": "1", "JOYOMNI_SKIP_LOAD_WARMUP": "1", "JOYOMNI_CUDA_GRAPH": "1",
    "TORCH_COMPILE_DISABLE": "1", "JOYOMNI_FIXED_PROMPT_GRAPH": "1", "JOYAI_COMPACT_PROMPT_GRAPHS": "1",
    "JOYAI_SINGLE_PROMPT_GRAPH": "0", "JOYOMNI_TRIM_PADDED_TEXT": "1", "JOYOMNI_WARMUP_NATIVE_ONLY": "1",
    "JOYOMNI_COMPILE_REFERENCE": "0", "JOYOMNI_SINK_REFRESH_CHUNKS": "0",
    "CUDA_HOME": str(ROOT / ".toolchain/Library"),
    "TRITON_CACHE_DIR": str(ROOT / "tulip/cache/triton"), "CUDA_CACHE_PATH": str(ROOT / "tulip/cache/cuda"),
})
os.environ["PATH"] = str(ROOT / ".toolchain/Library/bin") + os.pathsep + os.environ["PATH"]
if args.process_arm == "fp8_off":
    os.environ.update({"JOYOMNI_FP8_IMG": "0", "JOYOMNI_FP8_TXT": "0", "JOYOMNI_FP8_FAST_ACCUM": "0"})
elif args.process_arm == "sage_off":
    os.environ["JOYOMNI_SAGE_ATTN"] = "0"
elif args.process_arm == "graph_off":
    os.environ["JOYOMNI_CUDA_GRAPH"] = "0"

sys.path.insert(0, str(ROOT / "deploy"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from metrics import frame_metrics  # noqa: E402
from xvideo.serving.cache_decay import CacheDecay  # noqa: E402
from xvideo.serving.joyomni_streaming import JoyOmniRuntime, StreamingSettings  # noqa: E402
from xvideo.serving.shader_strength import ShaderStrength  # noqa: E402

# Arm -> session overrides. Each names one hypothesis.
ARMS = {
    "baseline": {},
    "no_decay": {"decay": 0.0},
    "decay15": {"decay": 15.0},
    "clamp_feedback": {"clamp_feedback": True},
    "no_kv": {"window": 1, "eager": True},
    "strength0": {"strength": 0.0},
    "strength15": {"strength": 0.15},
    "strength60": {"strength": 0.6},
    "steps3": {"steps": 3},
    "steps4": {"steps": 4},
    "position_free": {"max_temporal_ids": 128, "eager": True},
    "baseline_eager": {"eager": True},
    # Store the history KV from a slightly re-noised chunk: the next chunk then
    # treats inherited fine detail as uncertain instead of copying it.
    "hist_noise05": {"eager": True, "history_sigma": 0.05},
    "hist_noise10": {"eager": True, "history_sigma": 0.10},
    "hist_noise20": {"eager": True, "history_sigma": 0.20},
    "hist_noise15": {"eager": True, "history_sigma": 0.15},
    "hist_noise30": {"eager": True, "history_sigma": 0.30},
    # Same, but the scene-start sink is stored clean and never decays toward noisy
    # memory: noise only the previous-chunk path that carries the junk.
    "hn10_cleansink": {"eager": True, "history_sigma": 0.10, "clean_sink": True, "decay": 0.0},
    "hn15_cleansink": {"eager": True, "history_sigma": 0.15, "clean_sink": True, "decay": 0.0},
    "hn20_cleansink": {"eager": True, "history_sigma": 0.20, "clean_sink": True, "decay": 0.0},
    "hn20_cleansink_decay60": {"eager": True, "history_sigma": 0.20, "clean_sink": True},
    # Noise only the fine detail (latent minus a small blur): broad shapes and
    # motion carry over intact, the high-frequency junk path is randomized.
    "hfn20": {"eager": True, "history_sigma": 0.20, "clean_sink": True, "decay": 0.0, "hf_only": True},
    "hfn30": {"eager": True, "history_sigma": 0.30, "clean_sink": True, "decay": 0.0, "hf_only": True},
    "hfn30_t0": {"eager": True, "history_sigma": 0.30, "clean_sink": True, "decay": 0.0, "hf_only": True,
                 "store_timestep": 0.0},
    # Which memory entry carries the junk: [sink, current] vs [previous, current].
    "sink_only": {"eager": True, "window": 1},
    "no_sink": {"eager": True, "window": 2, "global_sink": False},
    # History filters: low-pass the stored chunk, or pull it toward the input.
    "hist_blur": {"eager": True, "history_blur": 0.5},
    "hist_ref20": {"eager": True, "history_ref": 0.2},
}
arms = [a for a in args.arms.split(",") if a]
# Dial sweeps: s45_n20 = strength 0.45, fine history noise 0.20 (n0 = off); all
# eager, clean sink, decay off, so only the dial and the noise differ.
import re as _re
for _a in arms:
    _m = _re.fullmatch(r"s(\d+)_n(\d+)", _a)
    if _m:
        _cfg = {"strength": int(_m.group(1)) / 100, "eager": True, "decay": 0.0, "clean_sink": True, "hf_only": True}
        if int(_m.group(2)):
            _cfg["history_sigma"] = int(_m.group(2)) / 100
        ARMS[_a] = _cfg
unknown = [a for a in arms if a not in ARMS]
if unknown:
    ap.error(f"unknown arms {unknown}; known {sorted(ARMS)}")

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
size = (args.width, args.height)
cap = cv2.VideoCapture(args.input)
src_fps = cap.get(cv2.CAP_PROP_FPS) or 24
step = max(1, round(src_fps / args.fps))
frames, n = [], 0
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    if n % step == 0:
        frames.append(cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    n += 1
cap.release()
if not frames:
    raise SystemExit("no input frames")
small = [cv2.resize(f, (480, 270), interpolation=cv2.INTER_AREA) for f in frames]

meta = {"args": vars(args), "inputFrames": len(frames), "inputStep": step, "cudaVisible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
(out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
models = ROOT / "tulip/models"
t0 = time.perf_counter()
runtime = JoyOmniRuntime.load(
    str(models / "JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth"),
    vae_ckpt=str(models / "JoyAI-Video-Edit/vae"), text_encoder_ckpt=str(models / "MiMo-VL-7B-RL-2508"),
    device="cuda:0", vae_encode_device="cuda:0", vae_decode_device="cuda:0", vae_pseudo_device="cuda:0",
    postprocess_device="cuda:0", warmup_height=args.height, warmup_width=args.width)
runtime.output_quality = 95
meta["load_s"] = round(time.perf_counter() - t0, 1)
(out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def latent_blur(x):
    """5x5 Gaussian (sigma ~1.2) over latent H, W per channel/frame, replicate-padded."""
    b, c, t, h, w = x.shape
    k1 = torch.tensor([1., 4., 6., 4., 1.], device=x.device) / 16.0
    k = (k1[:, None] * k1[None, :]).expand(c, 1, 5, 5)
    flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    out = torch.nn.functional.conv2d(torch.nn.functional.pad(flat, (2, 2, 2, 2), mode="replicate"), k, groups=c)
    return out.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def tensor_stats(t):
    t = t.float()
    return {"rms": float(t.square().mean().sqrt()), "absmax": float(t.abs().max())}


for arm in arms:
    cfg = {"strength": args.strength, "decay": args.decay, "steps": 2, "max_temporal_ids": 8, **ARMS[arm]}
    name = arm + (f"-{args.process_arm}" if args.process_arm else "")
    dest = out / name
    dest.mkdir(exist_ok=True)
    # Graph capture assumes the [sink, previous, current] window; arms that change
    # it run eager, compared against baseline_eager.
    os.environ["JOYOMNI_CUDA_GRAPH"] = "0" if cfg.get("eager") or args.process_arm == "graph_off" else "1"
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    session = runtime.create_v2v_session(args.prompt, settings=StreamingSettings(
        height=args.height, width=args.width, num_inference_steps=cfg["steps"], seed=42,
        max_temporal_ids=cfg["max_temporal_ids"], freeze_kv_on_static=False, profile_timings=False))
    session.shader_strength = ShaderStrength(mode="combined", value=cfg["strength"])
    session.cache_decay = CacheDecay("blend", cfg["decay"], native_fps=args.fps)
    if "window" in cfg:
        session.local_window_size = cfg["window"]
        # window 1 alone = no history at all (no_kv); window 1 + sink = sink_only.
        session.global_sink_chunk = cfg.get("global_sink", arm != "no_kv")
    if cfg.get("clamp_feedback"):
        original_pseudo = session._encode_next_decode_pseudo_latent

        def clamped(self, pixels, **kw):
            return original_pseudo(pixels.clamp(-1, 1), **kw)
        session._encode_next_decode_pseudo_latent = types.MethodType(clamped, session)
    pipeline = session.pipeline
    if cfg.get("history_blur") or cfg.get("history_ref"):
        blur_mix, ref_mix = float(cfg.get("history_blur") or 0), float(cfg.get("history_ref") or 0)
        original_store = pipeline._store_clean_chunk_kv_cache
        original_denoise_eager = session._denoise_chunk

        def remember_ref(self, ref_chunk_latent, **kw):
            self._drift_ref = ref_chunk_latent
            return original_denoise_eager(ref_chunk_latent, **kw)
        session._denoise_chunk = types.MethodType(remember_ref, session)

        def filtered_store(model, *, clean_chunk_latents, _blur=blur_mix, _ref=ref_mix, **kw):
            x = clean_chunk_latents
            if _blur:
                b, c, t, h, w = x.shape
                flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
                k = torch.tensor([1., 2., 1.], device=x.device)
                k = (k[:, None] * k[None, :]) / 16.0
                smooth = torch.nn.functional.conv2d(torch.nn.functional.pad(flat, (1, 1, 1, 1), mode="replicate"),
                                                    k.expand(c, 1, 3, 3), groups=c)
                smooth = smooth.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).to(x.dtype)
                x = torch.lerp(x, smooth, _blur)
            ref = getattr(session, "_drift_ref", None)
            if _ref and ref is not None and ref.shape == x.shape:
                x = torch.lerp(x, ref.to(x.dtype), _ref)
            return original_store(model, clean_chunk_latents=x, **kw)
        pipeline._store_clean_chunk_kv_cache = filtered_store
    if cfg.get("history_sigma"):
        sigma = float(cfg["history_sigma"])
        noise_gen = torch.Generator(device="cuda:0").manual_seed(1234)

        def noisy_store(model, *, clean_chunk_latents, chunk_temporal_ids, prompt_embeds, active_chunk_id,
                        history_chunk_ids, pre_rope=False, cached_temporal_ids=None, store_mode="store",
                        timestep_value=0.0, _sigma=sigma, _gen=noise_gen, _clean_sink=bool(cfg.get("clean_sink"))):
            if _clean_sink and active_chunk_id == 0:
                _sigma = 0.0
            noise = torch.randn(clean_chunk_latents.shape, generator=_gen, device=clean_chunk_latents.device,
                                dtype=torch.float32).to(clean_chunk_latents.dtype)
            if cfg.get("hf_only") and _sigma > 0:
                clean32 = clean_chunk_latents.float()
                high_clean = clean32 - latent_blur(clean32)
                n32 = noise.float()
                high_noise = n32 - latent_blur(n32)
                high_noise = high_noise / high_noise.std().clamp_min(1e-6) * high_clean.std()
                noisy = (clean32 + _sigma * (high_noise - high_clean)).to(clean_chunk_latents.dtype)
            else:
                noisy = (1 - _sigma) * clean_chunk_latents + _sigma * noise
            if "store_timestep" in cfg:
                _sigma = float(cfg["store_timestep"]) / 1000.0
            selected = ([pipeline._kv_cache_memory_id("clean", cid) for cid in (history_chunk_ids or [])]
                        if store_mode == "reuse_store" else [])
            with model.cache_context("cond"):
                model(hidden_states=noisy,
                      timestep=torch.full((noisy.shape[0],), _sigma * 1000.0, device=noisy.device, dtype=noisy.dtype),
                      encoder_hidden_states=prompt_embeds, current_temporal_ids=chunk_temporal_ids,
                      cached_temporal_ids=cached_temporal_ids, kv_cache_mode=store_mode, kv_cache_scope="cond",
                      kv_cache_chunk_id=pipeline._kv_cache_memory_id("clean", active_chunk_id),
                      kv_cache_selected_chunk_ids=selected, kv_cache_pre_rope=pre_rope, skip_text_stream=True)
        pipeline._store_clean_chunk_kv_cache = noisy_store  # instance attr; removed after the arm
    chunk_log = (dest / "chunks.jsonl").open("w", encoding="utf-8")
    original_decode = session._decode_chunk_pixels

    def diagnostic_decode(self, *pos, **kw):
        pixels = original_decode(*pos, **kw)
        p = kw["profile"]
        p["latent"] = tensor_stats(pos[0])
        p["pixels"] = tensor_stats(pixels)
        p["clipFraction"] = float((pixels.abs() > 1).float().mean())
        store = self.pipeline.transformer._inference_kv_cache.get("cond", {})
        sink = store.get(self.pipeline._kv_cache_memory_id("clean", 0))
        if sink:
            mid = sink[len(sink) // 2]
            p["sinkKey"], p["sinkValue"] = tensor_stats(mid["key"]), tensor_stats(mid["value"])
        return pixels
    session._decode_chunk_pixels = types.MethodType(diagnostic_decode, session)

    writer = cv2.VideoWriter(str(dest / "output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    metrics_log = (dest / "metrics.jsonl").open("w", encoding="utf-8")
    state = {"received": 0, "prev": None}
    started = time.perf_counter()

    def consume(result):
        prof = result.profile
        chunk_log.write(json.dumps({k: prof.get(k) for k in ("chunk_idx", "latent", "pixels", "clipFraction",
                                                             "sinkKey", "sinkValue", "temporal_position")}) + "\n")
        for data in result.jpegs or []:
            rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            i = state["received"]
            if i % args.fps == 0:  # once per second of output, against the exact input frame
                o_small = cv2.resize(rgb, (480, 270), interpolation=cv2.INTER_AREA)
                idx = i % len(small)
                prev_in = small[(i - args.fps) % len(small)] if state["prev"] is not None else None
                m = frame_metrics(o_small, small[idx], state["prev"], prev_in)
                m.update(frame=i, t=round(i / args.fps, 2))
                metrics_log.write(json.dumps(m) + "\n")
                state["prev"] = o_small
                if i % (args.fps * 30) == 0:
                    cv2.imwrite(str(dest / f"f{i:06d}.jpg"), cv2.cvtColor(np.hstack([o_small, small[idx]]), cv2.COLOR_RGB2BGR))
            state["received"] += 1
        if state["received"] % (args.fps * 60) < 8:
            metrics_log.flush()
            chunk_log.flush()
            print(json.dumps({"arm": name, "frames": state["received"], "elapsed_s": round(time.perf_counter() - started)}), flush=True)

    status = "ok"
    try:
        for seq in range(args.frames):
            deadline = time.perf_counter() + 120
            while session.inflight_chunks() >= 2:
                r = session.wait_async_result(.1)
                if r:
                    consume(r)
                if time.perf_counter() > deadline:
                    raise TimeoutError("no chunk progress")
            img = Image.fromarray(frames[seq % len(frames)])
            for r in session.push_frame(img, {"seq": seq, "t_capture_ms": seq * 1000 / args.fps}):
                consume(r)
        deadline = time.perf_counter() + 120
        while state["received"] < args.frames - 8:
            r = session.wait_async_result(.1)
            if r:
                consume(r)
            if time.perf_counter() > deadline:
                raise TimeoutError("incomplete output")
    except Exception as exc:  # noqa: BLE001
        status = f"failed: {exc!r}"
        raise
    finally:
        writer.release()
        metrics_log.close()
        chunk_log.close()
        session.close()
        pipeline.__dict__.pop("_store_clean_chunk_kv_cache", None)
        # Arms share one runtime: drop graph pools and cached blocks so memory
        # from earlier arms can't starve later ones (batch1 OOM'd on arm 5).
        runtime.graph_runners.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        (dest / "arm.json").write_text(json.dumps({"arm": name, "config": cfg, "status": status,
                                                   "frames": state["received"],
                                                   "elapsed_s": round(time.perf_counter() - started)}, indent=2))
