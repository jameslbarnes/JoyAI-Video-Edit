"""Controlled real-model cache ablation. One model load; isolated sessions.

No live prompt changes. Same input frames/prompt/seeds in each arm. All changes
below are test-session-only; no monkey patch survives process exit.
"""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy'))
parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
parser.add_argument('--input', required=True)
parser.add_argument('--frames', type=int, default=1025)
parser.add_argument('--arms', nargs='+', default=['baseline', 'eager', 'no_kv', 'clamp_feedback'])
parser.add_argument('--prompt', default='Transform the flowing neon shapes into an intricate bioluminescent alien coral reef, with translucent organic tendrils, rich texture and cinematic depth. Preserve the source movement and composition while completely replacing the flat synthetic surfaces with detailed living forms.')
parser.add_argument('--jobs', help='Optional JSON list of {name, arm, input, prompt, frames}; one runtime load')
args = parser.parse_args()
if args.frames % 8 != 1 or args.frames < 9:
    parser.error('frames must be 8n+1')
strength_arms = {'blend0', 'blend0.1', 'blend0.3', 'blend0.5', 'blend1', 'blend2', 'blend3', 'blend5',
                 'blend10', 'blend25', 'blend50', 'blend75', 'ref50', 'ref75', 'ref90',
                 'strength_ramp', 'fine_ramp', 'ref_ramp'}
jobs = json.loads(Path(args.jobs).read_text()) if args.jobs else [
    {'name': arm, 'arm': arm, 'input': args.input, 'prompt': args.prompt, 'frames': args.frames} for arm in args.arms]
for job in jobs:
    if Path(job['name']).name != job['name'] or job['name'] in {'.', '..'}:
        parser.error('Job name must be one directory component')
    if job['frames'] % 8 != 1 or job['frames'] < 9:
        parser.error('Job frames must be 8n+1')
args.arms = [job['arm'] for job in jobs]
if any(a not in {'baseline', 'eager', 'no_kv', 'clamp_feedback', 'position8', 'position2', 'relative'} | strength_arms for a in args.arms):
    parser.error('unknown arm')
out = Path(args.out)
out.mkdir(parents=True, exist_ok=False)
import cv2
import numpy as np
import torch
from PIL import Image
from xvideo.serving.joyomni_streaming import JoyOmniRuntime, StreamingSettings

physical = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
physical_free = float(subprocess.check_output(['nvidia-smi', '-i', physical, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)) * 2**20
free, total = torch.cuda.mem_get_info()
budget = min(free, physical_free) - .75 * 2**30
if budget < 21 * 2**30:
    raise RuntimeError('Need >=21GiB physical free for isolated test; other workloads untouched')
torch.cuda.set_per_process_memory_fraction(budget / total)
cap = cv2.VideoCapture(args.input)
frames = []
while len(frames) < 4096:
    ok, bgr = cap.read()
    if not ok:
        break
    frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(bgr, (648, 360)), cv2.COLOR_BGR2RGB)))
cap.release()
if not frames:
    raise RuntimeError('No input frames')
frames[0].save(out / 'input-first.png')
report = {'args': vars(args), 'inputLoopFrames': len(frames), 'arms': {},
          'jobs': jobs,
          'precision': 'FP8 DiT / BF16 VAE', 'gpu': physical,
          'note': 'Offline throughput, no pacing; playback 16fps. Fixed input loops identically across arms.'}
def save():
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
save()
models = ROOT / 'tulip/models'
start_load = time.perf_counter()
runtime = JoyOmniRuntime.load(str(models / 'JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth'),
    vae_ckpt=str(models / 'JoyAI-Video-Edit/vae'), text_encoder_ckpt=str(models / 'MiMo-VL-7B-RL-2508'),
    device='cuda:0', vae_encode_device='cuda:0', vae_decode_device='cuda:0', vae_pseudo_device='cuda:0',
    postprocess_device='cuda:0', warmup_height=360, warmup_width=648)
report['load_s'] = time.perf_counter() - start_load
runtime.output_quality = 95
for job in jobs:
    arm = job['arm']
    if job['input'] != args.input:
        cap = cv2.VideoCapture(job['input'])
        frames = []
        while len(frames) < 4096:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(bgr, (648, 360)), cv2.COLOR_BGR2RGB)))
        cap.release()
        if not frames:
            raise RuntimeError('No frames for job ' + job['name'])
        args.input = job['input']
    os.environ['JOYOMNI_CUDA_GRAPH'] = '0' if arm in {'eager', 'no_kv', 'relative'} else '1'
    os.environ['JOYOMNI_SINK_REFRESH_CHUNKS'] = '0'
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    dest = out / job['name']
    dest.mkdir()
    temporal_limit = 8 if arm in strength_arms else {'position8': 8, 'position2': 2, 'relative': None}.get(arm, 128)
    session = runtime.create_v2v_session(job['prompt'], settings=StreamingSettings(
        height=360, width=648, num_inference_steps=2, seed=42, max_temporal_ids=temporal_limit,
        freeze_kv_on_static=False, profile_timings=True))
    if arm in strength_arms:
        from strength_controls import prepare_strength_inputs, strength_for_chunk
        import xvideo.serving.joyomni_streaming as streaming_module
        original_denoise = session._denoise_chunk
        original_randn = streaming_module.randn_tensor
        def strength_denoise(self, reference, **kw):
            chunk = kw['chunk_idx']
            blend, gain = strength_for_chunk(arm, chunk)
            def controlled_randn(*pos, **noise_kw):
                noise = original_randn(*pos, **noise_kw)
                return prepare_strength_inputs(noise, reference, blend=blend)[0]
            kw['profile'].update(shader_blend=blend, reference_gain=gain)
            # One session/DiT worker in this isolated process. Cover eager startup
            # chunks too; the graph path only starts once history is populated.
            with self.runtime.dit_lock, patch.object(streaming_module, 'randn_tensor', controlled_randn):
                return original_denoise(reference * gain if gain != 1 else reference, **kw)
        session._denoise_chunk = types.MethodType(strength_denoise, session)
    if arm == 'no_kv':
        session.global_sink_chunk = False
        session.local_window_size = 1
    original_decode = session._decode_chunk_pixels
    def diagnostic_decode(self, *pos, **kw):
        pixels = original_decode(*pos, **kw)
        p = kw['profile']
        p['decoded_finite'] = bool(torch.isfinite(pixels).all().item())
        p['decoded_abs_max'] = float(pixels.abs().max().item())
        p['decoded_clip_fraction'] = float((pixels.abs() > 1).float().mean().item())
        p['latent_rms'] = float(pos[0].float().square().mean().sqrt().item())
        return pixels
    session._decode_chunk_pixels = types.MethodType(diagnostic_decode, session)
    if arm == 'clamp_feedback':
        original_pseudo = session._encode_next_decode_pseudo_latent
        def clamped(self, pixels, **kw):
            return original_pseudo(pixels.clamp(-1, 1), **kw)
        session._encode_next_decode_pseudo_latent = types.MethodType(clamped, session)
    writer = cv2.VideoWriter(str(dest / 'output.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 16, (648, 360))
    if not writer.isOpened():
        raise RuntimeError('Video writer unavailable')
    result_report = report['arms'][job['name']] = {'chunks': [], 'status': 'running', 'job': job, 'inputLoopFrames': len(frames)}
    received = 0
    started = time.perf_counter()
    def consume(result):
        global received
        result_report['chunks'].append(result.profile)
        for data in result.jpegs or []:
            rgb = np.array(Image.open(io.BytesIO(data)).convert('RGB'))
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if received in {0, 8, 32, 64, 120, 128, 136, 248, 256, 264, 376, 384, 392, 512, 768, 1024}:
                Image.fromarray(rgb).save(dest / f'frame-{received:04d}.png')
            received += 1
        if received % 128 == 1:
            print(json.dumps({'arm': arm, 'received': received, 'elapsed_s': time.perf_counter()-started}), flush=True)
            save()
    try:
        for seq in range(job['frames']):
            deadline = time.perf_counter() + 90
            while session.inflight_chunks() >= 2:
                r = session.wait_async_result(.1)
                if r:
                    consume(r)
                if time.perf_counter() > deadline:
                    raise TimeoutError('No chunk progress')
            for r in session.push_frame(frames[seq % len(frames)], {'seq': seq, 't_capture_ms': seq * 62.5}):
                consume(r)
        deadline = time.perf_counter() + 90
        while received < job['frames']:
            r = session.wait_async_result(.1)
            if r:
                consume(r)
            if time.perf_counter() > deadline:
                raise TimeoutError('Incomplete output')
        result_report.update(status='generated_pending_visual_review', frames=received,
                             elapsed_s=time.perf_counter()-started)
        if arm in strength_arms and any('shader_blend' not in p for p in result_report['chunks']):
            raise RuntimeError('A strength chunk bypassed the instrumented denoise path')
    except Exception as exc:
        result_report.update(status='failed', error=repr(exc))
        raise
    finally:
        writer.release()
        session.close()
        save()
