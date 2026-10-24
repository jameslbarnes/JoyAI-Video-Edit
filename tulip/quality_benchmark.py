"""Matched native-resolution trials; no live publishers or service mutations."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy'))
parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
parser.add_argument('--frames', type=int, default=193)
parser.add_argument('--target-native-fps', type=float, default=16)
parser.add_argument('--sizes', default='648x360,768x432,840x480,864x480')
parser.add_argument('--profile-timings', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--vae-device', default='cuda:0', choices=['cuda:0','cuda:1'])
parser.add_argument('--eager-after-first', action='store_true',
    help='Measure larger sizes without new compilation, with DiT stage timing to bound throughput')
args = parser.parse_args()
sizes = [tuple(map(int, size.split('x'))) for size in args.sizes.split(',')]
out = Path(args.out)
import cv2
import numpy as np
import torch
from PIL import Image
from xvideo.serving.joyomni_streaming import JoyOmniRuntime, StreamingSettings

cv2.setNumThreads(1)
physical = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
physical_free = float(subprocess.check_output(['nvidia-smi', '-i', physical,
    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)) * 2**20
free, total = torch.cuda.mem_get_info()
budget = min(free, physical_free) - .75 * 2**30
if budget < 21 * 2**30:
    raise RuntimeError('Insufficient physical headroom for isolated trial')
torch.cuda.set_per_process_memory_fraction(budget / total)
source = ROOT.parent / 'sana-streaming/tulip/evidence/shader-ab/input.mp4'
cap = cv2.VideoCapture(str(source))
inputs = []
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    inputs.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
cap.release()
if not inputs:
    raise RuntimeError('No benchmark source frames')
preset = json.loads((ROOT / 'tulip/presets/etherea-continuous-v2.json').read_text())
report = {'source': str(source), 'source_frames': len(inputs), 'prompt': preset['prompt'],
    'seed': 42, 'steps': 2, 'max_temporal_ids': 8, 'budget_gib': budget / 2**30,
    'profile_timings': args.profile_timings, 'vae_device': args.vae_device,
    'compile_disabled': os.environ.get('TORCH_COMPILE_DISABLE'),
    'fixed_prompt_graph': os.environ.get('JOYOMNI_FIXED_PROMPT_GRAPH','1'),
    'note': 'Unpaced offline native throughput; no RIFE/SR or live transport in native timing.', 'trials': []}

def save():
    (out / 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

save()
models = ROOT / 'tulip/models'
started = time.perf_counter()
runtime = JoyOmniRuntime.load(str(models / 'JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth'),
    vae_ckpt=str(models / 'JoyAI-Video-Edit/vae'),
    text_encoder_ckpt=str(models / 'MiMo-VL-7B-RL-2508'),
    device='cuda:0', vae_encode_device=args.vae_device, vae_decode_device=args.vae_device,
    vae_pseudo_device=args.vae_device, postprocess_device=args.vae_device,
    warmup_height=sizes[0][1], warmup_width=sizes[0][0])
runtime.raw_presentation_output = True
report['load_s'] = time.perf_counter() - started
save()
for size_index, (width, height) in enumerate(sizes):
    eager_larger = args.eager_after_first and size_index > 0
    if eager_larger:
        for vae in (runtime.pipeline.vae, runtime.decode_vae, runtime.pseudo_encode_vae):
            for name in ('_encode','_decode'):
                method=getattr(vae,name,None)
                original=getattr(method,'_torchdynamo_orig_callable',None)
                if original is not None: setattr(vae,name,original)
    profile_timings=args.profile_timings or eager_larger
    dest = out / f'{width}x{height}'
    dest.mkdir(exist_ok=True)
    session = None
    result = {'width': width, 'height': height, 'chunks': [], 'status': 'starting',
              'profile_timings':profile_timings,
              'vae_compilation':os.environ.get('TORCH_COMPILE_DISABLE')!='1' and not eager_larger}
    report['trials'].append(result)
    frames = []
    writer = cv2.VideoWriter(str(dest / 'native.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 16, (width, height))
    if not writer.isOpened():
        raise RuntimeError('Cannot create benchmark video')
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    try:
        session = runtime.create_v2v_session(preset['prompt'], settings=StreamingSettings(
            height=height, width=width, num_inference_steps=2, seed=42,
            max_temporal_ids=8, freeze_kv_on_static=False, profile_timings=profile_timings))
        start = time.perf_counter()
        def consume(chunk):
            now = time.perf_counter()
            count = chunk.valid_count if chunk.valid_count is not None else len(chunk.jpegs)
            result['chunks'].append({'at_s': now-start, 'count': count,
                'profile': {k:v for k,v in chunk.profile.items() if not isinstance(v, dict)}})
            for pixels in chunk.jpegs[:count]:
                if not isinstance(pixels, np.ndarray) or pixels.shape != (height,width,3):
                    raise RuntimeError('Unexpected native output geometry')
                n = len(frames)
                frames.append(pixels)
                writer.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
                if n in (0,32,64,96,128,160,192):
                    Image.fromarray(pixels).save(dest / f'frame-{n:03}.png')
            if len(frames) % 32 == 1:
                print(json.dumps({'native': f'{width}x{height}', 'frames': len(frames), 'seconds': now-start}), flush=True)
                save()
        for seq in range(args.frames):
            deadline = time.perf_counter()+120
            while session.inflight_chunks() >= 2:
                chunk = session.wait_async_result(.1)
                if chunk is not None: consume(chunk)
                if time.perf_counter()>deadline: raise TimeoutError('Native inference stalled')
            # Mirror-loop the same source to avoid a hard temporal jump.
            period = max(1,2*len(inputs)-2)
            index = seq % period
            if index >= len(inputs): index = period-index
            for chunk in session.push_frame(inputs[index], {'seq':seq,'t_capture_ms':seq*62.5}):
                consume(chunk)
        session.flush_pending()
        deadline = time.perf_counter()+120
        while len(frames)<args.frames:
            chunk = session.wait_async_result(.1)
            if chunk is not None: consume(chunk)
            if time.perf_counter()>deadline: raise TimeoutError('Incomplete native output')
        # Exclude startup and the first four full chunks from throughput.
        warm = result['chunks'][5:]
        base = result['chunks'][4]['at_s']
        gaps = [c['at_s'] - (warm[i-1]['at_s'] if i else base) for i,c in enumerate(warm)]
        from xvideo.serving.cadence_metrics import interval_metrics
        result['chunkTiming'] = interval_metrics([gap*1000 for gap in gaps], 8/args.target_native_fps*1000)
        result['cadenceScope'] = 'Native chunk completion only; live projector acceptance is required before claiming smooth playback.'
        result.update(status='generated_pending_visual_review', native_fps=sum(c['count'] for c in warm)/(warm[-1]['at_s']-base),
            chunk_median_ms=statistics.median(gaps)*1000, chunk_p95_ms=float(np.percentile(gaps,95))*1000,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_allocated_gib_by_device={str(i):torch.cuda.max_memory_allocated(i)/2**30
                for i in range(torch.cuda.device_count())},
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        np.save(dest/'native-last32.npy', np.stack(frames[-32:]))
        print(json.dumps({k:v for k,v in result.items() if k!='chunks'}), flush=True)
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        print(json.dumps({'failed':f'{width}x{height}','error':repr(exc)}),flush=True)
    finally:
        writer.release()
        if session is not None: session.close()
        session = None
        runtime.graph_runners.clear()
        runtime.pipeline.transformer.reset_inference_kv_cache()
        frames.clear()
        gc.collect()
        torch.cuda.empty_cache()
        save()
