"""Bounded GPU-0 output-only SR trial. Does not alter the live JoyAI/Flux services.

Uses the pinned official SRVGG source, removing only BasicSR's registry decorator
in memory. No dependency installation or upstream source edits. Safe weight load.
"""
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'a4abfb2979a7bbff3f69f58f58ae324608821e27'
HASHES = {
    'realesr-general-x4v3': '8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292',
    'realesr-general-wdn-x4v3': '1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca',
}
FLUX_REVISION='bf4c169edfc813424d14d8427c4bc37c7530f89b'
FLUX_HASHES={'flow_upscaler.safetensors':'91eb93b40179441e569e01fe04f20fd9b951e434f88c84dd0ba523f830a81839',
             'taef2.safetensors':'701d31c0ecbef3b609fe236e2e3a65ffb17f3a94237acdd771e8ea3194df557a'}


def request(url):
    with urllib.request.urlopen(url, timeout=4) as response:
        return json.load(response)


def live_status():
    streams = request('http://127.0.0.1:8793/control/multistream/list')['streams']
    joy = next(s for s in streams if s['type'] == 'joyai')
    stats = joy['runtime']['stats']
    return {'at': time.time(), 'status': joy['status'], 'epoch': stats['epoch'],
            **{k: stats.get(k) for k in ('modelFramesDisplayed', 'captureToCanvasMs',
               'droppedCaptureTicks', 'droppedOutputFrames', 'bufferedOutputFrames')},
            'input': joy.get('input'), 'fluxPaused': request('http://127.0.0.1:7865/health')['runtime']['paused']}


def capture(seconds, fps):
    from websockets.sync.client import connect
    socket = connect('ws://127.0.0.1:7779/?video=0', open_timeout=5, close_timeout=1, max_size=8*1024*1024)
    receiver = 'upscale-trial-' + str(os.getpid())
    frames, stamps = [], []
    try:
        socket.send(json.dumps({'id': 1, 'action': 'ndi_keyed_receive_start', 'params': {
            'receiverId': receiver, 'sourceName': 'TULIP (ETHEREA JoyAI)',
            'maxWidth': 648, 'maxHeight': 360, 'fps': fps, 'bandwidth': 'highest'}}))
        deadline = time.monotonic() + seconds + 15
        while len(frames) < round(seconds * fps) and time.monotonic() < deadline:
            message = socket.recv(timeout=5)
            if isinstance(message, str):
                reply = json.loads(message)
                if reply.get('error'):
                    raise RuntimeError(reply['error'])
                continue
            if len(message) < 10 or message[0] != 4:
                continue
            n = message[1]
            if message[2:2+n].decode() != receiver:
                continue
            w, h = struct.unpack_from('<II', message, 2+n)
            frame = np.frombuffer(message[10+n:], np.uint8).reshape(h, w, 4)[:, :, :3].copy()
            if (w, h) != (648, 360):
                raise ValueError(f'Expected native JoyAI 648x360, got {w}x{h}')
            frames.append(frame)
            stamps.append(time.time())
    finally:
        socket.close()
    if len(frames) < 2:
        raise RuntimeError('No actual JoyAI video captured')
    return frames, stamps


class Upscaler:
    def __init__(self, denoise=.5):
        if not isinstance(denoise,(int,float)) or not np.isfinite(denoise) or not 0<=denoise<=1:
            raise ValueError('denoise must be finite in [0,1]')
        repo = ROOT / 'tulip/research/Real-ESRGAN'
        revision = subprocess.check_output(['git', '-c', f'safe.directory={repo.as_posix()}',
            '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != REVISION:
            raise ValueError('Unexpected Real-ESRGAN source revision')
        source = repo / 'realesrgan/archs/srvgg_arch.py'
        tree = ast.parse(source.read_text())
        tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom) or n.module != 'basicsr.utils.registry']
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                node.decorator_list = []
        namespace = {}
        exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), namespace)
        weights = []
        for name, expected in HASHES.items():
            path = ROOT / f'tulip/models/realesrgan/{name}.pth'
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Checkpoint hash mismatch')
            state = torch.load(path, map_location='cpu', weights_only=True)
            weights.append(state.get('params_ema', state.get('params', state)))
        state = {k: weights[0][k]*denoise + weights[1][k]*(1-denoise) for k in weights[0]}
        self.model = namespace['SRVGGNetCompact'](num_in_ch=3, num_out_ch=3,
            num_feat=64, num_conv=32, upscale=4, act_type='prelu').eval()
        self.model.load_state_dict(state, strict=True)
        self.model.to('cuda:0', torch.float16, memory_format=torch.channels_last)

    @torch.inference_mode()
    def process(self, frame):
        begin = time.perf_counter()
        image = torch.from_numpy(frame.copy()).permute(2,0,1).unsqueeze(0).to(
            'cuda:0', torch.float16, memory_format=torch.channels_last) / 255
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = self.model(image).clamp_(0,1)
        end.record()
        # Native network is 4x. Reduce on GPU to 2x before device->host delivery.
        result = F.interpolate(result.float(), size=(frame.shape[0]*2,frame.shape[1]*2),
                               mode='bicubic', align_corners=False, antialias=True)
        pixels = result.clamp_(0,1).mul_(255).byte()[0].permute(1,2,0).cpu().numpy()
        return pixels, {'gpu_model_ms': start.elapsed_time(end),
                        'pipeline_ms': (time.perf_counter()-begin)*1000}


class FluxLatentUpscaler:
    """Experimental RGB->Flux tiny VAE->existing Flow upscaler->RGB.

    NOT JoyAI latent super-resolution. The extra encode/decode is measured and
    can lose detail. No Flux transformer/text encoder/session is loaded or run.
    """
    def __init__(self):
        root = ROOT.parent / 'FluxRT'
        for subdir, name in [('FlowUpscaler','flow_upscaler.safetensors'),('taef2','taef2.safetensors')]:
            if hashlib.sha256((root/subdir/name).read_bytes()).hexdigest()!=FLUX_HASHES[name]:
                raise ValueError('Unexpected Flux upscaler checkpoint')
        def module(name, path):
            spec=importlib.util.spec_from_file_location(name,str(path))
            result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
            return result
        tiny=module('trial_flux_tiny',root/'src/fluxrt/stream_processor/flux_tiny_vae.py')
        unet=module('trial_flow_unet',root/'src/fluxrt/flow_upscaler/upscaler_unet.py')
        self.vae=tiny.DiffusersTAEF2Wrapper(str(root/'taef2/taef2.safetensors')).eval().to('cuda:0',torch.bfloat16)
        from safetensors.torch import load_file
        self.unet=unet.UpscalerUNet().eval()
        self.unet.load_state_dict(load_file(str(root/'FlowUpscaler/flow_upscaler.safetensors')))
        self.unet.to('cuda:0',torch.bfloat16)
        path=root/'src/fluxrt/flow_upscaler/flow_upscaler_pipeline.py'
        tree=ast.parse(path.read_text())
        tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom) or not n.module.startswith('fluxrt.')]
        namespace={'UpscalerUNet':unet.UpscalerUNet}
        exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),namespace)
        from diffusers import FlowMatchEulerDiscreteScheduler
        cfg=json.loads((ROOT.parent.parent/'.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294/scheduler/scheduler_config.json').read_text())
        self.pipe=namespace['FlowUpscalerPipeline'](self.unet,FlowMatchEulerDiscreteScheduler.from_config(cfg))

    @torch.inference_mode()
    def process(self, frame):
        begin=time.perf_counter();h,w=frame.shape[:2]
        image=torch.from_numpy(frame.copy()).permute(2,0,1).unsqueeze(0).to('cuda:0',torch.bfloat16)/255
        image=F.pad(image,(0,(-w)%16,0,(-h)%16),mode='replicate')
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        latents=self.vae.encode(image*2-1).latent_dist.sample()
        result=self.pipe(latents,generator=torch.Generator(device='cuda:0').manual_seed(52))
        result=self.vae.decode(result,return_dict=False)[0]*.5+.5
        end.record()
        pixels=result[:,:,:h*2,:w*2].clamp(0,1).mul(255).byte()[0].permute(1,2,0).cpu().numpy()
        return pixels,{'gpu_model_ms':start.elapsed_time(end),'pipeline_ms':(time.perf_counter()-begin)*1000}


def encode(path, frames, fps, ffmpeg):
    h,w = frames[0].shape[:2]
    process = subprocess.Popen([ffmpeg, '-hide_banner','-loglevel','error','-nostdin',
        '-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fps),'-i','-',
        '-an','-c:v','libx264','-preset','fast','-crf','15','-pix_fmt','yuv420p',str(path)],
        stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=6)
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--denoise', type=float, default=.5)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--kind', choices=['pixel','flux-latent'], default='pixel')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a NEW evidence directory')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.12, 0)
    torch.backends.cudnn.benchmark = True
    baseline = [live_status()]
    if args.input:
        reader=cv2.VideoCapture(str(args.input)); frames=[]
        for _ in range(round(args.seconds*args.fps)):
            ok, frame=reader.read()
            if not ok: break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        reader.release(); stamps=[]
    else:
        frames, stamps = capture(args.seconds, args.fps)
    baseline.append(live_status())
    print(json.dumps({'captured':len(frames),'baseline':baseline}), flush=True)
    started = time.perf_counter()
    model = Upscaler(args.denoise) if args.kind == 'pixel' else FluxLatentUpscaler()
    load_s = time.perf_counter()-started
    warmup=[]
    for _ in range(4):
        _, timing = model.process(frames[0]); warmup.append(timing)
    torch.cuda.reset_peak_memory_stats()
    results, timings, live = [], [], [live_status()]
    for i, frame in enumerate(frames):
        output, timing=model.process(frame)
        if not np.isfinite(output).all() or output.std()<1:
            raise RuntimeError('Invalid upscaled output')
        results.append(output);timings.append(timing)
        if i % 16 == 0: live.append(live_status())
    live.append(live_status())
    report={'source':REVISION if args.kind=='pixel' else FLUX_REVISION,
        'hashes':HASHES if args.kind=='pixel' else FLUX_HASHES,'device':torch.cuda.get_device_name(),
        'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES','not specified'),
        'kind':args.kind,'precision':'FP16' if args.kind == 'pixel' else 'BF16', 'denoise':args.denoise,'torch':torch.__version__,
        'capture_fps':args.fps, 'capture_stamps':stamps,'frames':len(frames),
        'load_s':load_s,'warmup':warmup,'timings':timings,'baseline':baseline,'during':live,
        'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20}
    for key in timings[0]:
        values=[t[key] for t in timings]
        report[key]={'mean':float(np.mean(values)),'p50':float(np.percentile(values,50)),
                     'p95':float(np.percentile(values,95))}
    report['pipeline_processed_fps']=1000/report['pipeline_ms']['mean']
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    ffmpeg=str(ROOT.parent/'easel/external/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe')
    encode(args.out/'input.mp4',frames,args.fps,ffmpeg)
    encode(args.out/'sr-2x.mp4',results,args.fps,ffmpeg)
    baseline_frames=[cv2.resize(f,(1296,720),interpolation=cv2.INTER_LANCZOS4) for f in frames]
    encode(args.out/'lanczos-2x.mp4',baseline_frames,args.fps,ffmpeg)
    for i in (0,len(frames)//2,len(frames)-1):
        Image.fromarray(frames[i]).save(args.out/f'input-{i}.png')
        Image.fromarray(results[i]).save(args.out/f'sr-{i}.png')
        canvas=Image.new('RGB',(1296*2,752))
        canvas.paste(Image.fromarray(baseline_frames[i]),(0,32))
        canvas.paste(Image.fromarray(results[i]),(1296,32))
        draw=ImageDraw.Draw(canvas); draw.text((12,8),'Lanczos 2x',(255,255,255));draw.text((1308,8),args.kind+' SR 2x',(255,255,255))
        canvas.save(args.out/f'comparison-{i}.png')
    print(json.dumps({k:v for k,v in report.items() if k not in ('timings','during','baseline','capture_stamps','hashes')}),flush=True)


if __name__ == '__main__':
    main()
