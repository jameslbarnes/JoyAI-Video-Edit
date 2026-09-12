"""Bounded presentation-only 2x SR. No output ever feeds the model cache.

Pinned official SRVGG source/checkpoints are read-only; no BasicSR installation.
One frame on the SR GPU at a time, one chunk in the existing output pump. A
failure latches SR off and returns native frames, rather than killing generation.
"""
import ast
import hashlib
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
REVISION = 'a4abfb2979a7bbff3f69f58f58ae324608821e27'
HASHES = {
    'realesr-general-x4v3': '8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292',
    'realesr-general-wdn-x4v3': '1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca',
}


class PixelUpscaler:
    def __init__(self, device, denoise=.5):
        if not isinstance(denoise, (int, float)) or not np.isfinite(denoise) or not 0 <= denoise <= 1:
            raise ValueError('denoise must be finite in [0,1]')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Output SR requires an explicitly selected CUDA device')
        repo = ROOT / 'tulip/research/Real-ESRGAN'
        revision = subprocess.check_output(['git', '-c', f'safe.directory={repo.as_posix()}',
            '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != REVISION:
            raise ValueError('Unexpected Real-ESRGAN source revision')
        source = repo / 'realesrgan/archs/srvgg_arch.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
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
                raise ValueError(f'SR checkpoint hash mismatch: {name}')
            state = torch.load(path, map_location='cpu', weights_only=True)
            weights.append(state.get('params_ema', state.get('params', state)))
        state = {k: weights[0][k] * denoise + weights[1][k] * (1-denoise) for k in weights[0]}
        self.model = namespace['SRVGGNetCompact'](num_in_ch=3, num_out_ch=3,
            num_feat=64, num_conv=32, upscale=4, act_type='prelu').eval()
        self.model.load_state_dict(state, strict=True)
        with torch.cuda.device(self.device):
            self.model.to(self.device, torch.float16, memory_format=torch.channels_last)
            self.stream = torch.cuda.Stream(device=self.device)

    @torch.inference_mode()
    def process(self, frame):
        if frame.shape != (360, 648, 3) or frame.dtype != np.uint8:
            raise ValueError('SR supports only native 648x360 RGB uint8 frames')
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            image = torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).to(
                self.device, torch.float16, memory_format=torch.channels_last) / 255
            result = self.model(image).clamp_(0, 1)
            result = F.interpolate(result.float(), size=(720, 1296), mode='bicubic',
                                   align_corners=False, antialias=True)
            # Blocking D2H copy synchronizes this stream only, not JoyAI's GPU.
            return result.clamp_(0, 1).mul_(255).byte()[0].permute(1, 2, 0).cpu().numpy()


class OutputUpscale:
    """Resident model; serialized configuration/work, no additional work queue."""
    def __init__(self, device, denoise=.5):
        self.device, self.denoise = str(device), denoise
        # The four bounded JPEG workers provide CPU parallelism. Avoid each
        # color conversion spawning another machine-wide OpenCV worker team.
        # Process-local only; no changes to Easel/Flux/SDK environments.
        cv2.setNumThreads(1)
        self.model = None
        self.enabled = True
        self.error = None
        self.lock = threading.Lock()
        self.load_s = self.warmup_s = None
        self.last = {}
        self.encoder = ThreadPoolExecutor(max_workers=4, thread_name_prefix='joyai-jpeg')

    def close(self):
        self.encoder.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def encode(frame, quality):
        ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError('Final output JPEG encoding failed')
        return jpeg.tobytes()

    def status(self):
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        index = int(self.device.split(':')[-1])
        return {'supported': True, 'enabled': self.enabled, 'loaded': self.model is not None,
                'device': self.device, 'physical_gpu': visible[index] if index < len(visible) else None,
                'precision': 'float16', 'scale': 2 if self.enabled else 1,
                'denoise': self.denoise, 'error': self.error, 'load_s': self.load_s,
                'warmup_s': self.warmup_s, 'last': self.last}

    def set_enabled(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError('enabled must be boolean')
        with self.lock:
            if enabled and self.error:
                raise ValueError(f'SR disabled after failure; restart service to retry: {self.error}')
            self.enabled = enabled
        return self.status()

    def warmup(self):
        with self.lock:
            try:
                start = time.perf_counter()
                self.model = PixelUpscaler(self.device, self.denoise)
                self.load_s = time.perf_counter() - start
                start = time.perf_counter()
                self.model.process(np.zeros((360, 648, 3), dtype=np.uint8))
                self.warmup_s = time.perf_counter() - start
            except Exception as exc:
                self.error, self.enabled, self.model = str(exc), False, None
                print(f'#####[SR] Native-output fallback: {exc!r}', flush=True)

    def process(self, frames, profile, quality=95):
        if not frames or len(frames) > 16:
            raise ValueError('SR accepts one bounded presentation chunk (1..16 frames)')
        if any(not isinstance(f, np.ndarray) or f.shape != (360, 648, 3) or f.dtype != np.uint8 for f in frames):
            raise ValueError('SR expects matched 648x360 RGB uint8 input')
        start = time.perf_counter()
        with self.lock:
            sr_start = time.perf_counter()
            pixels = frames
            if self.enabled:
                try:
                    pixels = [self.model.process(f) for f in frames]
                except Exception as exc:
                    self.enabled, self.error = False, str(exc)
                    pixels = frames
                    print(f'#####[SR] Native-output fallback: {exc!r}', flush=True)
            sr_s = time.perf_counter() - sr_start
            jpeg_start = time.perf_counter()
            # At most 16 tasks, consumed before this call returns. No rolling
            # queue: a slow encoder backpressures the existing chunk pump.
            output = list(self.encoder.map(self.encode, pixels,
                [max(1, min(100, int(quality)))] * len(pixels)))
            self.last = {'sr_s': sr_s, 'jpeg_s': time.perf_counter()-jpeg_start,
                         'total_s': time.perf_counter()-start, 'count': len(output),
                         'width': pixels[0].shape[1], 'height': pixels[0].shape[0]}
            profile['upscale'] = {**self.last, 'enabled': self.enabled, 'error': self.error,
                                  'device': self.device, 'precision': 'float16'}
            return output
