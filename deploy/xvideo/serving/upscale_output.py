"""Bounded presentation-only SR, including UHD GPU video. No output feeds the model cache.

Pinned official SRVGG source/checkpoints are read-only; no BasicSR installation.
One frame on the SR GPU at a time, one chunk in the existing output pump. A
failure latches SR off and returns native frames, rather than killing generation.
"""
import ast
import hashlib
import json
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


def gpu_output_size(height):
    """GPU delivery uses exact UHD at 2160p; older presets keep their geometry."""
    if type(height) is not int or height not in (1080, 1440, 2160):
        raise ValueError('GPU video height must be 1080, 1440 or 2160')
    return (height, 3840 if height == 2160 else height * 9 // 5)


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
            # This GPU also hosts prompt encoding. Keep the paced video queue
            # supplied while that background work is running.
            self.stream = torch.cuda.Stream(device=self.device, priority=-1)
        self.graph = None
        self.graph_key = None
        self.graph_error = None

    def _render_gpu(self, image, output_size):
        result = self.model(image).clamp_(0, 1)
        result = F.interpolate(result.float(), size=output_size, mode='bicubic',
                               align_corners=False, antialias=True)
        return result.clamp_(0, 1).mul_(255).byte()[0].permute(1, 2, 0).contiguous()

    @torch.no_grad()
    def warmup_graph(self, native_size, output_size):
        """Capture once at service startup; never capture during a live frame."""
        if os.environ.get('JOYAI_SR_GRAPH', '0') != '1':
            return
        try:
            # CUDA graph RNG bookkeeping must remain writable by later DiT
            # captures outside inference_mode, including on another GPU.
            with torch.inference_mode(False), torch.no_grad(), torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                image = torch.zeros((1, 3, *native_size), device=self.device,
                    dtype=torch.float16).contiguous(memory_format=torch.channels_last)
                for _ in range(3):
                    self._render_gpu(image, output_size)
                self.stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.stream, capture_error_mode='thread_local'):
                    pixels = self._render_gpu(image, output_size)
                self.stream.synchronize()
                self.graph_input, self.graph_pixels = image, pixels
                self.graph_key = (tuple(native_size), tuple(output_size))
                self.graph = graph
                print(f'#####[SR] captured presentation graph {self.graph_key}', flush=True)
        except Exception as exc:
            # Eager computes identical SR, preserving quality on unsupported setups.
            self.graph_error = str(exc)
            print(f'#####[SR] graph unavailable; retaining eager SR: {exc!r}', flush=True)

    @torch.inference_mode()
    def process(self, frame, output_size=(720, 1296)):
        gpu_frame = self.process_gpu(frame, output_size)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            return gpu_frame.wait().cpu().numpy()

    @torch.inference_mode()
    def process_gpu(self, frame, output_size=(720, 1296)):
        from .gpu_video_output import GpuFrame
        from .native_config import NATIVE_SIZES
        if frame.ndim != 3 or frame.shape[2] != 3 or (frame.shape[1], frame.shape[0]) not in NATIVE_SIZES or frame.dtype != np.uint8:
            raise ValueError('SR requires a supported native-size RGB uint8 frame')
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            image = torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).to(
                self.device, torch.float16, memory_format=torch.channels_last) / 255
            if self.graph is not None and self.graph_key == (tuple(frame.shape[:2]), tuple(output_size)):
                self.graph_input.copy_(image)
                self.graph.replay()
                # NVENC consumes asynchronously. Each frame owns its storage so
                # the next replay cannot overwrite pixels still being encoded.
                pixels = self.graph_pixels.clone()
            else:
                pixels = self._render_gpu(image, output_size)
            return GpuFrame.record(pixels)


class OutputUpscale:
    """Resident model; serialized configuration/work, no additional work queue."""
    def __init__(self, device, denoise=.5):
        self.device, self.denoise = str(device), denoise
        from .native_config import load_native_config
        native = load_native_config()
        self.native_size = (native['height'], native['width'])
        # The four bounded JPEG workers provide CPU parallelism. Avoid each
        # color conversion spawning another machine-wide OpenCV worker team.
        # Process-local only; no changes to Easel/Flux/SDK environments.
        cv2.setNumThreads(1)
        self.model = None
        self.backend = 'realesrgan'
        self.preload_vsr = False
        self.models = {}
        self.enabled = True
        self.error = None
        self.lock = threading.Lock()
        self.load_s = self.warmup_s = None
        self.last = {}
        self.output_config = ROOT / 'tulip/output-presentation.json'
        self.output_size = (720, 1296)
        self.config_error = None
        self.direct_requested = False
        self.progressive = False
        self.rife_gpu = False
        from .direct_ndi_output import DirectNdiPublisher
        self.direct_publisher = DirectNdiPublisher()
        self.gpu_publisher = None
        self.gpu_config = None
        self.gpu_config_error = None
        self._refresh_output_size()
        self.encoder = ThreadPoolExecutor(max_workers=4, thread_name_prefix='joyai-jpeg')
        from .neural_rendering_output import NeuralRenderingOutput
        self.neural_rendering = NeuralRenderingOutput()

    def close(self):
        if self.gpu_publisher is not None: self.gpu_publisher.close()
        self.direct_publisher.close()
        self.neural_rendering.close()
        self.encoder.shutdown(wait=True, cancel_futures=True)
        for model in self.models.values():
            if hasattr(model, 'close'): model.close()

    @staticmethod
    def encode(frame, quality):
        ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError('Final output JPEG encoding failed')
        return jpeg.tobytes()

    @staticmethod
    def encode_preview(frame, quality):
        # Resize inside the existing bounded JPEG pool, not serially across
        # the entire chunk. This never changes the directly published pixels.
        if frame.shape[0] > 720:
            frame = cv2.resize(frame, (1296, 720), interpolation=cv2.INTER_AREA)
        return OutputUpscale.encode(frame, quality)

    def _refresh_output_size(self):
        # One immutable size per chunk. A bad/partial edit retains the last
        # working value; deleting the optional config restores the baseline.
        try:
            cfg = json.loads(self.output_config.read_text(encoding='utf-8-sig')) if self.output_config.exists() else {}
            height = cfg.get('height', 720)
            if type(height) is not int or height not in (720, 1080, 1440):
                raise ValueError('Output height must be 720, 1080 or 1440')
            direct = cfg.get('direct_ndi', False)
            if type(direct) is not bool:
                raise ValueError('direct_ndi must be boolean')
            progressive = cfg.get('progressive', False)
            if type(progressive) is not bool:
                raise ValueError('progressive must be boolean')
            rife_gpu = cfg.get('rife_gpu', False)
            if type(rife_gpu) is not bool:
                raise ValueError('rife_gpu must be boolean')
            denoise = cfg.get('denoise', .5)
            backend = cfg.get('backend', 'realesrgan')
            if backend not in ('realesrgan', 'vsr_ultra'):
                raise ValueError('SR backend must be realesrgan or vsr_ultra')
            preload_vsr = cfg.get('preload_vsr', False)
            if type(preload_vsr) is not bool:
                raise ValueError('preload_vsr must be boolean')
            if isinstance(denoise, bool) or not isinstance(denoise, (int, float)) or not 0 <= denoise <= 1:
                raise ValueError('SR denoise must be in [0,1]')
            gpu = cfg.get('gpu_video')
            if gpu is not None:
                if not isinstance(gpu,dict) or type(gpu.get('enabled',False)) is not bool:
                    raise ValueError('gpu_video must be an object with a boolean enabled field')
                if gpu.get('enabled'):
                    gpu_output_size(gpu.get('height'))
                    if not progressive: raise ValueError('GPU video requires progressive presentation')
                    if not isinstance(gpu.get('url'),str) or not gpu['url'].startswith('rtmp://127.0.0.1:1935/'):
                        raise ValueError('GPU video requires a local RTMP URL')
                    if type(gpu.get('bitrate')) is not int or not 1000000 <= gpu['bitrate'] <= 50000000:
                        raise ValueError('GPU video bitrate must be 1–50 Mbps')
                    if not isinstance(gpu.get('ffmpeg'),str): raise ValueError('GPU video requires its ffmpeg executable')
            if self.model is not None and backend != self.backend and backend not in self.models:
                raise ValueError('Requested SR backend is not prewarmed; restart JoyAI to load it')
            if hasattr(self, 'denoise') and denoise != self.denoise:
                if self.model is not None:
                    # Replace only the small SR model under the existing output
                    # lock. JoyAI weights, history and RIFE remain untouched.
                    self.models['realesrgan'] = PixelUpscaler(self.device, denoise)
                self.denoise = denoise
            if self.model is not None:
                self.model = self.models[backend]
            self.backend, self.preload_vsr = backend, preload_vsr
            self.gpu_config = gpu if gpu and gpu.get('enabled') else None
            self.direct_requested = direct
            self.progressive = progressive
            self.rife_gpu = rife_gpu
            self.output_size = (height, height * 9 // 5)
            self.config_error = None
        except (OSError, ValueError, AttributeError, RuntimeError) as exc:
            self.config_error = str(exc)

    def direct_status(self):
        status = self.direct_publisher.status()
        status['enabled'] = self.direct_requested and not status['error']
        return status

    def presentation_publishers(self):
        return [self.direct_publisher] + ([self.gpu_publisher] if self.gpu_publisher is not None else [])

    def begin_transition(self, source, *, moving=False):
        return [p.begin_transition(source,moving=moving) for p in self.presentation_publishers()]

    def cancel_transition(self):
        for p in self.presentation_publishers(): p.cancel_transition()

    def close_presentation(self, *, preserve=False):
        for p in self.presentation_publishers():
            if not (preserve and p.reset_scene()): p.close()

    def gpu_status(self):
        if self.gpu_publisher is not None:
            return {**self.gpu_publisher.status(),'config_error':self.gpu_config_error}
        return {'enabled':False,'error':self.gpu_config_error}

    def _configure_gpu_publisher(self):
        from .gpu_video_output import GpuVideoPublisher
        config = self.gpu_config
        if not config:
            if self.gpu_publisher is not None:
                self.gpu_publisher.close();self.gpu_publisher=None
            self.gpu_config_error=None
            return
        if self.gpu_publisher is not None and getattr(self,'_active_gpu_config',None)==config:
            return
        if self.gpu_publisher is not None: self.gpu_publisher.close()
        try:
            self.gpu_publisher=GpuVideoPublisher(self.device,config['ffmpeg'],config['url'],config['bitrate'])
            self._active_gpu_config=dict(config)
            self.gpu_config_error=None
        except Exception as exc:
            self.gpu_publisher=None
            self.gpu_config_error=str(exc)

    def status(self):
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        index = int(self.device.split(':')[-1])
        native_height, native_width = self.native_size
        return {'supported': True, 'enabled': self.enabled, 'loaded': self.model is not None,
                'device': self.device, 'physical_gpu': visible[index] if index < len(visible) else None,
                'backend': self.backend, 'prewarmed_backends': list(self.models),
                'precision': 'float32-io' if self.backend == 'vsr_ultra' else 'float16',
                'scale': self.output_size[0] / native_height if self.enabled else 1,
                'width': self.output_size[1] if self.enabled else native_width,
                'height': self.output_size[0] if self.enabled else native_height,
                'native_width': native_width, 'native_height': native_height,
                'config_error': self.config_error, 'direct_ndi': self.direct_status(), 'progressive': self.progressive,
                'rife_gpu_requested': self.rife_gpu,
                'denoise': self.denoise, 'error': self.error, 'load_s': self.load_s,
                'warmup_s': self.warmup_s, 'last': self.last,
                'sr_graph': bool(self.model is not None and getattr(self.model, 'graph', None) is not None),
                'sr_graph_error': getattr(self.model, 'graph_error', None),
                'gpu_video':self.gpu_status(),
                'neural_rendering': self.neural_rendering.status()}

    def interpolation_after(self):
        # Read only at chunk boundaries; malformed/missing config keeps baseline order.
        try:
            return self.neural_rendering._settings().get('output_order') == 'sr_nr_rife'
        except (OSError, ValueError):
            return False

    def gpu_interpolation_enabled(self, device):
        """Cached chunk settings; pending GPU frames remain valid during a toggle."""
        if not (self.rife_gpu and self.enabled and self.progressive and
                self.backend == 'vsr_ultra' and self.model is not None and
                torch.device(device) == torch.device(self.device)):
            return False
        try:
            settings = self.neural_rendering._settings()
            return not settings.get('enabled', False) and settings.get('output_order') != 'sr_nr_rife'
        except (OSError, ValueError, AttributeError):
            return False

    @staticmethod
    def _cpu_frame(frame):
        from .gpu_video_output import GpuFrame
        if isinstance(frame, GpuFrame):
            with torch.cuda.device(frame.tensor.device):
                return frame.wait().cpu().numpy()
        return frame

    def encode_frames(self, pixels, quality=95, profile=None):
        if not pixels or len(pixels) > 32:
            raise ValueError('Output accepts 1..32 presentation frames')
        profile = profile if profile is not None else {}
        if self.direct_requested and self.direct_publisher.available():
            self.direct_publisher.submit(pixels, float(profile.get('presentation_fps', 32)))
        elif self.direct_publisher.thread is not None:
            self.direct_publisher.close()
        direct = self.direct_status()
        profile['direct_ndi'] = direct
        preview_height = min(pixels[0].shape[0], 720) if self.direct_requested else pixels[0].shape[0]
        preview_width = preview_height * 9 // 5
        profile.setdefault('upscale', {}).update(width=preview_width, height=preview_height)
        encode = self.encode_preview if self.direct_requested else self.encode
        return list(self.encoder.map(encode, pixels, [quality] * len(pixels)))

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
                self.model.process(np.zeros((*self.native_size, 3), dtype=np.uint8), self.output_size)
                graph_size = self.output_size
                if self.gpu_config:
                    graph_size = gpu_output_size(self.gpu_config['height'])
                self.model.warmup_graph(self.native_size, graph_size)
                self.models['realesrgan'] = self.model
                if self.preload_vsr or self.backend == 'vsr_ultra':
                    from .vsr_output import VsrUpscaler
                    vsr = VsrUpscaler(self.device)
                    vsr.process(np.zeros((*self.native_size, 3), dtype=np.uint8), graph_size)
                    self.models['vsr_ultra'] = vsr
                self.model = self.models[self.backend]
                self.warmup_s = time.perf_counter() - start
                self.neural_rendering.process([np.zeros((720, 1296, 3), dtype=np.uint8)])
            except Exception as exc:
                self.error, self.enabled, self.model = str(exc), False, None
                print(f'#####[SR] Native-output fallback: {exc!r}', flush=True)

    def process(self, frames, profile, quality=95, *, raw_output=False):
        if not frames or len(frames) > 32:
            raise ValueError('SR accepts one bounded presentation chunk (1..32 frames)')
        from .native_config import NATIVE_SIZES
        from .gpu_video_output import GpuFrame
        gpu_input = isinstance(frames[0], GpuFrame)
        def valid(frame):
            if gpu_input:
                if not isinstance(frame, GpuFrame): return False
                pixels = frame.tensor
                if pixels.device != torch.device(self.device) or pixels.dtype != torch.uint8: return False
            else:
                pixels = frame
                if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8: return False
            return (pixels.ndim == 3 and pixels.shape[2] == 3 and
                    (pixels.shape[1], pixels.shape[0]) in NATIVE_SIZES and
                    pixels.shape == frames[0].shape)
        if not all(valid(f) for f in frames):
            raise ValueError('SR expects matched supported native-size RGB uint8 input')
        start = time.perf_counter()
        with self.lock:
            self.native_size = frames[0].shape[:2]
            self._refresh_output_size()
            self._configure_gpu_publisher()
            # NR and raw-output callers retain their original chunk contract.
            try:
                nr_requested = self.neural_rendering._settings().get('enabled', False)
            except (OSError, ValueError, AttributeError):
                nr_requested = True
            # A config change can land after RIFE has prepared one GPU chunk.
            # Drain that chunk through the existing CPU contract if its new
            # consumer requires it; never discard or reinterpret its storage.
            direct_gpu = (self.progressive and self.enabled and self.backend == 'vsr_ultra' and
                          not raw_output and not nr_requested and
                          ((self.direct_requested and self.direct_publisher.available()) or self.gpu_config))
            profile['rife_gpu_handoff'] = {'gpu_input': gpu_input, 'input_download_bytes': 0}
            if gpu_input and not direct_gpu:
                profile['rife_gpu_handoff']['input_download_bytes'] = sum(f.tensor.numel() for f in frames)
                frames = [self._cpu_frame(f) for f in frames]
            if (self.progressive and self.enabled and
                    ((self.direct_requested and self.direct_publisher.available()) or self.gpu_config) and
                    not raw_output and not nr_requested):
                return self._process_progressive(frames, profile, quality, start)
            if self.gpu_publisher is not None:
                self.gpu_publisher.close()
                self.gpu_config_error='GPU video requires enabled SR, progressive delivery and NR/RIFE-before-SR order'
            sr_start = time.perf_counter()
            pixels = frames
            if self.enabled:
                try:
                    pixels = [self.model.process(f, self.output_size) for f in frames]
                except Exception as exc:
                    self.enabled, self.error = False, str(exc)
                    pixels = frames
                    print(f'#####[SR] Native-output fallback: {exc!r}', flush=True)
            sr_s = time.perf_counter() - sr_start
            pixels = [f for i in range(0, len(pixels), 16)
                      for f in self.neural_rendering.process(pixels[i:i+16])]
            jpeg_start = time.perf_counter()
            # At most 32 tasks, consumed before this call returns. No rolling
            # queue: a slow encoder backpressures the existing chunk pump.
            output = pixels if raw_output else self.encode_frames(pixels, max(1, min(100, int(quality))), profile)
            self.last = {'sr_s': sr_s, 'jpeg_s': time.perf_counter()-jpeg_start,
                         'total_s': time.perf_counter()-start, 'count': len(output),
                         'width': pixels[0].shape[1], 'height': pixels[0].shape[0],
                         'neural_rendering': self.neural_rendering.status(),
                         'rife_gpu_handoff': profile['rife_gpu_handoff']}
            preview_size = profile.get('upscale', {})
            profile['upscale'] = {**self.last, **preview_size, 'enabled': self.enabled, 'error': self.error,
                                  'device': self.device, 'precision': 'float16'}
            return output

    def _process_progressive(self, frames, profile, quality, start):
        """Publish completed pixels in order; overlap bounded CPU preview work."""
        futures = []
        sr_s = 0.
        first_publish_s = None
        try:
            for frame in frames:
                tick = time.perf_counter()
                try:
                    if not self.enabled:
                        raise RuntimeError(self.error)
                    gpu = getattr(self,'gpu_publisher',None)
                    if gpu is not None and not gpu.error:
                        gpu_frame=self.model.process_gpu(frame,gpu_output_size(self.gpu_config['height']))
                        gpu.submit([gpu_frame],float(profile.get('presentation_fps',32)))
                        # Only the independent <=1080p NDI/preview branch downloads.
                        # Mobile's high-resolution pixels stay resident through encode.
                        with torch.cuda.device(self.model.device),torch.cuda.stream(self.model.stream):
                            preview=gpu_frame.wait()
                            if tuple(preview.shape[:2])!=self.output_size:
                                preview=F.interpolate(preview.permute(2,0,1)[None].float(),size=self.output_size,
                                    mode='bicubic',align_corners=False,antialias=True)[0].permute(1,2,0).clamp_(0,255).byte()
                            pixels=preview.cpu().numpy()
                    else:
                        pixels = self.model.process(frame, self.output_size)
                except Exception as exc:
                    # Frames already published cannot be recalled. Keep geometry
                    # stable for the remainder of this chunk if SR fails midway.
                    self.enabled, self.error = False, str(exc)
                    pixels = cv2.resize(self._cpu_frame(frame), self.output_size[::-1], interpolation=cv2.INTER_CUBIC)
                sr_s += time.perf_counter() - tick
                if self.direct_requested and self.direct_publisher.available():
                    self.direct_publisher.submit([pixels], float(profile.get('presentation_fps', 32)))
                if first_publish_s is None:
                    first_publish_s = time.perf_counter() - start
                futures.append(self.encoder.submit(self.encode_preview, pixels, max(1, min(100, int(quality)))))
            tick = time.perf_counter()
            output = [f.result() for f in futures]
            jpeg_wait_s = time.perf_counter() - tick
        finally:
            # No jobs survive this bounded chunk, including on encoder failure.
            for future in futures:
                if not future.cancelled():
                    future.exception()
        profile['direct_ndi'] = self.direct_status()
        if getattr(self,'gpu_publisher',None) is not None: profile['gpu_video']=self.gpu_status()
        self.last = {'sr_s': sr_s, 'jpeg_s': jpeg_wait_s,
                     'total_s': time.perf_counter()-start, 'count': len(output),
                     'width': self.output_size[1], 'height': self.output_size[0],
                     'progressive': True, 'first_publish_s': first_publish_s,
                     'rife_gpu_handoff': profile.get('rife_gpu_handoff'),
                     'neural_rendering': self.neural_rendering.status()}
        preview_height = min(self.output_size[0], 720)
        profile['upscale'] = {**self.last, 'width': preview_height*9//5, 'height': preview_height,
                              'enabled': self.enabled, 'error': self.error,
                              'device': self.device, 'precision': 'float16'}
        return output
