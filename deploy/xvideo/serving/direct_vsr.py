"""VSR using persistent RGBA8 GPU buffers and one explicitly ordered CUDA stream."""
import collections
from contextlib import suppress
import ctypes as C
import threading
import time

import numpy as np
import torch

from .native_vfx import api


class DirectVsrUpscaler:
    def __init__(self, device, quality=4):
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("VSR requires an explicitly indexed CUDA device")
        if quality not in (0, 1, 2, 3, 4, 16, 17, 18, 19, 21, 23):
            raise ValueError("Unsupported VSR upscaling quality")
        self.api, self.quality = api(), quality
        self.lock = threading.RLock()
        self.handle = C.c_void_p()
        self.closed = False
        self.geometry = None
        self.input = self.output = self.src_image = self.dst_image = None
        self.pending = collections.deque()
        self.last = {}
        self.frames = 0
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device, priority=-1)
            self.api.check(self.api.create(b"VideoSuperRes", C.byref(self.handle)), "NvVFX_CreateEffect")
            try:
                self.api.check(self.api.set_u32(self.handle, b"QualityLevel", quality), "VSR quality")
                self.api.check(self.api.set_stream(self.handle, b"CudaStream", self.stream.cuda_stream),
                               "VSR stream")
            except Exception:
                self.api.destroy(self.handle)
                self.handle = C.c_void_p()
                raise

    def _configure(self, input_size, output_size):
        geometry = (tuple(input_size), tuple(output_size))
        if self.geometry == geometry:
            return
        # Configuration changes are infrequent and must retire this effect's work.
        # Steady-state frames never synchronize the device or host.
        self.stream.synchronize()
        self.input = torch.empty((*input_size, 4), device=self.device, dtype=torch.uint8)
        self.output = torch.empty((*output_size, 4), device=self.device, dtype=torch.uint8)
        self.src_image, self.dst_image = self.api.wrap_rgba(self.input), self.api.wrap_rgba(self.output)
        self.api.check(self.api.set_image(self.handle, b"SrcImage0", C.byref(self.src_image)), "VSR input")
        self.api.check(self.api.set_image(self.handle, b"DstImage0", C.byref(self.dst_image)), "VSR output")
        self.api.check(self.api.load(self.handle), "NvVFX_Load")
        self.geometry = geometry

    @torch.inference_mode()
    def process_gpu(self, frame, output_size=(720, 1296)):
        from .gpu_video_output import GpuFrame
        from .native_config import NATIVE_SIZES
        owned = frame if isinstance(frame, GpuFrame) else None
        pixels = owned.tensor if owned is not None else frame
        tensor = isinstance(pixels, torch.Tensor)
        if (pixels.ndim != 3 or pixels.shape[2] != 3 or
                (pixels.shape[1], pixels.shape[0]) not in NATIVE_SIZES or
                pixels.dtype != (torch.uint8 if tensor else np.uint8)):
            raise ValueError("VSR requires supported native-size RGB uint8 input")
        if tensor and pixels.device != self.device:
            raise ValueError("VSR tensor input must already be on its output GPU")
        if (len(output_size) != 2 or any(type(x) is not int or x <= 0 for x in output_size)):
            raise ValueError("Invalid VSR output size")
        with self.lock, torch.cuda.device(self.device):
            if self.closed:
                raise RuntimeError("VSR is closed")
            producer = torch.cuda.current_stream(self.device)
            with torch.cuda.stream(self.stream):
                self._configure(tuple(pixels.shape[:2]), output_size)
                if owned is not None:
                    owned.wait()
                elif tensor:
                    self.stream.wait_stream(producer)
                else:
                    pixels = torch.from_numpy(pixels.copy()).to(self.device)
                pixels.record_stream(self.stream)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(self.stream)
                self.input[..., :3].copy_(pixels)
                self.input[..., 3].fill_(255)
                tick = time.perf_counter()
                self.api.check(self.api.run(self.handle, 1), "NvVFX_Run(async)")
                run_ms = (time.perf_counter() - tick) * 1000
                # This RGB copy owns its storage. On this stream it finishes
                # before a subsequent invocation overwrites the SDK destination.
                result = self.output[..., :3].contiguous()
                end.record(self.stream)
                self.frames += 1
                self.pending.append((start, end))
                while self.pending and self.pending[0][1].query():
                    before, after = self.pending.popleft()
                    self.last["gpu_span_ms"] = before.elapsed_time(after)
                while len(self.pending) > 64:
                    self.pending.popleft()
                self.last.update(run_host_ms=run_ms)
                return GpuFrame(result, end)

    @torch.inference_mode()
    def process(self, frame, output_size=(720, 1296)):
        result = self.process_gpu(frame, output_size)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            return result.wait().cpu().numpy()

    def status(self):
        return dict(implementation="native-rgba8", sdk=self.api.version, quality=self.quality,
                    async_run=True, device=str(self.device), frames=self.frames, **self.last)

    def close(self):
        with self.lock, torch.cuda.device(self.device):
            if self.closed:
                return
            self.stream.synchronize()
            if self.handle:
                self.api.destroy(self.handle)
                self.handle = C.c_void_p()
            self.input = self.output = self.src_image = self.dst_image = None
            self.pending.clear()
            self.closed = True

    def __del__(self):
        # A failed load must not strand the effect's allocations. Explicit close
        # remains the normal path; interpreter shutdown may already have torn CUDA down.
        with suppress(Exception):
            self.close()
