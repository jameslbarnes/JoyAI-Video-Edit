"""Paced GPU UYVY conversion and pinned readback to the existing NDI owner.

Retains immutable GPU frames shared with NVENC; never downloads full RGB output.
One pinned staging buffer is reused only after CUDA completion and sendall().
The bridge owns NDI discovery/identity, native asynchronous send and SDK lifetime.
"""
import socket
import struct
import threading
import time

import numpy as np
import torch

from .direct_ndi_output import DirectNdiPublisher
from .gpu_video_output import GpuFrame, GpuVideoPublisher
from .ndi_pixels import pack_uyvy


class GpuNdiPublisher(DirectNdiPublisher):
    gpu_input = True
    _resize_source = GpuVideoPublisher._resize_source
    _blend = GpuVideoPublisher._blend
    _validate_frames = GpuVideoPublisher._validate_frames

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device(device)
        self.stage_ms = {}
        self.raw_download_bytes = 0
        self.stream = None

    def status(self):
        return {**super().status(), 'pixel_format': 'UYVY', 'gpu_conversion': True,
                'device': str(self.device), 'stage_ms': dict(self.stage_ms),
                'raw_download_bytes': self.raw_download_bytes}

    @torch.inference_mode()
    def warmup(self, size):
        # Compile the exact geometry while the service is warming, not on the
        # first due video frame. Device selection follows the output GPU.
        with torch.cuda.device(self.device):
            sample = torch.zeros((*size,3), dtype=torch.uint8, device=self.device)
            packed = pack_uyvy(sample)
            host = torch.empty(packed.shape, dtype=torch.uint8, pin_memory=True)
            host.copy_(packed, non_blocking=True)
            torch.cuda.current_stream(self.device).synchronize()

    def submit(self, frames, fps):
        # Preserve the existing presentation-only fallback if SR reports failure.
        if frames and isinstance(frames[0], np.ndarray):
            with torch.cuda.device(self.device):
                frames = [GpuFrame.record(torch.from_numpy(f).to(self.device)) for f in frames]
        return super().submit(frames, fps)

    @torch.inference_mode()
    def _run(self):
        deadline = time.perf_counter() + self.prime_s
        key = None
        try:
            with torch.cuda.device(self.device):
                self.stream = torch.cuda.Stream(device=self.device, priority=-1)
                copy_ready = torch.cuda.Event()
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                while True:
                    with self.condition:
                        while not self.stopping:
                            if not self.frames and self.transition is None:
                                if time.perf_counter() >= deadline: self._record_underrun()
                                self.condition.wait(.1)
                                continue
                            delay = deadline-time.perf_counter()
                            if delay > 0:
                                self.condition.wait(delay)
                                continue
                            selected = self._next_frame(time.perf_counter())
                            if selected is None:
                                self.condition.wait(.01)
                                continue
                            frame, next_key = selected
                            break
                        else:
                            return
                    if next_key != key:
                        self._connect(next_key)
                        key = next_key
                        w, h, fps = key
                        packed = torch.empty((h,w,2), dtype=torch.uint8, device=self.device)
                        host = torch.empty((h,w,2), dtype=torch.uint8, pin_memory=True)
                        payload = memoryview(host.numpy()).cast('B')
                        header = struct.pack('<4sBBHIII', b'EVSF', 1, 5, 0, w, h, len(payload))
                    started = time.perf_counter()
                    pack_uyvy(frame.wait(), packed)
                    host.copy_(packed, non_blocking=True)
                    copy_ready.record(self.stream)
                    copy_ready.synchronize()  # only this owned staging copy, never all GPUs
                    copied = time.perf_counter()
                    self.socket.sendall(header)
                    self.socket.sendall(payload)
                    finished = time.perf_counter()
                    with self.condition:
                        self.sent += 1
                        self.last_send_at = time.time()
                        self.last_send_ms = (finished-started)*1000
                        self.last_frame, self.last_key = frame, key
                        self.raw_download_bytes += len(payload)
                        self.stage_ms = {'pack_download': (copied-started)*1000,
                                         'transport': (finished-copied)*1000}
                    period = 1/fps
                    deadline = deadline+period if started-deadline < period else started+period
        except Exception as exc:
            with self.condition:
                if not self.stopping:
                    self.error = self.last_error = f'{type(exc).__name__}: {exc}'
                    self.error_at = time.monotonic()
                    self.dropped += len(self.frames)
                self.frames.clear()
        finally:
            if self.stream is not None: self.stream.synchronize()
            if self.socket is not None:
                self.socket.close()
                self.socket = None
            with self.condition:
                if self.thread is threading.current_thread():
                    self.thread = None
                    self.playback_primed = False
