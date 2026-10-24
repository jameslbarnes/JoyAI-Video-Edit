"""Explicit local shader source: Spout -> CUDA, sampled by existing input credit.

No image FIFO. Every capture owns its pixels and rejects a stale/disconnected
sender. Cross-adapter copies are reported honestly; CUDA peer access is optional.
"""
import ctypes
from contextlib import nullcontext
import json
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .fresh_input import FreshInputTimeout

ROOT = Path(__file__).resolve().parents[3]
SETTINGS = ROOT / "tulip/input-transport.json"


def load_input_config(path=SETTINGS):
    config = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    transport = config.get("transport", "mjpeg")
    if transport not in ("mjpeg", "spout"):
        raise ValueError("Input transport must be mjpeg or spout")
    if transport == "spout":
        source = config.get("source")
        device = config.get("device")
        if not isinstance(source, str) or not source or len(source) > 200:
            raise ValueError("Spout input requires an explicit source name")
        if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError("Spout input requires an explicit CUDA adapter")
    return config


class GpuInputFrame:
    """Immutable RGB pixels; CPU conversion is only for explicit PIL consumers."""
    def __init__(self, pixels, metrics=None):
        if pixels.dtype != torch.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("GPU input requires HWC RGB uint8")
        self.pixels = pixels
        self.metrics = metrics
        self.ready = None
        if pixels.device.type == "cuda":
            self.ready = torch.cuda.Event()
            self.ready.record(torch.cuda.current_stream(pixels.device))

    @property
    def size(self):
        return self.pixels.shape[1], self.pixels.shape[0]

    def tensor(self, device=None):
        with torch.cuda.device(self.pixels.device) if self.ready else nullcontext():
            if self.ready:
                torch.cuda.current_stream(self.pixels.device).wait_event(self.ready)
            target = torch.device(device) if device is not None else self.pixels.device
            if target != self.pixels.device and self.metrics is not None:
                self.metrics["cross_gpu_bytes"] += self.pixels.numel()
            # PyTorch stages through host memory when CUDA peer access is absent.
            return self.pixels.to(target)

    def copy(self):
        return self  # Immutable owned snapshot, never the reusable receiver buffer.

    def convert(self, mode):
        return self if mode == "RGB" else self.to_pil().convert(mode)

    def resize(self, size, resample=None):
        if tuple(size) == self.size:
            return self
        pixels = self.tensor()
        resized = pixels.permute(2, 0, 1)[None].float()
        # PIL's uint8 bicubic resize rounds/clamps between horizontal and
        # vertical passes. Match that boundary behavior on high-contrast edges.
        if size[0] != self.size[0]:
            resized = F.interpolate(resized, size=(self.size[1], size[0]),
                mode="bicubic", align_corners=False, antialias=True).round().clamp(0, 255)
        if size[1] != self.size[1]:
            resized = F.interpolate(resized, size=(size[1], size[0]),
                mode="bicubic", align_corners=False, antialias=True)
        return GpuInputFrame(resized[0].permute(1, 2, 0).round().clamp(0, 255).byte().contiguous(),
                             self.metrics)

    def to_pil(self):
        if self.metrics is not None:
            self.metrics["pil_downloads"] += 1
        return Image.fromarray(self.tensor().cpu().numpy())

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.to_pil(), dtype=dtype)


class SpoutInput:
    def __init__(self, source, device):
        self.source = source
        self.device = torch.device(device)
        self.lock = threading.RLock()
        self.handle = None
        self.storage = None
        self.metrics = dict(transport="spout", source=source, device=str(self.device),
            frames=0, frame_number=0, read_ms=0, last_frame_at=0, width=0, height=0,
            jpeg_bytes=0, pil_downloads=0, cross_gpu_bytes=0, error=None)
        self.lib = ctypes.CDLL(str(ROOT / "tulip/spout-native/build/Release/joyai_spout_input.dll"))
        self.lib.jsi_create.argtypes = [ctypes.c_char_p]
        self.lib.jsi_create.restype = ctypes.c_void_p
        self.lib.jsi_read.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
                                      ctypes.POINTER(ctypes.c_uint64)]
        self.lib.jsi_read.restype = ctypes.c_int
        self.lib.jsi_error.argtypes = [ctypes.c_void_p]
        self.lib.jsi_error.restype = ctypes.c_char_p
        self.lib.jsi_close.argtypes = [ctypes.c_void_p]
        with torch.cuda.device(self.device):
            torch.cuda.init()
            self.stream = torch.cuda.Stream(device=self.device)
            self.handle = self.lib.jsi_create(source.encode("utf-8"))
            if not self.handle:
                raise RuntimeError(self.lib.jsi_error(None).decode())

    @torch.inference_mode()
    def capture(self, width, height, timeout=.5):
        with self.lock, torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            if not self.handle:
                raise RuntimeError("Spout input is closed")
            start = time.perf_counter()
            info = (ctypes.c_uint64 * 4)()
            while time.perf_counter() - start < timeout:
                rc = self.lib.jsi_read(self.handle, self.storage.data_ptr() if self.storage is not None else 0,
                    self.storage.numel() if self.storage is not None else 0, info)
                if rc < 0:
                    self.metrics["error"] = self.lib.jsi_error(self.handle).decode()
                    raise RuntimeError(self.metrics["error"])
                if rc == 2:
                    self.storage = torch.empty((info[1], info[0], 4), device=self.device, dtype=torch.uint8)
                elif rc == 1:
                    w, h, fmt, number = info
                    rgba = self.storage.view(-1)[:w*h*4].view(h, w, 4)
                    # Own the snapshot before the next native call can reuse storage.
                    rgb = rgba[:, :, [2, 1, 0]].contiguous() if fmt == 87 else rgba[:, :, :3].contiguous()
                    frame = GpuInputFrame(rgb, self.metrics).resize((width, height))
                    # Native copies use their own stream. Finish reading reusable
                    # input storage before releasing this lock to another caller.
                    frame.ready.synchronize()
                    self.metrics.update(frames=self.metrics["frames"] + 1, frame_number=int(number),
                        width=int(w), height=int(h), last_frame_at=time.time(),
                        read_ms=(time.perf_counter()-start)*1000, error=None)
                    return frame
                else:
                    time.sleep(.001)
            self.metrics["error"] = f"No fresh frame from Spout source {self.source!r} within {timeout}s"
            raise FreshInputTimeout(self.metrics["error"])

    def status(self):
        return dict(self.metrics)

    def close(self):
        with self.lock:
            if self.handle:
                self.lib.jsi_close(self.handle)
                self.handle = None
            self.storage = None


class SpoutShaderBridge:
    """Short-lived GPU shader preview while the next scene is initialized."""
    def __init__(self, source, device, seconds=20):
        self.reader = SpoutInput(source, device)
        self.stopped = threading.Event()
        self.error = None
        try:
            self.frame = self.reader.capture(648, 360)
        except Exception:
            self.reader.close()
            raise
        self.last_at = time.monotonic()
        self.until = self.last_at + seconds
        self.thread = threading.Thread(target=self._run, name="spout-scene-input", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stopped.wait(1/32) and time.monotonic() < self.until:
                self.frame = self.reader.capture(648, 360)
                self.last_at = time.monotonic()
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.reader.close()

    def latest(self):
        return self.frame if time.monotonic()-self.last_at < 2 else None

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("Spout scene receiver did not stop")
