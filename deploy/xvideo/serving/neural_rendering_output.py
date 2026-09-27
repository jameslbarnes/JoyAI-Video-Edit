"""Optional, bounded SR -> Neural Rendering presentation stage.

One persistent local child and one synchronous chunk at a time. Any failure
latches NR off; the original SR chunk is returned intact. A local JSON toggle
is polled at chunk boundaries so rollback does not restart the live model.
"""
import atexit
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'tulip/neural-rendering.json'


class NeuralRenderingOutput:
    def __init__(self, config=CONFIG):
        self.config_path = Path(config)
        self.process_handle = None
        self.pool = None
        self.log = None
        self.error = None
        self.enabled = False
        self.frames = 0
        self.last_s = 0.0
        self.intensity = 1.0
        self.adapter = 0
        atexit.register(self.close)

    def status(self):
        return {'enabled': self.enabled and not self.error,
                'loaded': self.process_handle is not None and self.process_handle.poll() is None,
                'error': self.error, 'frames': self.frames, 'last_s': self.last_s,
                'intensity': self.intensity, 'adapter': self.adapter,
                'position': 'after_realesrgan', 'communityRuntime': True}

    def close(self):
        p, self.process_handle = self.process_handle, None
        if p is not None:
            # Kill only our owned helper to unblock pending pipe I/O on failure.
            if p.poll() is None:
                p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        if p is not None:
            for pipe in (p.stdin, p.stdout):
                if pipe is not None:
                    pipe.close()
        if self.log is not None:
            self.log.close()
            self.log = None

    def _settings(self):
        if not self.config_path.exists():
            return {'enabled': False}
        cfg = json.loads(self.config_path.read_text(encoding='utf-8-sig'))
        if type(cfg.get('enabled', False)) is not bool:
            raise ValueError('NR enabled must be boolean')
        if cfg.get('enabled'):
            if cfg.get('intensity', 1.0) != 1.0 or cfg.get('adapter', 0) != 0:
                raise ValueError('Live trial supports Natural intensity1.0 on adapter0')
        return cfg

    def _start(self, cfg):
        exe = Path(cfg['executable']).resolve(strict=True)
        log_path = ROOT / 'tulip/neural-rendering-live.log'
        self.log = log_path.open('ab', buffering=0)
        self.process_handle = subprocess.Popen(
            [str(exe), '--nr-video', '--nr-in', '1296x720', '--nr-style', '1',
             '--nr-intensity', '1.0', '--nr-motion', '1', '--nr-motion-engine', 'auto',
             '--adapter', '0'], cwd=exe.parent, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, bufsize=0,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='joyai-nr-pipe')

    def _write(self, frames):
        for frame in frames:
            rgba = np.empty((720, 1296, 4), dtype=np.uint8)
            rgba[:, :, :3] = frame
            rgba[:, :, 3] = 255
            data = memoryview(rgba).cast('B')
            while data:
                n = self.process_handle.stdin.write(data)
                if not n:
                    raise RuntimeError('NR input pipe closed')
                data = data[n:]

    def _read(self, count):
        result = []
        for _ in range(count):
            data = bytearray(1296 * 720 * 4)
            view = memoryview(data)
            while view:
                n = self.process_handle.stdout.readinto(view)
                if not n:
                    raise RuntimeError('NR output pipe closed')
                view = view[n:]
            result.append(np.frombuffer(data, dtype=np.uint8).reshape(720, 1296, 4)[:, :, :3].copy())
        return result

    def process(self, frames):
        start = time.perf_counter()
        try:
            cfg = self._settings()
            self.enabled = cfg.get('enabled', False)
            if not self.enabled or self.error:
                self.close()
                return frames
            if not 1 <= len(frames) <= 16:
                raise ValueError('NR requires one chunk of1..16 frames')
            if any(f.shape != (720, 1296, 3) or f.dtype != np.uint8 for f in frames):
                # SR bypass remains available; discard temporal NR history.
                self.close()
                return frames
            cold = self.process_handle is None
            if cold:
                self._start(cfg)
            deadline = time.monotonic() + (15 if cold else 3)
            writer = self.pool.submit(self._write, frames)
            reader = self.pool.submit(self._read, len(frames))
            result = reader.result(timeout=max(.01, deadline-time.monotonic()))
            writer.result(timeout=max(.01, deadline-time.monotonic()))
            self.frames += len(result)
            return result
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.close()
            print(f'#####[NR] SR-output fallback: {self.error}', flush=True)
            return frames
        finally:
            self.last_s = time.perf_counter() - start
