"""Opt-in, bounded recording of already-host-resident native RGB frames.

No GPU reads, codec work or disk writes on the presentation thread. A full
queue invalidates the capture, never the renderer. Completed IDs cannot rerun.
"""
import json
from pathlib import Path
import queue
import re
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


class _Capture:
    def __init__(self, directory, capture_id, limit):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory, self.capture_id, self.limit = directory, capture_id, limit
        self.queue = queue.Queue(maxsize=4)
        self.accepted = self.written = 0
        self.sealed = self.complete = False
        self.error = None
        self.chunks = []
        self.started = time.time()
        self.thread = threading.Thread(target=self._write, name='native-capture', daemon=True)
        self.thread.start()

    def status(self):
        return dict(id=self.capture_id, accepted=self.accepted, written=self.written,
                    complete=self.complete, error=self.error, queued_chunks=self.queue.qsize())

    def submit(self, frames, metas, segment):
        if self.sealed or self.error: return
        take = min(len(frames), self.limit - self.accepted)
        if take <= 0: return
        if len(frames) != len(metas): raise ValueError('Capture metadata mismatch')
        if len(frames) > 8: raise ValueError('Capture accepts at most eight native frames')
        for frame in frames[:take]:
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError('Capture requires existing CPU RGB uint8 arrays')
            if frame.shape[0] > 720 or frame.shape[1] > 1296:
                raise ValueError('Capture geometry exceeds native bound')
        # Own the bytes: callers may reuse arrays after this method returns.
        pixels = np.stack(frames[:take])
        metadata = [{k:m.get(k) for k in ('seq','t_capture_ms')} for m in metas[:take]]
        try: self.queue.put_nowait((pixels, metadata, segment))
        except queue.Full:
            self.error = 'Capture disk queue full; capture invalid, renderer unaffected'
            self.sealed = True
            return
        self.accepted += take
        if self.accepted == self.limit: self.sealed = True

    def _write(self):
        try:
            while not self.sealed or not self.queue.empty():
                try: pixels, metas, segment = self.queue.get(timeout=.1)
                except queue.Empty: continue
                name = f'chunk-{len(self.chunks):04d}.npy'
                with (self.directory / name).open('xb') as f: np.save(f, pixels)
                self.chunks.append(dict(file=name, segment=segment, frames=len(pixels),
                                        shape=list(pixels.shape), metadata=metas))
                self.written += len(pixels)
            self.complete = self.written == self.limit and not self.error
        except Exception as exc:
            self.error = repr(exc)
            self.sealed = True
        finally:
            manifest = dict(**self.status(), limit=self.limit, started=self.started,
                            finished=time.time(), chunks=self.chunks,
                            position='native RGB before interpolation and upscaling', gpu_download_bytes=0)
            try:
                pending = self.directory / 'manifest.pending.json'
                pending.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
                pending.replace(self.directory / 'manifest.json')
            except Exception as exc: self.error = repr(exc); self.complete = False

    def close(self):
        if not self.sealed:
            self.error = 'Session closed before capture limit'
            self.sealed = True
        self.thread.join(timeout=1)


class NativeCaptureTap:
    def __init__(self, settings=None, output_root=None):
        self.settings = Path(settings) if settings else ROOT / 'tulip/native-capture.json'
        self.output_root = Path(output_root) if output_root else ROOT / 'tulip/evidence/native-captures'
        self.job = None
        self.segment = 0
        self.error = None
        self.seen = set()

    def reset(self): self.segment += 1

    def observe(self, frames, metas, profile):
        started = time.perf_counter()
        try:
            command = json.loads(self.settings.read_text(encoding='utf-8-sig')) if self.settings.exists() else {}
            capture_id = command.get('id')
            if capture_id and capture_id not in self.seen:
                if self.job and self.job.thread.is_alive():
                    raise ValueError('Prior capture still writing; new capture not started')
                self.seen.add(capture_id)
                if not isinstance(capture_id, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', capture_id):
                    raise ValueError('Invalid capture ID')
                limit = command.get('max_frames', 192)
                if type(limit) is not int or not 1 <= limit <= 600:
                    raise ValueError('Capture limit must be 1..600 frames')
                self.job = _Capture(self.output_root / capture_id, capture_id, limit)
                self.error = None
            if self.job:
                if capture_id != self.job.capture_id and not self.job.sealed:
                    self.job.error = 'Capture disarmed before limit'; self.job.sealed = True
                if not self.error: self.job.submit(frames, metas, self.segment)
        except Exception as exc:
            self.error = repr(exc)
            if self.job and not self.job.sealed:
                self.job.error = self.error; self.job.sealed = True
        if self.job or self.error:
            profile['native_capture'] = self.job.status() if self.job else {'error':self.error}
            profile['native_capture']['submit_ms'] = (time.perf_counter() - started) * 1000
            if self.error: profile['native_capture']['error'] = self.error

    def close(self):
        if self.job: self.job.close()
