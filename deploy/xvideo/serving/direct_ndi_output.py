"""Bounded, paced raw TCP output to the existing local NDI bridge.

No GPU work, JPEGs, browser readback, or additional model/input session. Socket
failure latches this publisher off until reset; the caller retains preview output.
"""
import collections
import json
import socket
import struct
import threading
import time

import cv2
import numpy as np


class DirectNdiPublisher:
    def __init__(self, host='127.0.0.1', port=7780, max_frames=48, prime_s=.35, retry_s=2.0, send_timeout_s=2.0,
                 sender_id='primary-joyai', name='ETHEREA JoyAI'):
        self.address = (host, port)
        self.sender_id, self.name = sender_id, name
        self.max_frames, self.prime_s = max_frames, prime_s
        # A transport error used to latch forever (one slow bridge read at restart
        # killed NDI for hours). Errors now retry after retry_s; a full 1944x1080
        # RGBA frame is 8.4 MB, so sends get more than the 0.5 s connect timeout.
        self.retry_s, self.send_timeout_s = retry_s, send_timeout_s
        self.error_at = None
        self.last_error = None
        self.reconnects = 0
        self.condition = threading.Condition()
        self.frames = collections.deque()
        self.thread = self.socket = None
        self.stopping = False
        self.error = None
        self.sent = self.dropped = 0
        self.last_send_at = None
        self.last_send_ms = 0
        self.geometry = None
        self.last_frame = self.last_key = None
        self.transition = None
        self.transition_frames = self.transitions_completed = self.scene_discarded = 0
        self.playback_primed = False
        self.underruns = 0
        self._starved = False

    def status(self):
        with self.condition:
            return dict(enabled=self.thread is not None and not self.stopping and not self.error,
                        error=self.error, last_error=self.last_error, reconnects=self.reconnects, sent=self.sent, dropped=self.dropped,
                        queued=len(self.frames), max_frames=self.max_frames,
                        last_send_at=self.last_send_at, last_send_ms=self.last_send_ms,
                        geometry=self.geometry, sender_id=self.sender_id,
                        transition_phase=('waiting' if self.transition['reset'] else 'fade-out') if self.transition else None,
                        transition_frames=self.transition_frames, transitions_completed=self.transitions_completed,
                        scene_discarded=self.scene_discarded,
                        playback_primed=self.playback_primed, underruns=self.underruns)

    def begin_transition(self, source, *, moving=False):
        """Keep presentation alive using the existing native shader input."""
        with self.condition:
            if self.last_frame is None or self.error or self.stopping:
                return False
            # Each preparation owns a new handoff. Reusing a prior reset=True
            # transition lets its queued frames finish it during preparation;
            # reset_scene() then fails and closes the persistent publisher.
            self.transition = dict(start=time.perf_counter(), source=source,
                base=self.last_frame, key=self.last_key, reset=False,
                shader=None, source_frame=None, moving=moving, in_start=None)
            self.condition.notify_all()
            return True

    def reset_scene(self):
        """Drain old model output without closing the paced shader bridge."""
        with self.condition:
            if self.transition is None:
                return False
            self.scene_discarded += len(self.frames)
            self.frames.clear()
            self.playback_primed = False
            self._starved = False
            self.transition['reset'] = True
            self.condition.notify_all()
            return True

    def cancel_transition(self):
        with self.condition:
            self.transition = None
            self.condition.notify_all()

    def _resize_source(self, source, key):
        width, height, _ = key
        return cv2.resize(np.asarray(source, dtype=np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)

    def _blend(self, first, second, mix):
        return cv2.addWeighted(first, 1-mix, second, mix, 0)

    def _next_frame(self, now):
        # Called under condition. Only transitions do CPU resize/compositing;
        # ordinary output retains its zero-copy queue and native SR pixels.
        transition = self.transition
        if transition is not None:
            age = now - transition['start']
            # A failed controller must not retain a shader bridge indefinitely.
            if age > 20:
                self.transition = None
            elif not transition['moving'] and transition['reset'] and len(self.frames) >= min(self.max_frames, round(transition['key'][2])) and age >= .6:
                # The first 16 frames contain RIFE's shader-to-scene morph.
                # Wait for its following real chunk so it cannot drain to a freeze.
                self.transition = None
                self.transitions_completed += 1
                self.playback_primed = True
            else:
                if not transition['reset'] and self.frames:
                    transition['base'], transition['key'] = self.frames.popleft()
                source = transition['source']()
                if source is not None and (source is not transition['source_frame'] or
                        transition['shader'] is None or transition['shader'].shape != transition['base'].shape):
                    transition['shader'] = self._resize_source(source, transition['key'])
                    transition['source_frame'] = source
                shader = transition['shader']
                if transition['moving'] and transition['reset']:
                    if transition['in_start'] is None and len(self.frames) >= round(transition['key'][2] * .5) and age >= .6:
                        transition['in_start'] = now
                        self.playback_primed = True
                    # Keep the source moving until enough new output exists.
                    # Previewing frames[0] here exposes a frozen first image
                    # while the following generated chunk is still arriving.
                    if transition['in_start'] is not None and self.frames:
                        generated, key = self.frames.popleft()
                        t = min(1., (now-transition['in_start'])/.2)
                        mix = t*t*(3-2*t)
                        if t >= 1:
                            self.transition = None
                            self.transitions_completed += 1
                        self.transition_frames += 1
                        return (generated if shader is None else self._blend(shader, generated, mix)), key
                t = max(0., min(1., age / .6))
                mix = t * t * (3 - 2 * t)
                frame = transition['base'] if shader is None else (
                    shader if mix == 1 else self._blend(transition['base'], shader, mix))
                self.transition_frames += 1
                return frame, transition['key']
        if not self.playback_primed and self.frames:
            if len(self.frames) < min(self.max_frames, max(1, round(self.frames[0][1][2] * .5))):
                return None
            self.playback_primed = True
        if self.frames:
            self._starved = False
            return self.frames.popleft()
        return None

    def available(self):
        """True when frames can be submitted; clears a transport error once its retry delay has passed."""
        with self.condition:
            if (self.error and not self.stopping and self.thread is None
                    and time.monotonic() - self.error_at >= self.retry_s):
                self.error = None
                self.reconnects += 1
            return not self.error and not self.stopping

    def _record_underrun(self):
        if self.playback_primed and self.transition is None and not self._starved:
            self.underruns += 1
            self._starved = True

    def submit(self, frames, fps):
        if not frames or len(frames) > self.max_frames:
            raise ValueError('Direct NDI requires one bounded chunk')
        if not 1 <= fps <= 64:
            raise ValueError('Direct NDI fps must be in [1,64]')
        shape = self._validate_frames(frames)
        with self.condition:
            if self.error or self.stopping:
                return False
            key = (shape[1], shape[0], float(fps))
            if self.frames and self.frames[-1][1] != key:
                self.dropped += len(self.frames)
                self.frames.clear()
            while len(self.frames) + len(frames) > self.max_frames:
                self.frames.popleft()
                self.dropped += 1
            # The upstream arrays are immutable after publication. Retain them,
            # rather than copying a full chunk, and convert only the due frame.
            self.frames.extend((f, key) for f in frames)
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name='joyai-direct-output', daemon=True)
                self.thread.start()
            self.condition.notify_all()
            return True

    def _validate_frames(self, frames):
        shape = frames[0].shape
        if shape not in ((360, 648, 3), (720, 1296, 3), (1080, 1944, 3), (1440, 2592, 3)):
            raise ValueError('Unsupported direct NDI geometry')
        if any(f.shape != shape or f.dtype != np.uint8 for f in frames):
            raise ValueError('Direct NDI requires matching RGB uint8 frames')
        return shape

    def close(self):
        with self.condition:
            self.stopping = True
            self.frames.clear()
            self.transition = None
            self.last_frame = self.last_key = None
            sock, worker = self.socket, self.thread
            self.condition.notify_all()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if worker is not None:
            worker.join(timeout=2)
            if worker.is_alive():
                raise RuntimeError('Direct NDI worker did not stop')
        with self.condition:
            self.socket = self.thread = None
            self.last_frame = self.last_key = None
            self.stopping = False
            self.playback_primed = False
            self._starved = False

    def _connect(self, key):
        if self.socket is not None:
            self.socket.close()
        sock = socket.create_connection(self.address, timeout=.5)
        sock.settimeout(self.send_timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.socket = sock
        width, height, fps = key
        hello = dict(senderId=self.sender_id, name=self.name, width=width,
                     height=height, fps=fps, frameRateN=round(fps * 1000),
                     frameRateD=1000, sourceType='joyai-direct')
        sock.sendall((json.dumps(hello) + '\n').encode())
        self.geometry = key

    def _run(self):
        deadline = time.perf_counter() + self.prime_s
        key = None
        try:
            while True:
                with self.condition:
                    while not self.stopping:
                        if not self.frames and self.transition is None:
                            if time.perf_counter() >= deadline:
                                self._record_underrun()
                            self.condition.wait(.1)
                            continue
                        delay = deadline - time.perf_counter()
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
                start = time.perf_counter()
                rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                width, height, fps = key
                header = struct.pack('<4sBBHIII', b'EVSF', 1, 1, 0, width, height, rgba.nbytes)
                self.socket.sendall(header)
                self.socket.sendall(memoryview(rgba).cast('B'))
                with self.condition:
                    self.sent += 1
                    self.last_send_at = time.time()
                    self.last_send_ms = (time.perf_counter() - start) * 1000
                    self.last_frame, self.last_key = frame, key
                # Keep fractional deadlines; never burst to catch up after a
                # transport stall or a model pause. The queue remains bounded.
                period = 1 / fps
                deadline = deadline + period if start - deadline < period else start + period
        except Exception as exc:
            with self.condition:
                if not self.stopping:
                    self.error = self.last_error = f'{type(exc).__name__}: {exc}'
                    self.error_at = time.monotonic()
                    self.dropped += len(self.frames)
                self.frames.clear()
        finally:
            if self.socket is not None:
                self.socket.close()
                self.socket = None
            with self.condition:
                if self.thread is threading.current_thread():
                    self.thread = None  # lets submit() start a fresh worker after retry
                    self.playback_primed = False
