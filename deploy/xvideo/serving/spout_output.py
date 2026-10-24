"""GPU-only local presentation. Native code owns the texture ring and clock."""
import ctypes
from collections import OrderedDict
import uuid
from pathlib import Path
import threading
import time

import numpy as np
import torch

from .direct_ndi_output import DirectNdiPublisher
from .gpu_video_output import GpuFrame, GpuVideoPublisher
from .feeder_trace import FeederTrace


class NativeStats(ctypes.Structure):
    _fields_ = [(k, ctypes.c_uint64) for k in (
        'submitted', 'sent', 'queued', 'discarded', 'underruns', 'missed_ticks',
        'access_misses', 'last_token', 'high_resolution')] + [
        (k, ctypes.c_double) for k in (
            'last_send_at', 'send_ms', 'copy_ms', 'late_ms', 'max_gap_ms',
            'first_send_at', 'gap_sum_ms', 'gap_square_sum')]


class NativeCopyTrace(ctypes.Structure):
    _fields_ = [('token', ctypes.c_uint64)] + [(key, ctypes.c_double) for key in (
        'at', 'context_ms', 'event_wait_ms', 'map_ms', 'array_ms',
        'enqueue_ms', 'unmap_ms', 'synchronize_ms', 'total_ms',
        'source_ready', 'gpu_span_ms', 'priority')]


class NativePresentSample(ctypes.Structure):
    _fields_ = [('serial', ctypes.c_uint64), ('token', ctypes.c_uint64)] + [
        (key, ctypes.c_double) for key in (
            'at', 'gap_ms', 'late_ms', 'empty_wait_ms', 'send_ms')] + [
        ('queued_before', ctypes.c_uint64), ('advanced', ctypes.c_uint64)]


class NativeFrameSample(ctypes.Structure):
    _fields_ = [('serial', ctypes.c_uint64), ('token', ctypes.c_uint64),
                ('sender_frame', ctypes.c_int64), ('published_ms', ctypes.c_double),
                ('monotonic_ms', ctypes.c_double)]


def library(path=None):
    path = path or Path(__file__).resolve().parents[3] / 'tulip/spout-native/build/Release/joyai_spout.dll'
    dll = ctypes.CDLL(str(path))
    signatures = {
        'jsp_create': ([ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_double,
                        ctypes.c_uint, ctypes.c_uint], ctypes.c_void_p),
        'jsp_submit': ([ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64], ctypes.c_int),
        'jsp_replace': ([ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64], ctypes.c_int),
        'jsp_stats': ([ctypes.c_void_p, ctypes.POINTER(NativeStats)], None),
        'jsp_clear': ([ctypes.c_void_p], None),
        'jsp_error': ([ctypes.c_void_p], ctypes.c_char_p),
        'jsp_close': ([ctypes.c_void_p], None),
    }
    for name, (args, result) in signatures.items():
        getattr(dll, name).argtypes = args
        getattr(dll, name).restype = result
    # Older rollback DLLs retain the original Stats ABI and can still publish.
    if hasattr(dll, 'jsp_copy_trace'):
        dll.jsp_copy_trace.argtypes = [ctypes.c_void_p, ctypes.POINTER(NativeCopyTrace), ctypes.c_uint]
        dll.jsp_copy_trace.restype = ctypes.c_int
    if hasattr(dll, 'jsp_set_copy_priority'):
        dll.jsp_set_copy_priority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        dll.jsp_set_copy_priority.restype = ctypes.c_int
    if hasattr(dll, 'jsp_present_trace'):
        dll.jsp_present_trace.argtypes = [ctypes.c_void_p, ctypes.POINTER(NativePresentSample),
                                         ctypes.c_uint, ctypes.c_uint]
        dll.jsp_present_trace.restype = ctypes.c_int
    if hasattr(dll, 'jsp_frame_trace'):
        dll.jsp_frame_trace.argtypes = [ctypes.c_void_p, ctypes.POINTER(NativeFrameSample),
                                        ctypes.c_uint, ctypes.c_uint]
        dll.jsp_frame_trace.restype = ctypes.c_int
    return dll


class GpuSpoutPublisher(DirectNdiPublisher):
    gpu_input = True
    transport = 'spout'
    _resize_source = GpuVideoPublisher._resize_source
    _blend = GpuVideoPublisher._blend
    _validate_frames = GpuVideoPublisher._validate_frames

    def __init__(self, device, name='ETHEREA JoyAI Spout', **kwargs):
        super().__init__(name=name, **kwargs)
        self.device = torch.device(device)
        self.dll = library()
        self.native = None
        self.sequence = 0
        self.retained = {}
        self.stream = None
        self.native_stats = {}
        self.replace_next = False
        self.native_capacity = 18
        self.feeder_trace = FeederTrace()
        self.copy_priority = 0
        self.applied_copy_priority = None
        self.present_events = []
        self.transition_reserve = 2
        self.profile_preparation = False
        self.preparation_events = None
        self.repeated_transition_frames = 0
        self.publisher_id = None
        self.frame_origins = OrderedDict()

    def _frame_trace(self):
        if not self.native or not hasattr(self.dll, 'jsp_frame_trace'):
            return []
        samples = (NativeFrameSample * 256)()
        count = self.dll.jsp_frame_trace(self.native, samples, 256, ctypes.sizeof(NativeFrameSample))
        if not 0 <= count <= 256:
            raise RuntimeError('Native frame provenance ABI mismatch')
        return [{**{key: getattr(row, key) for key, _ in row._fields_},
                 'origin': dict(self.frame_origins[row.token]) if self.frame_origins.get(row.token) else None}
                for row in samples[:count]]

    def frame_trace(self):
        # Dedicated diagnostic read only. Never attach this ring to per-frame
        # WebSocket metadata or ordinary health/status replies.
        with self.condition:
            return {'version': 1, 'supported': hasattr(self.dll, 'jsp_frame_trace'),
                    'publisher_id': self.publisher_id, 'name': self.name,
                    'frames': self._frame_trace(), 'at_ms': time.time() * 1000}

    def configure_preparation_profile(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('Spout preparation profile must be boolean')
        with self.condition:
            self.profile_preparation = enabled

    def configure_transition_reserve(self, value):
        # One additional 24-fps texture adds ~42 ms of transition headroom.
        # Keep this bounded and independent of the ordinary 16-frame reserve.
        if type(value) is not int or not 2 <= value <= 6:
            raise ValueError('Spout transition reserve must be an integer in [2,6]')
        with self.condition:
            self.transition_reserve = value
            self.condition.notify_all()

    def _native_reserve(self):
        return self.transition_reserve if self.transition is not None else 16

    def configure_copy_priority(self, value):
        if type(value) is not int or value not in (0, -1):
            raise ValueError('Spout copy priority must be 0 or -1')
        if value and not hasattr(self.dll, 'jsp_set_copy_priority'):
            raise ValueError('Spout native library does not support copy priority')
        with self.condition:
            self.copy_priority = value

    def _stats(self):
        if self.native:
            value = NativeStats()
            self.dll.jsp_stats(self.native, ctypes.byref(value))
            self.native_stats = {key: getattr(value, key) for key, _ in value._fields_}
        return self.native_stats

    def _presentation_events(self):
        # Health reads only. The high-frequency feeder queries _stats() and
        # must not copy/allocate the history on every queue check.
        if self.native and hasattr(self.dll, 'jsp_present_trace'):
            events = (NativePresentSample * 64)()
            count = self.dll.jsp_present_trace(self.native, events, 64,
                                               ctypes.sizeof(NativePresentSample))
            if not 0 <= count <= 64:
                raise RuntimeError('Native presentation diagnostics ABI mismatch')
            self.present_events = [
                {key: getattr(event, key) for key, _ in event._fields_}
                for event in events[:count]]
        return list(self.present_events)

    def status(self):
        with self.condition:
            result = super().status()
            s = self._stats()
            return {**result, 'transport': 'spout', 'name': self.name, 'gpu_conversion': True,
                    'pixel_format': 'RGBA', 'raw_download_bytes': 0, 'device': str(self.device),
                    'sent': s.get('sent', 0),
                    'queued': len(self.frames) + (s.get('queued', 0) if self.native else 0),
                    'max_frames': self.max_frames + self.native_capacity,
                    'last_send_at': s.get('last_send_at') or None,
                    'last_send_ms': s.get('send_ms', 0),
                    'underruns': s.get('underruns', 0),
                    'feeder': self.feeder_trace.snapshot(),
                    'copy_priority': self.applied_copy_priority,
                    'transition_reserve': self.transition_reserve,
                    'target_native_reserve': self._native_reserve(),
                    'profile_preparation': self.profile_preparation,
                    'repeated_transition_frames': self.repeated_transition_frames,
                    'native_present_trace_supported': hasattr(self.dll, 'jsp_present_trace'),
                    'native_present_events': self._presentation_events(),
                    'frame_trace_version': 1 if hasattr(self.dll, 'jsp_frame_trace') else 0,
                    'publisher_id': self.publisher_id,
                    'native': s}

    def warmup(self, size):
        # Prepare transition kernels before playback, rather than paying their
        # first-use cost immediately after clearing the old scene's queue.
        with torch.cuda.device(self.device):
            h, w = size
            shader = self._resize_source(np.zeros((360, 648, 3), dtype=np.uint8), (w, h, 24))
            self._blend(shader, shader, .5).ready.synchronize()

    def submit(self, frames, fps):
        if frames and isinstance(frames[0], np.ndarray):
            with torch.cuda.device(self.device):
                frames = [GpuFrame.record(torch.from_numpy(f).to(self.device)) for f in frames]
        return super().submit(frames, fps)

    def begin_transition(self, source, *, moving=False):
        with self.condition:
            s = self._stats()
            shown = self.retained.get(s.get('last_token'))
            if shown is not None:
                self.last_frame, self.last_key = shown
            accepted = super().begin_transition(source, moving=moving)
            if accepted:
                self.replace_next = True
            return accepted

    def reset_scene(self):
        with self.condition:
            if self.transition is None:
                return False
            # The native queue already contains the moving shader handoff.
            # Keep it playing while discarding only old generated input frames.
            return super().reset_scene()

    @torch.inference_mode()
    def _run(self):
        key = None
        try:
            with torch.cuda.device(self.device):
                self.stream = torch.cuda.Stream(device=self.device, priority=-1)
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                while True:
                    with self.condition:
                        if self.stopping:
                            return
                        # close() retains the last counters for diagnostics, but
                        # that retired native queue cannot throttle a new worker.
                        s = self._stats() if self.native else {}
                        if self.native:
                            error = self.dll.jsp_error(self.native).decode()
                            if error:
                                raise RuntimeError(error)
                        # This feeder may wait; the separate native thread continues
                        # presenting its owned textures without Python or the GIL.
                        reserve = self._native_reserve()
                        if ((s.get('queued', 0) >= reserve and not self.replace_next) or
                                (not self.frames and self.transition is None)):
                            self.condition.wait(.004)
                            continue
                        context = dict(queued=s.get('queued', 0),
                                       generated_queued=len(self.frames),
                                       transition=self.transition is not None)
                        profile = self.profile_preparation
                        if profile:
                            if self.preparation_events is None:
                                self.preparation_events = tuple(
                                    torch.cuda.Event(enable_timing=True) for _ in range(3))
                            self.preparation_events[0].record(self.stream)
                        with self.feeder_trace.measure('select', **context):
                            selected = self._next_frame(time.perf_counter())
                        if selected is None:
                            self.condition.wait(.004)
                            continue
                        if profile:
                            self.preparation_events[1].record(self.stream)
                        frame, next_key = selected
                        repeated_transition = context['transition'] and frame is self.last_frame
                        if next_key != key:
                            if self.native:
                                self.dll.jsp_close(self.native)
                            w, h, fps = next_key
                            # Ensure the output CUDA primary context is current.
                            rgba = torch.empty((h, w, 4), dtype=torch.uint8, device=self.device)
                            self.native = self.dll.jsp_create(self.name.encode(), w, h, fps, self.native_capacity, 12)
                            if not self.native:
                                raise RuntimeError(self.dll.jsp_error(None).decode())
                            key = self.geometry = next_key
                            self.retained.clear()
                            self.publisher_id = uuid.uuid4().hex
                            self.frame_origins.clear()
                            self.present_events.clear()
                            self.applied_copy_priority = 0
                        replace, self.replace_next = self.replace_next, False
                        copy_priority = self.copy_priority
                    if copy_priority != self.applied_copy_priority:
                        if self.dll.jsp_set_copy_priority(self.native, copy_priority) != 1:
                            raise RuntimeError(self.dll.jsp_error(None).decode() or 'Spout copy priority update failed')
                        self.applied_copy_priority = copy_priority
                    # CUDA waits must not own the producer/scene/status lock.
                    # This worker alone creates, submits to and closes native;
                    # close() signals it and joins before resources are released.
                    input_frame_ready = bool(frame.ready.query()) if profile else None
                    with self.feeder_trace.measure('pack', **context):
                        rgba[..., :3].copy_(frame.wait())
                        rgba[..., 3].fill_(255)
                        ready = self.preparation_events[2] if profile else torch.cuda.Event()
                        ready.record(self.stream)
                    self.sequence += 1
                    # Register before native submission: the native presenter can
                    # publish while ctypes has released the GIL. A diagnostic
                    # read must never observe a published token without its ID.
                    with self.condition:
                        origin = getattr(frame, 'origin', None)
                        self.frame_origins[self.sequence] = origin.as_dict() if origin else None
                        while len(self.frame_origins) > 512:
                            self.frame_origins.popitem(last=False)
                    # Native submit returns only after its independent texture
                    # owns a copy. Reusing rgba cannot race presentation.
                    send = self.dll.jsp_replace if replace else self.dll.jsp_submit
                    with self.feeder_trace.measure('interop', **context) as details:
                        outcome = send(self.native, rgba.data_ptr(), ready.cuda_event, self.sequence)
                        if profile and outcome == 1:
                            # Native already waited for this ready event before
                            # owning the copy. Reading elapsed times adds no wait.
                            start_event, selected_event, packed_event = self.preparation_events
                            details.update(
                                input_frame_ready=input_frame_ready,
                                select_gpu_span_ms=start_event.elapsed_time(selected_event),
                                pack_gpu_span_ms=selected_event.elapsed_time(packed_event),
                                prepare_gpu_span_ms=start_event.elapsed_time(packed_event))
                        if hasattr(self.dll, 'jsp_copy_trace'):
                            native_trace = NativeCopyTrace()
                            if self.dll.jsp_copy_trace(self.native, ctypes.byref(native_trace), ctypes.sizeof(native_trace)):
                                if native_trace.token == self.sequence:
                                    details.update({name: getattr(native_trace, name)
                                                    for name, _ in native_trace._fields_})
                    if outcome != 1:
                        raise RuntimeError(self.dll.jsp_error(self.native).decode() or 'Spout staging queue full')
                    with self.feeder_trace.measure('retire', **context):
                        with self.condition:
                            if repeated_transition:
                                # Immutable GpuFrame identity proves the same
                                # payload was staged twice; no pixel readback.
                                self.repeated_transition_frames += 1
                            self.retained[self.sequence] = (frame, key)
                            current = self._stats().get('last_token', 0)
                            self.retained = {token: item for token, item in self.retained.items() if token >= current}
                            self.last_frame, self.last_key = frame, key
        except Exception as exc:
            with self.condition:
                if not self.stopping:
                    self.error = self.last_error = f'{type(exc).__name__}: {exc}'
                    self.error_at = time.monotonic()
                self.frames.clear()
        finally:
            with self.condition:
                if self.native:
                    self._stats()
                    self.dll.jsp_close(self.native)
                    self.native = None
                self.retained.clear()
                self.last_frame = self.last_key = None
                self.thread = None
                self.playback_primed = False
