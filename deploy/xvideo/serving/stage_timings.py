"""Opt-in CUDA stage spans collected after completion, without synchronizing work."""
import collections
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import threading
import time

import torch


class StageTimings:
    def __init__(self, config_path=None, capacity=256):
        self.config_path = Path(config_path) if config_path else Path(__file__).resolve().parents[3] / 'tulip/stage-profile.json'
        self.capacity = capacity
        self.pending = collections.deque()
        self.completed = collections.deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.enabled = False
        self.checked_at = 0.
        self.skipped = 0
        self.error = None

    def _refresh(self):
        now = time.monotonic()
        if now - self.checked_at < 1: return
        self.checked_at = now
        try:
            cfg = json.loads(self.config_path.read_text(encoding='utf-8-sig')) if self.config_path.exists() else {}
            if type(cfg.get('enabled', False)) is not bool: raise ValueError('enabled must be boolean')
            self.enabled = cfg.get('enabled', False)
            self.error = None
        except (OSError, ValueError, AttributeError) as exc:
            self.enabled = False
            self.error = str(exc)

    def _collect(self):
        waiting = collections.deque()
        while self.pending:
            start, end, row = self.pending.popleft()
            if end.query():
                self.completed.append({**row, 'gpu_ms': start.elapsed_time(end)})
            else:
                waiting.append((start, end, row))
        self.pending = waiting

    @contextmanager
    def measure(self, name, device, chunk_idx, **tags):
        with self.lock:
            self._refresh()
            self._collect()
            enabled = self.enabled and chunk_idx is not None and chunk_idx >= 4
            if len(self.pending) >= self.capacity:
                self.skipped += int(enabled)
                enabled = False
        if not enabled:
            yield
            return
        device = torch.device(device)
        stream = torch.cuda.current_stream(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        started = time.perf_counter()
        try:
            yield
        finally:
            end.record(stream)
            row = {'stage': name, 'device': str(device), 'chunk': chunk_idx,
                   'at': time.time(), 'submission_ms': (time.perf_counter()-started)*1000, **tags}
            with self.lock:
                if len(self.pending) < self.capacity:
                    self.pending.append((start, end, row))
                else:
                    self.skipped += 1

    def snapshot(self):
        with self.lock:
            self._refresh()
            self._collect()
            rows = list(self.completed)
            groups = {}
            for row in rows:
                groups.setdefault(row['stage'] + '@' + row['device'], []).append(row['gpu_ms'])
            summary = {name: {'count': len(values), 'medianMs': statistics.median(values),
                       'meanMs': statistics.mean(values), 'p95Ms': sorted(values)[min(len(values)-1, int(len(values)*.95))]}
                       for name, values in groups.items()}
            return {'enabled': self.enabled, 'error': self.error, 'pending': len(self.pending),
                    'skipped': self.skipped, 'summary': summary, 'samples': rows,
                    'note': 'CUDA stream spans, including launch gaps/contention; no synchronization added. Startup chunks <4 excluded.'}
