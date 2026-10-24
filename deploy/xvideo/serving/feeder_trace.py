"""Bounded CPU call spans for one output feeder; no GPU waits or disk I/O.

A long span reports where the feeder waited, not whether its underlying cause
was GPU execution, driver synchronization, Python scheduling or the GIL.
Snapshots never take the publisher's lock. The one writer replaces immutable
tuples/dicts so a health reader cannot delay the timed work.
"""
from contextlib import contextmanager
import time


class FeederTrace:
    def __init__(self, capacity=64, threshold_ms=25, clock=time.perf_counter,
                 wall=time.time):
        self.capacity = capacity
        self.threshold_ms = threshold_ms
        self.clock, self.wall = clock, wall
        self.current = None
        self.recent = ()
        self.totals = {}
        self.serial = 0

    @contextmanager
    def measure(self, stage, **context):
        begin, at = self.clock(), self.wall()
        self.current = dict(stage=stage, begin=begin, at=at, **context)
        details = {}
        try:
            yield details
        finally:
            elapsed = max(0., (self.clock() - begin) * 1000)
            previous = self.totals.get(stage, {})
            self.totals = {**self.totals, stage: dict(
                count=previous.get('count', 0) + 1,
                total_ms=previous.get('total_ms', 0.) + elapsed,
                max_ms=max(previous.get('max_ms', 0.), elapsed))}
            if elapsed >= self.threshold_ms:
                self.serial += 1
                row = dict(serial=self.serial, stage=stage, at=at,
                           duration_ms=elapsed, **context)
                if details:
                    row['details'] = dict(details)
                self.recent = (*self.recent, row)[-self.capacity:]
            self.current = None

    def snapshot(self):
        current = self.current
        return dict(current=({k: v for k, v in current.items() if k != 'begin'} |
                             {'age_ms': max(0., (self.clock()-current['begin'])*1000)})
                    if current else None,
                    totals=self.totals, events=list(self.recent),
                    threshold_ms=self.threshold_ms,
                    scope='CPU call spans; includes GPU/driver/GIL waits, not GPU execution times')
