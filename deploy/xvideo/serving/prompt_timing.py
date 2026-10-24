"""Bounded prompt provenance. No text, pixels, file I/O or GPU synchronization.

Times are host Unix milliseconds. Collectors must reject clock discontinuities.
"""
from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import threading
import time
import uuid


@dataclass(frozen=True)
class FrameOrigin:
    session: str
    revision: int
    prompt_hash: str
    chunk: int
    requested_ms: float
    applied_ms: float
    encode_started_ms: float | None = None
    encode_finished_ms: float | None = None
    generated_fraction: float = 1.0

    def as_dict(self):
        return dict(self.__dict__)


def blended_origin(first, second, mix):
    """A shader-only handoff is not evidence of the new generated scene."""
    if mix <= 0:
        return first
    if mix >= 1:
        return second
    if second is not None:
        return replace(second, generated_fraction=second.generated_fraction * mix)
    if first is not None:
        return replace(first, generated_fraction=first.generated_fraction * (1 - mix))
    return None


class PromptTiming:
    def __init__(self, prompt, *, clock=lambda: time.time() * 1000, capacity=64):
        self.clock, self.capacity = clock, capacity
        self.session = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.requests = OrderedDict()
        self.active = None
        self.requested(0, prompt)

    def requested(self, revision, prompt):
        with self.lock:
            self.requests[revision] = dict(prompt_hash=hashlib.sha256(prompt.encode('utf8')).hexdigest(),
                                           requested_ms=self.clock())
            while len(self.requests) > self.capacity:
                self.requests.popitem(last=False)

    def stage(self, revision, stage):
        if stage not in ('encode_started_ms', 'encode_finished_ms'):
            raise ValueError('Unknown prompt timing stage')
        with self.lock:
            if revision in self.requests:
                self.requests[revision][stage] = self.clock()

    def applied(self, revision, chunk):
        with self.lock:
            request = self.requests.get(revision)
            self.active = FrameOrigin(self.session, revision, chunk=chunk,
                                      applied_ms=self.clock(), **request) if request else None

    def frame(self, revision, chunk):
        with self.lock:
            if self.active is None and revision == 0:
                request = self.requests.get(0)
                if request:
                    self.active = FrameOrigin(self.session, 0, chunk=chunk,
                                              applied_ms=self.clock(), **request)
            return replace(self.active, chunk=chunk) if self.active and self.active.revision == revision else None
