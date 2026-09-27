"""Bounded replay of generated motion, with forward overlaps instead of holds.

Frames stay on their owner's device. The caller supplies a blend operation and
advances once per output frame; this class neither copies pixels nor owns a clock.
"""


class MotionBridge:
    def __init__(self, frames, blend, fps, previous):
        if len(frames) < 2:
            raise ValueError('A motion bridge requires at least two generated frames')
        self.frames = tuple(frames)
        self.blend = blend
        self.overlap = max(1, min(round(fps * .25), len(frames) // 3))
        self.index = 0
        self.entry = previous
        self.cycles = 0

    def next(self):
        i, n, overlap = self.index, len(self.frames), self.overlap
        frame = self.frames[i]
        if self.entry is not None and i < overlap:
            # Enter the moving segment from the exact preceding output. Every
            # step changes the incoming image; no last-frame repetition is used.
            frame = self.blend(self.entry, frame, (i + 1) / overlap)
        elif i >= n - overlap:
            j = i - (n - overlap)
            frame = self.blend(frame, self.frames[j], (j + 1) / overlap)
        self.index += 1
        if self.index == n:
            self.index = overlap
            self.entry = None
            self.cycles += 1
        return frame
