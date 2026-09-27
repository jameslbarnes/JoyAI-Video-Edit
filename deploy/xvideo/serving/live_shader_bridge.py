"""One replaceable CPU shader frame, independent of model input/backpressure."""
import io
import time
import numpy as np
from PIL import Image


class LiveShaderBridge:
    def __init__(self):
        self.frame = None
        self.updated_at = 0.
        self.frames = 0
        self.max_gap_ms = 0.

    def publish(self, jpeg):
        if not 0 < len(jpeg) <= 512 * 1024:
            raise ValueError('Shader JPEG must be at most 512 KiB')
        with Image.open(io.BytesIO(jpeg)) as image:
            if image.size != (648, 360):
                raise ValueError('Shader bridge requires 648x360')
            pixels = np.array(image.convert('RGB'), copy=True)
        now = time.perf_counter()
        if self.updated_at:
            self.max_gap_ms = max(self.max_gap_ms, (now-self.updated_at)*1000)
        self.frame, self.updated_at = pixels, now
        self.frames += 1

    def latest(self):
        return self.frame if time.perf_counter()-self.updated_at < .5 else None

    def status(self):
        return {'frames': self.frames, 'max_gap_ms': self.max_gap_ms,
                'age_ms': (time.perf_counter()-self.updated_at)*1000 if self.updated_at else None}
