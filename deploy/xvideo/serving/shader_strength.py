"""Bounded shader-influence control. Neutral preserves the original pipeline."""
import math
import threading


def validate_strength(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('shader strength must be a number in [0, 1]')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('shader strength must be finite in [0, 1]')
    return value


class ShaderStrength:
    # Dial percentages are normalized control values, NOT latent blend percent.
    MAX_BLEND = .005
    MIN_REFERENCE_GAIN = .75
    MAX_CHUNK_CHANGE = .125

    def __init__(self, value=.5):
        self.lock = threading.Lock()
        self.current = self.target = validate_strength(value)
        self.revision = 0

    def set(self, value):
        value = validate_strength(value)
        with self.lock:
            self.target = value
            self.revision += 1
            return self._snapshot()

    def snapshot(self):
        with self.lock:
            return self._snapshot()

    def advance(self):
        with self.lock:
            delta = max(-self.MAX_CHUNK_CHANGE, min(self.MAX_CHUNK_CHANGE, self.target-self.current))
            self.current += delta
            return self._snapshot()

    def _snapshot(self):
        v = self.current
        return {'value': v, 'target': self.target, 'revision': self.revision,
                'settled': abs(v-self.target) < 1e-9,
                'blend': self.MAX_BLEND * max(0, 2*v-1)**2,
                'referenceGain': 1 - (1-self.MIN_REFERENCE_GAIN) * max(0, 1-2*v),
                'neutral': .5, 'cachePreserved': True}
