"""Gradual forgetting of the scene's first-chunk attention sink.

The streaming window attends to [sink, previous, current]. The sink is the first
chunk of a scene and is otherwise kept for the whole scene, so structure it
captured can persist indefinitely. Decay moves the sink toward the latest
committed chunk a little every chunk, so old content fades with a configurable
half-life instead of being dropped by a scene reset.

Modes (per chunk, alpha derived from the half-life):
  blend    sink = lerp(sink, current, alpha) on pre-RoPE K/V (RoPE is linear).
  dissolve a random alpha fraction of sink tokens is replaced by the current
           chunk's tokens at the same spatial positions (no averaged vectors).
Half-life 0 disables decay. Settings can change live; they apply next chunk.
"""
import json
import threading

MODES = ('blend', 'dissolve')


def validate(mode, half_life_s):
    if mode not in MODES:
        raise ValueError('cache decay mode must be blend or dissolve')
    half_life_s = float(half_life_s)
    if not half_life_s >= 0 or half_life_s == float('inf'):
        raise ValueError('cache decay half_life_s must be a finite number >= 0')
    return mode, half_life_s


def validate_every(every_chunks):
    if isinstance(every_chunks, bool) or int(every_chunks) != every_chunks or not 1 <= every_chunks <= 64:
        raise ValueError('cache decay every_chunks must be an integer from 1 to 64')
    return int(every_chunks)


def load_decay_config(path):
    if not path.exists():
        return {'mode': 'blend', 'half_life_s': 0.0}
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    mode, half_life_s = validate(config.get('mode', 'blend'), config.get('half_life_s', 0))
    loaded = {'mode': mode, 'half_life_s': half_life_s}
    if 'every_chunks' in config:
        loaded['every_chunks'] = validate_every(config['every_chunks'])
    return loaded


def chunk_alpha(half_life_s, chunk_s):
    """Fraction of the sink replaced per chunk so that the old share halves every half_life_s."""
    if half_life_s <= 0 or chunk_s <= 0:
        return 0.0
    return 1.0 - 0.5 ** (chunk_s / half_life_s)


class CacheDecay:
    def __init__(self, mode='blend', half_life_s=0.0, native_fps=12.0, every_chunks=4):
        self._lock = threading.Lock()
        self._mode, self._half_life_s = validate(mode, half_life_s)
        if not float(native_fps) > 0:
            raise ValueError('cache decay native_fps must be > 0')
        self.native_fps = float(native_fps)  # generated frames/s; chunk seconds = frames / this
        # Each update re-seeds the graph pool (~36 ms at 1152x648); stepping every few
        # chunks with a proportionally larger alpha keeps the same half-life.
        self._every_chunks = validate_every(every_chunks)
        self._revision = 0

    def set(self, half_life_s=None, mode=None, every_chunks=None):
        with self._lock:
            mode, half_life_s = validate(self._mode if mode is None else mode,
                                         self._half_life_s if half_life_s is None else half_life_s)
            every = self._every_chunks if every_chunks is None else validate_every(every_chunks)
            self._mode, self._half_life_s, self._every_chunks = mode, half_life_s, every
            self._revision += 1
            return self._snapshot_locked()

    def snapshot(self):
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self):
        return {'mode': self._mode, 'half_life_s': self._half_life_s, 'native_fps': self.native_fps,
                'every_chunks': self._every_chunks,
                'enabled': self._half_life_s > 0, 'revision': self._revision}


def decay_sink(sink, current, alpha, mode, generator=None):
    """In-place decay of sink K/V entries toward current (no host sync).

    Both are {layer: {"key": T, "value": T, ...}} with token dim 1 and the same
    RoPE convention. The caller must own sink storage (never an alias of rotating
    graph publication buffers).
    """
    import torch
    if alpha <= 0:
        return
    if mode == 'blend':
        for li, entry in sink.items():
            src = current[li]
            entry['key'].lerp_(src['key'].to(entry['key'].dtype), alpha)
            entry['value'].lerp_(src['value'].to(entry['value'].dtype), alpha)
        return
    first = next(iter(sink.values()))['key']
    tokens = first.shape[1]
    mask = torch.rand(tokens, device=first.device, generator=generator) < alpha
    for li, entry in sink.items():
        src = current[li]
        for name in ('key', 'value'):
            dst = entry[name]
            shape = [1] * dst.dim()
            shape[1] = tokens
            dst.copy_(torch.where(mask.view(shape), src[name].to(dst.dtype), dst))


def own_sink(sink):
    """Independent copy of a sink entry dict, safe to modify in place."""
    return {li: {**entry, 'key': entry['key'].clone(), 'value': entry['value'].clone()}
            for li, entry in sink.items()}


def rope_compatible(sink, current):
    """Blending is only meaningful on position-free (pre-RoPE) keys in both entries."""
    return all(sink[li].get('pre_rope', False) and current[li].get('pre_rope', False)
               for li in sink)
