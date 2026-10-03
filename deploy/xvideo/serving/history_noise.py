"""History noise: store each finished chunk's attention memory from a slightly
re-noised latent instead of the exact output.

Each chunk is denoised in two steps while attending to the stored KV of the
previous chunk. That KV was computed from the previous output as if it were
perfectly clean (timestep 0), so any fine detail the model invents is copied
forward and amplified; on 2026-10-02 offline replays reproduced the live
"mosaic" junk within ~2 minutes and isolated it to the previous-chunk memory.
Storing that memory from (1 - sigma) * clean + sigma * noise at timestep
sigma * 1000 (the scheduler's flow-matching convention) makes the next chunk
treat inherited detail as uncertain: sigma 0.20 removed the drift over 4 min
with continuity close to baseline. sigma 0 is exactly today's behavior.
"""
import json
import threading

MAX_SIGMA = 0.5


def validate_sigma(sigma):
    if isinstance(sigma, bool):
        raise ValueError('history noise sigma must be a number')
    sigma = float(sigma)
    if not 0 <= sigma <= MAX_SIGMA:
        raise ValueError(f'history noise sigma must be between 0 and {MAX_SIGMA}')
    return sigma


def load_history_noise_config(path):
    if not path.exists():
        return {'sigma': 0.0}
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    return {'sigma': validate_sigma(config.get('sigma', 0))}


def noisy_history_latents(clean, sigma, generator=None):
    """(latents, timestep) for the history KV store. sigma 0 returns ``clean``
    itself and timestep 0, without touching the generator."""
    if sigma <= 0:
        return clean, 0.0
    import torch
    noise = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=torch.float32)
    return ((1 - sigma) * clean.float() + sigma * noise).to(clean.dtype), sigma * 1000.0


class HistoryNoise:
    def __init__(self, sigma=0.0):
        self._lock = threading.Lock()
        self._sigma = validate_sigma(sigma)
        self._revision = 0

    @property
    def sigma(self):
        with self._lock:
            return self._sigma

    def set(self, sigma):
        with self._lock:
            self._sigma = validate_sigma(sigma)
            self._revision += 1
            return self._snapshot_locked()

    def snapshot(self):
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self):
        return {'sigma': self._sigma, 'enabled': self._sigma > 0, 'revision': self._revision}
