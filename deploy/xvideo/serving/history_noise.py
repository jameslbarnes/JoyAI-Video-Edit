"""History noise: store each finished chunk's attention memory from a slightly
re-noised latent instead of the exact output.

Each chunk is denoised in two steps while attending to the stored KV of the
previous chunk. That KV was computed from the previous output as if it were
perfectly clean (timestep 0), so any fine detail the model invents is copied
forward and amplified; on 2026-10-02 offline replays reproduced the live
"mosaic" junk within ~2 minutes and isolated it to the previous-chunk memory.
Storing that memory from (1 - sigma) * clean + sigma * noise at timestep
sigma * 1000 (the scheduler's flow-matching convention) makes the next chunk
treat inherited detail as uncertain. sigma 0 is exactly today's behavior.

Modes: 'full' mixes noise into the whole latent; 'fine' (default) replaces
only the fine detail (latent minus a 5x5 blur) with energy-matched noise, so
broad shapes and motion carry over intact. Replays 2026-10-03: fine at 0.20
had less junk than full at 0.20 (0.047 vs 0.052 at 3 min) and a 34% smaller
chunk-boundary jump than the original pipeline (full 0.20 was 60% larger).
"""
import json
import threading

MAX_SIGMA = 0.5
MODES = ('fine', 'full')


def validate_sigma(sigma):
    if isinstance(sigma, bool):
        raise ValueError('history noise sigma must be a number')
    sigma = float(sigma)
    if not 0 <= sigma <= MAX_SIGMA:
        raise ValueError(f'history noise sigma must be between 0 and {MAX_SIGMA}')
    return sigma


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError('history noise mode must be fine or full')
    return mode


def load_history_noise_config(path):
    if not path.exists():
        return {'sigma': 0.0, 'mode': 'fine'}
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    return {'sigma': validate_sigma(config.get('sigma', 0)), 'mode': validate_mode(config.get('mode', 'fine'))}


_BLUR_KERNELS = {}


def _blur_kernel(device, dtype, channels):
    # Built once per device/dtype/channels and reused: creating it inside a
    # CUDA graph capture is a host-to-device copy, which invalidates the
    # capture (2026-10-03). The graph runner's warm-up pass builds it first.
    key = (str(device), dtype, channels)
    kernel = _BLUR_KERNELS.get(key)
    if kernel is None:
        import torch
        k1 = torch.tensor([1., 4., 6., 4., 1.], dtype=dtype) / 16.0
        kernel = (k1[:, None] * k1[None, :]).expand(channels, 1, 5, 5).contiguous().to(device)
        _BLUR_KERNELS[key] = kernel
    return kernel


def latent_blur(x):
    """5x5 binomial blur over the spatial dims of a (B, C, T, H, W) latent."""
    import torch
    b, c, t, h, w = x.shape
    k = _blur_kernel(x.device, x.dtype, c)
    flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    out = torch.nn.functional.conv2d(torch.nn.functional.pad(flat, (2, 2, 2, 2), mode='replicate'), k, groups=c)
    return out.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def unit_fine_noise(shape, generator, device):
    """High-pass Gaussian noise normalized to unit std (scaled to the clean
    latent's fine-detail energy at mix time)."""
    import torch
    n = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    high = n - latent_blur(n)
    return high / high.std().clamp_min(1e-6)


def mix_history(clean32, noise32, sigma, fine):
    """Store-input mix shared by the eager path and the captured graph.
    sigma/fine may be numbers or 0-dim tensors (graph buffers); noise32 is
    unit fine noise when fine, plain Gaussian otherwise. At sigma 0 the result
    equals clean32 exactly in both modes."""
    high = clean32 - latent_blur(clean32)
    fine_mix = clean32 + sigma * (noise32 * high.std() - high)
    full_mix = clean32 * (1 - sigma) + noise32 * sigma
    return fine * fine_mix + (1 - fine) * full_mix


def noisy_history_latents(clean, sigma, generator=None, mode='fine'):
    """(latents, timestep) for the history KV store. sigma 0 returns ``clean``
    itself and timestep 0, without touching the generator."""
    if sigma <= 0:
        return clean, 0.0
    import torch
    fine = 1.0 if mode == 'fine' else 0.0
    noise = (unit_fine_noise(clean.shape, generator, clean.device) if fine
             else torch.randn(clean.shape, generator=generator, device=clean.device, dtype=torch.float32))
    return mix_history(clean.float(), noise, sigma, fine).to(clean.dtype), sigma * 1000.0


class HistoryNoise:
    def __init__(self, sigma=0.0, mode='fine'):
        self._lock = threading.Lock()
        self._sigma = validate_sigma(sigma)
        self._mode = validate_mode(mode)
        self._revision = 0

    @property
    def sigma(self):
        with self._lock:
            return self._sigma

    @property
    def mode(self):
        with self._lock:
            return self._mode

    def set(self, sigma=None, mode=None):
        with self._lock:
            new_sigma = self._sigma if sigma is None else validate_sigma(sigma)
            new_mode = self._mode if mode is None else validate_mode(mode)
            self._sigma, self._mode = new_sigma, new_mode
            self._revision += 1
            return self._snapshot_locked()

    def snapshot(self):
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self):
        return {'sigma': self._sigma, 'mode': self._mode, 'enabled': self._sigma > 0, 'revision': self._revision}
