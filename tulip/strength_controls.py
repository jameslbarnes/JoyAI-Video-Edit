"""Experimental pre-DiT controls; not enabled in the live provider.

Zero blend / unit reference gain is an exact no-op. Reference gain scales
encoded video, not attention weights, and is NOT a calibrated adherence dial.
"""
import math

import torch


def prepare_strength_inputs(noise, reference, *, blend=0.0, reference_gain=1.0):
    blend, reference_gain = float(blend), float(reference_gain)
    if not math.isfinite(blend) or not 0 <= blend <= 1:
        raise ValueError('blend must be finite in [0, 1]')
    if not math.isfinite(reference_gain) or not 0 <= reference_gain <= 1:
        raise ValueError('reference_gain must be finite in [0, 1]')
    if noise.shape != reference.shape:
        raise ValueError('source and generated latent shapes must match')
    if noise.device != reference.device or noise.dtype != reference.dtype:
        raise ValueError('source and generated latents must share device/dtype')
    # Out-of-place: never mutate VAE state, reference latents or cache buffers.
    return (torch.lerp(noise, reference, blend) if blend else noise,
            reference * reference_gain if reference_gain != 1 else reference)


def strength_for_chunk(arm, chunk):
    if arm == 'fine_ramp':
        # Smooth triangular cycle across 65 chunks; return to neutral.
        return (max(0, 1 - abs(chunk - 32) / 32) * .03, 1.0)
    if arm == 'ref_ramp':
        return (0.0, 1 - max(0, 1 - abs(chunk - 32) / 32) * .5)
    if arm == 'strength_ramp':
        # 0 -> .25 -> .5 -> 0 in one persistent 65-chunk session.
        return ([0, .25, .5, 0][min(chunk // 16, 3)], 1.0)
    if arm.startswith('blend'):
        return (float(arm[5:]) / 100, 1.0)
    if arm.startswith('ref'):
        return (0.0, int(arm[3:]) / 100)
    raise ValueError(f'Unknown strength arm: {arm}')
