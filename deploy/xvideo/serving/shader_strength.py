"""Chunk-smoothed shader influence and independently controlled presentation."""
import json
import math
import os
from pathlib import Path
import tempfile
import threading

MODES = ('direct_latent_blend', 'reference_only', 'combined', 'influence')


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError('shader strength mode must be ' + ', '.join(MODES))
    return mode


def load_strength_config(path):
    if not path.exists():
        return {'mode': 'direct_latent_blend', 'value': 0}
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    result = {'mode': validate_mode(config['mode']), 'value': validate_strength(config['value'])}
    if result['mode'] == 'influence':
        result['output_mix'] = validate_strength(config.get('outputMix', 0))
    return result


def save_strength_config(path, config):
    """Durable file contents, then atomic replacement. Called off the frame lock."""
    path = Path(path)
    # Never truncate the last accepted setting, including on disk-full failures.
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as file:
            temp = Path(file.name)
            json.dump(config, file, indent=2, allow_nan=False)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def migrate_strength_state(state):
    """Map a settled legacy dial without changing its actual inputs or overlay."""
    if not state.get('settled'):
        raise ValueError('Wait for shader strength to settle before migration')
    validate_mode(state['mode'])
    blend = validate_strength(state['blend'])
    gain = validate_strength(state['referenceGain'])
    overlay = validate_strength(state['outputMix'])
    if blend == 0:
        value = gain * .5
    elif gain == 1 and blend <= .1:
        value = .5 + .5 * math.sqrt(blend / .1)
    else:
        raise ValueError('Current shader inputs cannot be preserved by the influence curve')
    return {'mode': 'influence', 'value': value, 'outputMix': overlay}


def prepare_strength_inputs(noise, reference, state):
    """Preserve source/cache tensors; apply equally before eager or graph inference."""
    import torch
    blend, gain = state['blend'], state['referenceGain']
    generated = torch.lerp(noise, reference, blend) if blend else noise
    conditioned = (torch.zeros_like(reference) if gain == 0 else
                   reference * gain if gain != 1 else reference)
    return generated, conditioned


def strength_inputs(mode, value):
    """Initial artistic curve; input units are not perceptual percentages."""
    if mode == 'influence':
        # One curve for all clients; useful reference range first, then a gentler
        # source/noise blend. Raw presentation is an independent control.
        return .1 * max(0.0, 2 * value - 1) ** 2, min(2 * value, 1.0), 0.0
    if mode == 'combined':
        if value == 1:
            return .1, 1.0, 1.0
        reference = min(value / .6, 1.0)
        upper = max(0.0, (value - .6) / .4)
        blend = .1 * upper ** 4
        fade = max(0.0, min(1.0, (value - .9) / .1))
        return blend, reference, fade * fade * (3 - 2 * fade)
    return (0.0, value, 0.0) if mode == 'reference_only' else (value, 1.0, 0.0)


def mix_shader_presentation(generated, source, state):
    """Display-only THWC uint8 blend. Never mutate history or download pixels."""
    import torch
    current, previous = state.get('outputMix', 0.0), state.get('outputMixPrevious', 0.0)
    if current == previous == 0:
        return generated
    if source is None or source.shape != generated.shape or source.dtype != torch.uint8:
        raise ValueError('Raw shader presentation requires matched source frames')
    source = source.to(generated.device)
    if current == previous == 1:
        return source
    weights = torch.arange(1, len(generated)+1, device=generated.device, dtype=torch.float32)
    weights = (previous + (current-previous) * weights / len(generated)).view(-1,1,1,1)
    mixed = torch.lerp(generated.float(), source.float(), weights)
    return mixed.round_().clamp_(0,255).to(torch.uint8)


def validate_strength(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('shader strength must be a number in [0, 1]')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('shader strength must be finite in [0, 1]')
    return value


class ShaderStrength:
    MAX_CHUNK_CHANGE = .125

    def __init__(self, value=0, mode='direct_latent_blend', output_mix=0, config_path=None):
        self.lock = threading.Lock()
        self.control_lock = threading.Lock()
        self.config_path = Path(config_path) if config_path is not None else None
        self.mode = validate_mode(mode)
        self.overlay_target = validate_strength(output_mix)
        self.current = self.target = validate_strength(value)
        self.blend, self.reference_gain, self.output_mix = self._inputs(self.current)
        self.previous_output_mix = self.output_mix
        self.mode_transition = False
        self.revision = 0

    def _inputs(self, value):
        blend, gain, overlay = strength_inputs(self.mode, value)
        return blend, gain, self.overlay_target if self.mode == 'influence' else overlay

    def set(self, value, mode=None, output_mix=None):
        value = validate_strength(value)
        if mode is not None:
            validate_mode(mode)
        if output_mix is not None:
            output_mix = validate_strength(output_mix)
        return self._set(value=value, mode=mode, output_mix=output_mix)

    def set_overlay(self, value):
        return self._set(output_mix=validate_strength(value))

    def _set(self, value=None, mode=None, output_mix=None):
        # Serialize accepted commands, but never block inference on filesystem I/O.
        with self.control_lock:
            with self.lock:
                new_mode = self.mode if mode is None else mode
                new_value = self.target if value is None else value
                overlay = self.overlay_target
                if new_mode == 'influence' and self.mode != new_mode:
                    overlay = self.output_mix
                if output_mix is not None:
                    if new_mode != 'influence':
                        raise ValueError('Independent shader overlay requires influence mode')
                    overlay = output_mix
            config = {'mode': new_mode, 'value': new_value}
            if new_mode == 'influence':
                config['outputMix'] = overlay
            if self.config_path is not None:
                save_strength_config(self.config_path, config)
            with self.lock:
                self.mode_transition = self.mode_transition or new_mode != self.mode
                self.mode, self.target, self.overlay_target = new_mode, new_value, overlay
                self.revision += 1
                return self._snapshot()

    def snapshot(self):
        with self.lock:
            return self._snapshot()

    def advance(self):
        with self.lock:
            def approach(value, target):
                return value + max(-self.MAX_CHUNK_CHANGE, min(self.MAX_CHUNK_CHANGE, target-value))
            self.current = approach(self.current, self.target)
            # Follow the nonlinear curve within a mode. Across modes, move
            # directly between actual settings, not unrelated dial coordinates.
            blend, gain, output_mix = self._inputs(self.target if self.mode_transition else self.current)
            self.blend = approach(self.blend, blend)
            self.reference_gain = approach(self.reference_gain, gain)
            self.previous_output_mix = self.output_mix
            # Presentation already ramps across every native frame, followed by
            # RIFE. Avoid a second chunk ramp that delays the endpoint another 5s.
            self.output_mix = output_mix
            state = self._snapshot()
            if state['settled']:
                self.mode_transition = False
            return state

    def _snapshot(self):
        v = self.current
        blend_target, gain_target, output_target = self._inputs(self.target)
        return {'value': v, 'target': self.target, 'revision': self.revision,
                'settled': all(abs(a-b) < 1e-9 for a, b in
                    ((v, self.target), (self.blend, blend_target), (self.reference_gain, gain_target),
                     (self.output_mix, output_target), (self.previous_output_mix, output_target))),
                'blend': self.blend, 'referenceGain': self.reference_gain,
                'blendTarget': blend_target, 'referenceGainTarget': gain_target,
                'outputMix': self.output_mix, 'outputMixPrevious': self.previous_output_mix,
                'outputMixTarget': output_target,
                'overlayIndependent': self.mode == 'influence',
                'controlVersion': 2 if self.mode == 'influence' else 1,
                'persistent': self.config_path is not None,
                'mode': self.mode, 'uiMax': .3 if self.mode == 'direct_latent_blend' else 1,
                'neutral': .5 if self.mode == 'influence' else .6 if self.mode == 'combined' else 1 if self.mode == 'reference_only' else 0,
                'cachePreserved': True}
