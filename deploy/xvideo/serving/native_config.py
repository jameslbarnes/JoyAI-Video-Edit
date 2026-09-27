"""Bounded, machine-local native quality preset, read at service startup."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NATIVE_SIZES = {(648, 360), (768, 432), (840, 480), (864, 480), (1056, 600), (1152, 648), (1296, 720)}
DEFAULT = {'width': 648, 'height': 360, 'fps': 16, 'rife_multiplier': 2}

def load_native_config(path=None):
    path = Path(path) if path is not None else ROOT / 'tulip/native-quality.json'
    if not path.exists():
        return dict(DEFAULT)
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict) or set(config) - set(DEFAULT):
        raise ValueError('Native quality accepts width, height, fps and rife_multiplier')
    value = {**DEFAULT, **config}
    if any(type(v) is not int for v in value.values()):
        raise ValueError('Native quality settings must be integers')
    if (value['width'], value['height']) not in NATIVE_SIZES:
        raise ValueError('Unsupported native resolution')
    if value['fps'] not in (6, 8, 10, 12, 16, 20, 24):
        raise ValueError('Unsupported native FPS')
    if value['rife_multiplier'] not in (1, 2, 4) or value['fps'] * value['rife_multiplier'] > 64:
        raise ValueError('Presentation FPS must not exceed 64')
    return value
