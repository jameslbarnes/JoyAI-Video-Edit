"""Explicit, process-start selection of a side-by-side NVIDIA VFX release.

Import checks do not create an effect or start CUDA inference. Never reload
native VFX libraries in a process that already imported another release.
"""
from dataclasses import dataclass
import importlib
import json
import math
import os
from pathlib import Path
import sys


RELEASES = {
    '0.1.0.1': ('1.2.0', 'vfx-probe'),
    '0.2.0.0': ('1.3.0', 'vfx-0.2.0.0'),
}
UPSCALE_MODES = frozenset((0, 1, 2, 3, 4, 16, 17, 18, 19, 21, 23))


@dataclass(frozen=True)
class VsrSettings:
    release: str = '0.1.0.1'
    quality: int = 4
    strength: float | None = None

    def __post_init__(self):
        if self.release not in RELEASES:
            raise ValueError(f'Unsupported JOYAI_VFX_RELEASE: {self.release}')
        if type(self.quality) is not int or self.quality not in UPSCALE_MODES:
            raise ValueError('VSR quality must be an upscaling mode: 0..4, 16..19, 21 or 23')
        if self.release == '0.1.0.1' and self.quality in (21, 23):
            raise ValueError('Streaming VSR requires JOYAI_VFX_RELEASE=0.2.0.0')
        if self.strength is not None:
            if (isinstance(self.strength, bool) or
                    not isinstance(self.strength, (int, float)) or
                    not math.isfinite(self.strength) or not 0 <= self.strength <= 1):
                raise ValueError('JOYAI_VSR_STRENGTH must be finite and in [0, 1]')
            if self.release == '0.1.0.1':
                raise ValueError('Explicit VSR strength requires release 0.2.0.0')

    @classmethod
    def from_environment(cls, env=None):
        env = os.environ if env is None else env
        release = env.get('JOYAI_VFX_RELEASE', '0.1.0.1')
        quality = int(env.get('JOYAI_VSR_QUALITY', '4'))
        strength = env.get('JOYAI_VSR_STRENGTH')
        return cls(release, quality, None if strength is None else float(strength))

    @classmethod
    def from_startup(cls, repo=None, env=None):
        repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[3]
        path = repo / 'tulip' / 'vfx-runtime.json'
        values = {}
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(data, dict) or set(data) - {'release', 'quality', 'strength'}:
                raise ValueError('vfx-runtime.json accepts only release, quality and strength')
            cls(**data)  # Validate JSON types before converting to environment strings.
            values = {f'JOYAI_VFX_RELEASE' if key == 'release' else f'JOYAI_VSR_{key.upper()}': str(value)
                      for key, value in data.items() if value is not None}
        env = os.environ if env is None else env
        for name in ('JOYAI_VFX_RELEASE', 'JOYAI_VSR_QUALITY', 'JOYAI_VSR_STRENGTH'):
            if name in env:
                values[name] = env[name]
        return cls.from_environment(values)

    def package_path(self, repo=None):
        repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[3]
        return repo / 'tulip' / 'research' / RELEASES[self.release][1]

    def effect_arguments(self, device, quality_enum):
        args = {'quality': quality_enum(self.quality), 'device': device}
        # Preserve the old constructor exactly. The new binding exposes Strength;
        # make its documented default explicit for reproducible comparisons.
        if self.release == '0.2.0.0':
            args['strength'] = 1.0 if self.strength is None else self.strength
        return args


def load_bindings(settings, repo=None):
    package = settings.package_path(repo).resolve()
    module_root = package / 'nvvfx'
    if not (module_root / '__init__.py').is_file():
        raise RuntimeError(f'Pinned nvidia-vfx {settings.release} is missing at {package}')
    # Removing modules from sys.modules cannot unload their native DLLs safely.
    # Reject contamination before importing a second copy, including submodules.
    for name, module in tuple(sys.modules.items()):
        if name == 'nvvfx' or name.startswith('nvvfx.'):
            origin = getattr(module, '__file__', None)
            if not origin or not Path(origin).resolve().is_relative_to(module_root):
                raise RuntimeError('Another VFX installation is already imported; use a fresh process')
    sys.path.insert(0, str(package))
    try:
        bindings = importlib.import_module('nvvfx')
        quality_enum = importlib.import_module('nvvfx.effects.video_super_res').QualityLevel
    finally:
        sys.path.remove(str(package))
    if bindings.__version__ != settings.release:
        raise RuntimeError(f'VFX package mismatch: expected {settings.release}, got {bindings.__version__}')
    sdk = bindings.get_sdk_version()
    expected = RELEASES[settings.release][0]
    if sdk != expected:
        raise RuntimeError(f'VFX SDK mismatch: expected {expected}, got {sdk}; check native DLL loading')
    # Validate the named mode against the actual installed Python API too.
    quality_enum(settings.quality)
    return bindings, quality_enum
