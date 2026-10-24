"""CPU-only release/parameter isolation tests; no CUDA effect is constructed."""
from enum import IntEnum
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from xvideo.serving.vfx_runtime import VsrSettings, load_bindings
from tulip.test_vsr_release_live import require_idle


class Quality(IntEnum):
    ULTRA = 4
    HIGHBITRATE_ULTRA = 19
    STREAMING_MEDIUM = 21
    STREAMING_ULTRA = 23


class VfxRuntimeTests(unittest.TestCase):
    def test_startup_file_is_optional_validated_and_read_once(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(VsrSettings.from_startup(root, {}), VsrSettings())
            path = Path(root)/'tulip/vfx-runtime.json'
            path.parent.mkdir()
            path.write_text(json.dumps({'release': '0.2.0.0', 'quality': 4, 'strength': 1.0}))
            selected = VsrSettings.from_startup(root, {})
            self.assertEqual(selected, VsrSettings('0.2.0.0', 4, 1.0))
            self.assertEqual(VsrSettings.from_startup(root, {'JOYAI_VSR_QUALITY': '23'}).quality, 23)
            path.write_text(json.dumps({'release': '0.2.0.0', 'quality': 19}))
            self.assertEqual(selected.quality, 4, 'Loaded settings must not hot-reload')
            path.unlink()
            self.assertEqual(VsrSettings.from_startup(root, {}), VsrSettings())

    def test_invalid_startup_file_fails_without_silent_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'tulip/vfx-runtime.json'
            path.parent.mkdir()
            for content in ('{', '[]', '{"typo":4}', '{"quality":true}',
                            '{"quality":4.0}', '{"release":"0.1.0.1","quality":23}'):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises((ValueError, TypeError)):
                    VsrSettings.from_startup(root, {})

    def test_gpu_fixture_refuses_active_or_unknown_show_state(self):
        for state in ({'ok': True, 'session': {}}, {'ok': True}, {}, None,
                      {'ok': False, 'session': None}):
            with self.subTest(state=state), self.assertRaises(RuntimeError):
                require_idle(state)
        require_idle({'ok': True, 'session': None})

    def test_default_preserves_old_release_and_constructor(self):
        settings = VsrSettings.from_environment({})
        self.assertEqual(settings, VsrSettings('0.1.0.1', 4))
        self.assertEqual(settings.effect_arguments(1, Quality),
                         {'quality': Quality.ULTRA, 'device': 1})

    def test_new_release_modes_and_explicit_strength(self):
        for quality in (4, 19, 21, 23):
            settings = VsrSettings.from_environment({'JOYAI_VFX_RELEASE': '0.2.0.0',
                'JOYAI_VSR_QUALITY': str(quality), 'JOYAI_VSR_STRENGTH': '.75'})
            self.assertEqual(settings.effect_arguments(1, Quality),
                             {'quality': Quality(quality), 'device': 1, 'strength': .75})
        self.assertEqual(VsrSettings('0.2.0.0').effect_arguments(0, Quality)['strength'], 1.0)

    def test_rejects_unsupported_modes_before_loading_native_code(self):
        for quality in (5, 6, 7, 8, 11, 12, 15, 20, 22, 24, True, 4.0):
            with self.subTest(quality=quality), self.assertRaises(ValueError):
                VsrSettings('0.2.0.0', quality)
        for quality in (21, 23):
            with self.assertRaisesRegex(ValueError, 'requires'):
                VsrSettings('0.1.0.1', quality)
        with self.assertRaises(ValueError):
            VsrSettings('latest')

    def test_rejects_nonfinite_strength_and_legacy_override(self):
        for strength in (-.01, 1.01, float('nan'), float('inf'), True, '0.5'):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                VsrSettings('0.2.0.0', 4, strength)
        with self.assertRaises(ValueError):
            VsrSettings('0.1.0.1', 4, .5)
        for strength in ('nan', 'inf', '', 'bad'):
            with self.assertRaises(ValueError):
                VsrSettings.from_environment({'JOYAI_VFX_RELEASE': '0.2.0.0',
                                               'JOYAI_VSR_STRENGTH': strength})

    def make_package(self, root, settings):
        package = settings.package_path(root) / 'nvvfx'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('')
        return SimpleNamespace(__file__=str(package/'__init__.py'),
                               __version__=settings.release,
                               get_sdk_version=lambda: '1.3.0')

    def test_missing_package_does_not_fall_back_to_installed_version(self):
        with tempfile.TemporaryDirectory() as root, patch('importlib.import_module') as importer:
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                load_bindings(VsrSettings('0.2.0.0'), root)
            importer.assert_not_called()

    def test_wrong_loaded_package_or_submodule_is_rejected_before_import(self):
        settings = VsrSettings('0.2.0.0')
        with tempfile.TemporaryDirectory() as root:
            self.make_package(root, settings)
            old = SimpleNamespace(__file__=str(Path(root)/'vfx-probe/nvvfx/__init__.py'))
            for name in ('nvvfx', 'nvvfx._ext'):
                with patch.dict('sys.modules', {name: old}), patch('importlib.import_module') as importer:
                    with self.assertRaisesRegex(RuntimeError, 'fresh process'):
                        load_bindings(settings, root)
                    importer.assert_not_called()

    def test_import_validates_package_and_loaded_native_sdk(self):
        settings = VsrSettings('0.2.0.0', 23)
        with tempfile.TemporaryDirectory() as root:
            package = self.make_package(root, settings)
            for version, sdk, expected in [('0.1.0.1', '1.3.0', 'package mismatch'),
                                          ('0.2.0.0', '1.2.0', 'SDK mismatch'),
                                          ('0.2.0.0', '1.3.0', None)]:
                package.__version__ = version
                package.get_sdk_version = lambda: sdk
                with patch('importlib.import_module', side_effect=[package, SimpleNamespace(QualityLevel=Quality)]):
                    if expected:
                        with self.assertRaisesRegex(RuntimeError, expected):
                            load_bindings(settings, root)
                    else:
                        self.assertEqual(load_bindings(settings, root), (package, Quality))


if __name__ == '__main__':
    unittest.main()
