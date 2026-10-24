"""CPU-only geometry/config tests; no live GPU benchmarks."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

source = Path(__file__).resolve().parents[1] / 'deploy/xvideo/serving/upscale_output.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OutputUpscale')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_refresh_output_size')
namespace = {'json': json, '__package__': 'xvideo.serving'}
geometry = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gpu_output_size')
exec(compile(ast.Module(body=[geometry, method], type_ignores=[]), str(source), 'exec'), namespace)
class Config:
    refresh = namespace['_refresh_output_size']
    def __init__(self):
        self.device = 'cuda:0'
        self.direct_publisher = SimpleNamespace(transport='ndi', gpu_input=False, close=Mock())

class OutputGeometry(unittest.TestCase):
    def test_copy_priority_changes_without_replacing_sender_or_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            c = Config()
            c.model, c.backend, c.models = None, 'vsr_ultra', {}
            c.output_size = (1440, 2592)
            sender = SimpleNamespace(transport='spout', gpu_input=True,
                configure_copy_priority=Mock(), configure_transition_reserve=Mock(),
                configure_preparation_profile=Mock(), close=Mock())
            c.direct_publisher = sender
            c.output_config = Path(folder)/'output.json'
            cfg = dict(height=1440, backend='vsr_ultra', progressive=True,
                       direct_ndi=True, local_transport='spout')
            for priority in (-1, 0):
                cfg['spout_copy_priority'] = priority
                c.output_config.write_text(json.dumps(cfg))
                c.refresh()
                self.assertIsNone(c.config_error)
                sender.configure_copy_priority.assert_called_with(priority)
                self.assertIs(c.direct_publisher, sender)
                self.assertEqual(c.output_size, (1440, 2592))
            count = sender.configure_copy_priority.call_count
            for invalid in (True, '-1', -1., -2, 1, None):
                cfg['spout_copy_priority'] = invalid
                c.output_config.write_text(json.dumps(cfg))
                c.refresh()
                self.assertIsNotNone(c.config_error)
                self.assertEqual(sender.configure_copy_priority.call_count, count)
            sender.close.assert_not_called()

    def test_spout_selection_and_missing_library_preserve_previous_publisher(self):
        with tempfile.TemporaryDirectory() as folder:
            c = Config()
            c.model, c.backend, c.models = None, 'realesrgan', {}
            c.output_size = (1080, 1944)
            c.output_config = Path(folder)/'output.json'
            c.output_config.write_text(json.dumps({'height':1440,'backend':'vsr_ultra',
                'progressive':True,'direct_ndi':True,'local_transport':'spout'}))
            original = c.direct_publisher
            factory = Mock(side_effect=OSError('missing native library'))
            with patch.dict('sys.modules', {'xvideo.serving.spout_output': SimpleNamespace(GpuSpoutPublisher=factory)}):
                c.refresh()
                self.assertIs(c.direct_publisher, original)
                original.close.assert_not_called()
                self.assertEqual(c.output_size, (1080,1944))
                self.assertIn('missing native library', c.config_error)
                factory.side_effect = None
                replacement = SimpleNamespace(transport='spout',gpu_input=True,configure_copy_priority=Mock(),
                    configure_transition_reserve=Mock(), configure_preparation_profile=Mock())
                factory.return_value = replacement
                c.refresh()
                self.assertIs(c.direct_publisher, replacement)
                original.close.assert_called_once()
                self.assertEqual(c.output_size,(1440,2592))
                self.assertIsNone(c.config_error)

    def test_uhd_gpu_output_keeps_ndi_geometry_and_rejects_partial_edits(self):
        self.assertEqual(namespace['gpu_output_size'](2160), (2160, 3840))
        self.assertEqual(namespace['gpu_output_size'](1440), (1440, 2592))
        self.assertEqual(namespace['gpu_output_size'](1080), (1080, 1944))
        with tempfile.TemporaryDirectory() as folder:
            c = Config()
            c.model, c.backend, c.models = None, 'realesrgan', {}
            c.output_config = Path(folder) / 'output.json'
            cfg = {'height': 1080, 'direct_ndi': True, 'progressive': True,
                   'gpu_video': {'enabled': True, 'height': 2160, 'bitrate': 15000000,
                                 'url': 'rtmp://127.0.0.1:1935/test', 'ffmpeg': 'ffmpeg.exe'}}
            c.output_config.write_text(json.dumps(cfg))
            c.refresh()
            self.assertIsNone(c.config_error)
            self.assertEqual(c.output_size, (1080, 1944))
            self.assertEqual(c.gpu_config['height'], 2160)
            for invalid in (True, 2160.0, '2160', 4320, None):
                cfg['gpu_video']['height'] = invalid
                c.output_config.write_text(json.dumps(cfg))
                c.refresh()
                self.assertIsNotNone(c.config_error)
                self.assertEqual(c.gpu_config['height'], 2160)
                self.assertEqual(c.output_size, (1080, 1944))

    def test_live_change_invalid_edit_and_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            c = Config()
            c.model, c.backend, c.models = None, 'realesrgan', {}
            c.output_config = Path(folder) / 'output.json'
            c.refresh()
            self.assertEqual(c.output_size, (720, 1296))
            c.output_config.write_text('{"height":1080}')
            c.refresh()
            self.assertEqual(c.output_size, (1080, 1944))
            for invalid in ('{', '[]', '{"height":true}', '{"height":1080.0}', '{"height":2160}', '{"height":720,"direct_ndi":"yes"}'):
                c.output_config.write_text(invalid)
                c.refresh()
                self.assertEqual(c.output_size, (1080, 1944))
                self.assertIsNotNone(c.config_error)
            c.output_config.write_text('{"height":1440,"direct_ndi":true,"progressive":true}')
            c.refresh()
            self.assertEqual(c.output_size, (1440, 2592))
            self.assertTrue(c.direct_requested)
            self.assertTrue(c.progressive)
            self.assertIsNone(c.config_error)
            c.output_config.write_text('{"height":720}')
            c.refresh()
            self.assertEqual(c.output_size, (720, 1296))
            self.assertIsNone(c.config_error)
            c.output_config.unlink()
            c.refresh()
            self.assertEqual(c.output_size, (720, 1296))

    def test_switch_only_to_prewarmed_backend_and_keep_current_on_error(self):
        with tempfile.TemporaryDirectory() as folder:
            c = Config()
            favorite, vsr = object(), object()
            c.model, c.backend, c.models = favorite, 'realesrgan', {'realesrgan': favorite}
            c.denoise = .25
            c.output_config = Path(folder) / 'output.json'
            c.output_config.write_text('{"backend":"vsr_ultra","denoise":0.25}')
            c.refresh()
            self.assertIs(c.model, favorite)
            self.assertIsNotNone(c.config_error)
            c.models['vsr_ultra'] = vsr
            c.refresh()
            self.assertIs(c.model, vsr)
            self.assertEqual(c.backend, 'vsr_ultra')
            c.output_config.write_text('{"backend":"realesrgan","denoise":0.25}')
            c.refresh()
            self.assertIs(c.model, favorite)
            self.assertIsNone(c.config_error)

if __name__ == '__main__':
    unittest.main()
