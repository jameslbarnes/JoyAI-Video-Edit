"""CPU checks for reversible, bounded transition-buffer trials."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from xvideo.serving.spout_output import GpuSpoutPublisher
from xvideo.serving.upscale_output import OutputUpscale


class SpoutReserveTests(unittest.TestCase):
    def publisher(self):
        with patch('xvideo.serving.spout_output.library', return_value=Mock()):
            return GpuSpoutPublisher('cpu')

    def test_live_change_keeps_scene_frames_clock_and_native_identity(self):
        p = self.publisher()
        frame, scene, worker = object(), {'start': 123.}, object()
        p.frames.append(frame)
        p.transition, p.thread = scene, worker
        p.geometry, p.sequence = (2592, 1440, 24), 100
        p.retained[99] = frame
        for reserve in (3, 4, 2):
            p.configure_transition_reserve(reserve)
            self.assertEqual(p._native_reserve(), reserve)
            self.assertIs(p.transition, scene)
            self.assertIs(p.thread, worker)
            self.assertEqual(list(p.frames), [frame])
            self.assertEqual(p.retained, {99: frame})
            self.assertEqual(p.sequence, 100)
            self.assertEqual(p.geometry, (2592, 1440, 24))
        p.transition = None
        self.assertEqual(p._native_reserve(), 16)

    def test_invalid_values_cannot_replace_working_reserve(self):
        p = self.publisher()
        p.configure_transition_reserve(3)
        for value in (None, True, 1, 7, 3.0, '3'):
            with self.assertRaises(ValueError):
                p.configure_transition_reserve(value)
            self.assertEqual(p.transition_reserve, 3)

    def test_profile_toggle_does_not_allocate_gpu_resources(self):
        p = self.publisher()
        with patch('torch.cuda.Event', side_effect=AssertionError('GPU allocation on control thread')):
            p.configure_preparation_profile(True)
            self.assertTrue(p.profile_preparation)
            p.configure_preparation_profile(False)
            self.assertFalse(p.profile_preparation)
            self.assertIsNone(p.preparation_events)
            for value in (None, 0, 1, 'true'):
                with self.assertRaises(ValueError):
                    p.configure_preparation_profile(value)

    def test_chunk_reload_and_missing_override_restore_default_without_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            o = OutputUpscale.__new__(OutputUpscale)
            o.output_config = Path(directory) / 'presentation.json'
            o.model, o.backend, o.denoise = None, 'vsr_ultra', .25
            o.models = {}
            p = o.direct_publisher = self.publisher()
            cfg = dict(height=1440, local_transport='spout', direct_ndi=True,
                       progressive=True, backend='vsr_ultra', denoise=.25,
                       rife_gpu=True, spout_transition_reserve=3, spout_profile_preparation=True)
            for expected in (3, 2):
                if expected == 2:
                    cfg.pop('spout_transition_reserve')
                o.output_config.write_text(json.dumps(cfg))
                o._refresh_output_size()
                self.assertIsNone(o.config_error)
                self.assertIs(o.direct_publisher, p)
                self.assertEqual(p.transition_reserve, expected)
                self.assertTrue(p.profile_preparation)
                self.assertEqual(p._native_reserve(), 16)
                self.assertEqual(o.output_size, (1440, 2592))
                self.assertEqual((o.backend, o.denoise, o.rife_gpu), ('vsr_ultra', .25, True))
            cfg.update(height=720, spout_transition_reserve=True)
            o.output_config.write_text(json.dumps(cfg))
            o._refresh_output_size()
            self.assertIn('transition reserve', o.config_error)
            self.assertEqual(o.output_size, (1440, 2592))
            self.assertEqual(p.transition_reserve, 2)


if __name__ == '__main__':
    unittest.main()
