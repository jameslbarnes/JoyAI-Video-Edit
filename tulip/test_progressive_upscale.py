import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
from xvideo.serving.upscale_output import OutputUpscale


class ProgressiveUpscale(unittest.TestCase):
    def test_rollback_keeps_serial_preview_contract_for_new_clients(self):
        output, published = self.make_output()
        frames = [np.full((360, 648, 3), i, np.uint8) for i in [1, 2, 3]]
        with ThreadPoolExecutor(max_workers=2) as output.encoder:
            previews = output._process_progressive(frames, {"presentation_fps": 32},
                95, time.perf_counter(), defer_preview=True)
        self.assertEqual(published, [1, 2, 3])
        self.assertEqual(previews, [bytes([i, 95]) for i in [1, 2, 3]])
        self.assertFalse(output.last["preview_deferred"])
        self.assertEqual(output.last["preview_execution"], "serial_gpu_cpu_jpeg_workers")

    def make_output(self, fail=False):
        output = OutputUpscale.__new__(OutputUpscale)
        output.enabled, output.error = True, None
        output.device, output.output_size = 'cuda:1', (720, 1296)
        output.direct_requested = True
        output.neural_rendering = SimpleNamespace(status=lambda: {'enabled': False})
        published = []
        encoding = threading.Event()
        def upscale(frame, size):
            value = int(frame[0,0,0])
            if value == 2:
                self.assertEqual(published, [1], 'first frame must publish before second SR')
                self.assertTrue(encoding.wait(2), 'JPEG work must overlap SR')
                if fail:
                    raise ValueError('simulated SR failure')
            return np.full((*size, 3), value, np.uint8)
        def publish(frames, fps):
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].shape, (720,1296,3))
            self.assertEqual(fps, 32)
            published.append(int(frames[0][0,0,0]))
        def encode(frame, quality):
            encoding.set()
            return bytes([int(frame[0,0,0]), quality])
        output.model = SimpleNamespace(process=upscale)
        output.direct_publisher = SimpleNamespace(error=None,submit=publish, available=lambda: True, status=lambda: {'error': None})
        output.encode_preview = encode
        return output, published

    def test_publish_order_overlap_and_exact_preview_results(self):
        output, published = self.make_output()
        frames = [np.full((360,648,3), i, np.uint8) for i in [1,2,3]]
        profile = {'presentation_fps':32}
        with ThreadPoolExecutor(max_workers=2) as output.encoder:
            result = output._process_progressive(frames, profile, 95, time.perf_counter())
        self.assertEqual(result, [bytes([i,95]) for i in [1,2,3]])
        self.assertEqual(published, [1,2,3])
        self.assertTrue(profile['upscale']['progressive'])
        self.assertEqual(profile['upscale']['height'],720)

    def test_partial_failure_preserves_geometry_order_and_reports_error(self):
        output, published = self.make_output(fail=True)
        frames = [np.full((360,648,3), i, np.uint8) for i in [1,2,3]]
        profile = {}
        with ThreadPoolExecutor(max_workers=2) as output.encoder:
            result = output._process_progressive(frames, profile, 95, time.perf_counter())
        self.assertEqual(published,[1,2,3])
        self.assertEqual(result,[bytes([i,95]) for i in [1,2,3]])
        self.assertFalse(output.enabled)
        self.assertEqual(profile['upscale']['error'],'simulated SR failure')

if __name__ == '__main__':
    unittest.main()
