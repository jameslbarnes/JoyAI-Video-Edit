"""Real GPU SR+RIFE and CPU fail-open/bounds tests; no model session needed."""
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy'))
import numpy as np
import torch
from PIL import Image
from xvideo.serving.upscale_output import OutputUpscale
from xvideo.serving.rife_output import RifeOutput


class UpscalePipelineTest(unittest.TestCase):
    def test_bypass_bounds_and_failed_model_latches_off(self):
        sr = OutputUpscale('cuda:0')
        self.addCleanup(sr.close)
        sr.model = Mock()
        sr.model.process.side_effect = RuntimeError('simulated allocation failure')
        frame = np.zeros((360, 648, 3), np.uint8)
        profile = {}
        out = sr.process([frame], profile)
        self.assertEqual(Image.open(io.BytesIO(out[0])).size, (648, 360))
        self.assertFalse(sr.enabled)
        sr.process([frame], {})
        sr.model.process.assert_called_once()
        with self.assertRaises(ValueError): sr.set_enabled(True)
        with self.assertRaises(ValueError): sr.set_enabled('false')
        with self.assertRaises(ValueError): sr.process([frame] * 17, {})
        with self.assertRaises(ValueError): sr.process([frame[:20]], {})

    def test_real_raw_rife_sr_and_reset(self):
        sr = OutputUpscale('cuda:0')
        self.addCleanup(sr.close)
        sr.warmup()
        self.assertTrue(sr.enabled, sr.error)
        rife = RifeOutput('cuda:0')
        frames = []
        for i in range(8):
            frame = np.zeros((360, 648, 3), np.uint8)
            frame[90:220, 100+i*5:240+i*5] = [230, 80, 35]
            frames.append(frame)
        metas = [{'seq': i+1} for i in range(8)]
        profiles = []
        for _ in range(3):
            rife.reset()
            profile = {}
            raw, mapped = rife.process(frames, metas, profile, raw_output=True)
            self.assertEqual(len(raw), 15)
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(raw[::2], frames)))
            self.assertEqual(mapped[1]['source_seq_previous'], 1)
            out = sr.process(raw, profile)
            self.assertTrue(all(Image.open(io.BytesIO(j)).size == (1296, 720) for j in out))
            self.assertNotEqual(out[1], out[0])
            profiles.append(profile)
        sr.set_enabled(False)
        out = sr.process(raw, {})
        self.assertTrue(all(Image.open(io.BytesIO(j)).size == (648, 360) for j in out))
        sr.set_enabled(True)
        rife.reset()
        raw, _ = rife.process(frames[:1], metas[:1], {}, raw_output=True)
        self.assertEqual(len(raw), 1, 'No interpolation across reset')
        print(json.dumps({'upscale': sr.status(), 'profiles': profiles}))


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    unittest.main()
