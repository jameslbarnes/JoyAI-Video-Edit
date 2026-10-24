import json
import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import numpy as np
from PIL import Image
import torch
from einops import rearrange
from xvideo.serving.spout_input import GpuInputFrame, load_input_config


class InputTests(unittest.TestCase):
    def test_vae_batch_keeps_frame_order_normalization_and_owned_pixels(self):
        # Exercise the actual session methods without loading a model/checkpoint.
        path = Path(__file__).resolve().parents[1] / "deploy/xvideo/serving/joyomni_streaming.py"
        module = ast.parse(path.read_text())
        cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "JoyOmniV2VStreamingSession")
        wanted = {"_resize_frame", "_frames_to_tensor"}
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        ns = {"__package__": "xvideo.serving", "Image": Image, "np": np, "torch": torch,
              "rearrange": rearrange, "_module_device": lambda _: torch.device("cpu")}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns)
        session = ns["JoyOmniV2VStreamingSession"]()
        session.settings = SimpleNamespace(width=4, height=4)
        session.pipeline = SimpleNamespace(vae=None)
        pixels = [torch.full((4, 4, 3), n, dtype=torch.uint8) for n in [0, 127, 255]]
        frames = [GpuInputFrame(p) for p in pixels]
        job = SimpleNamespace()
        with patch.object(GpuInputFrame, "to_pil", side_effect=AssertionError("Unexpected download")):
            batch = session._frames_to_tensor(frames, source_job=job)
        self.assertEqual(tuple(batch.shape), (1, 3, 3, 4, 4))
        for i, n in enumerate([0, 127, 255]):
            torch.testing.assert_close(batch[0, :, i], torch.full((3, 4, 4), n/127.5-1))
        self.assertTrue(torch.equal(job.source_pixels, torch.stack(pixels)))

    def test_config_requires_explicit_source_and_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "input.json"
            self.assertEqual(load_input_config(p), {})
            for cfg in [{"transport": "other"}, {"transport": "spout"},
                        {"transport": "spout", "source": "shader", "device": "cuda"}]:
                p.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    load_input_config(p)
            cfg = {"transport": "spout", "source": "shader", "device": "cuda:1"}
            p.write_text(json.dumps(cfg))
            self.assertEqual(load_input_config(p), cfg)

    def test_resize_and_model_tensor_access_do_not_download(self):
        pixels = torch.arange(48, dtype=torch.uint8).reshape(4, 4, 3)
        frame = GpuInputFrame(pixels)
        with patch.object(GpuInputFrame, "to_pil", side_effect=AssertionError("CPU download")):
            resized = frame.resize((8, 8))
            self.assertEqual(resized.size, (8, 8))
            self.assertEqual(resized.tensor().dtype, torch.uint8)
            self.assertIs(frame.convert("RGB"), frame)
            self.assertIs(frame.copy(), frame)
            self.assertTrue(torch.equal(frame.tensor(), pixels))

    def test_color_channel_and_pil_boundary_are_explicit(self):
        p = np.zeros((8, 16, 3), dtype=np.uint8)
        p[:4, :8] = [255, 32, 17]
        p[4:, 8:] = [8, 128, 245]
        metrics = {"pil_downloads": 0, "cross_gpu_bytes": 0}
        frame = GpuInputFrame(torch.from_numpy(p), metrics)
        np.testing.assert_array_equal(np.asarray(frame.to_pil()), p)
        self.assertEqual(metrics["pil_downloads"], 1)
        expected = np.asarray(Image.fromarray(p).resize((32, 16), Image.Resampling.BICUBIC)).astype(int)
        actual = frame.resize((32, 16)).tensor().numpy().astype(int)
        self.assertLessEqual(np.abs(actual - expected).max(), 1)


if __name__ == "__main__":
    unittest.main()
