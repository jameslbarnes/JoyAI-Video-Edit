"""CPU tests of history noise (deploy/xvideo/serving/history_noise.py) and its wiring."""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
from xvideo.serving.history_noise import (HistoryNoise, load_history_noise_config,  # noqa: E402
                                          noisy_history_latents, validate_sigma)


class HistoryNoiseTests(unittest.TestCase):
    def test_zero_sigma_is_exactly_the_clean_latent(self):
        clean = torch.randn(1, 4, 1, 6, 8)
        gen = torch.Generator().manual_seed(1)
        state = gen.get_state()
        out, t = noisy_history_latents(clean, 0.0, gen)
        self.assertIs(out, clean)
        self.assertEqual(t, 0.0)
        self.assertTrue(torch.equal(gen.get_state(), state))  # generator untouched

    def test_flow_matching_mix_and_timestep(self):
        clean = torch.ones(1, 2, 1, 32, 32)
        out, t = noisy_history_latents(clean, 0.2, torch.Generator().manual_seed(3))
        self.assertEqual(t, 200.0)
        noise = (out - 0.8 * clean) / 0.2
        self.assertAlmostEqual(noise.mean().item(), 0.0, delta=0.1)
        self.assertAlmostEqual(noise.std().item(), 1.0, delta=0.1)
        self.assertEqual(out.dtype, clean.dtype)

    def test_validation_config_and_live_set(self):
        for bad in (-0.1, 0.6, True, "x"):
            with self.assertRaises((ValueError, TypeError)):
                validate_sigma(bad)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "history-noise.json"
            self.assertEqual(load_history_noise_config(path), {"sigma": 0.0})
            path.write_text(json.dumps({"sigma": 0.2}))
            self.assertEqual(load_history_noise_config(path), {"sigma": 0.2})
        h = HistoryNoise()
        self.assertEqual(h.set(0.15), {"sigma": 0.15, "enabled": True, "revision": 1})
        self.assertEqual(h.sigma, 0.15)


class PipelineStoreTimestepTests(unittest.TestCase):
    def test_store_passes_the_requested_timestep(self):
        from xvideo.models.pipeline import Pipeline  # noqa: F401  (import check)
        src = (ROOT / "deploy/xvideo/models/pipeline.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_store_clean_chunk_kv_cache")
        ns = {"torch": torch, "Optional": __import__("typing").Optional, "List": list}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "pipeline", "exec"), ns)
        seen = {}

        class Model:
            def cache_context(self, scope):
                import contextlib
                return contextlib.nullcontext()

            def __call__(self, **kw):
                seen.update(kw)

        owner = SimpleNamespace(_kv_cache_memory_id=lambda kind, cid=None: (kind, cid))
        latents = torch.zeros(1, 2, 1, 4, 4, dtype=torch.bfloat16)
        ns["_store_clean_chunk_kv_cache"](owner, Model(), clean_chunk_latents=latents,
                                          chunk_temporal_ids=torch.zeros(1, 1), prompt_embeds=None,
                                          active_chunk_id=3, history_chunk_ids=[], timestep_value=200.0)
        self.assertEqual(seen["timestep"].tolist(), [200.0])
        ns["_store_clean_chunk_kv_cache"](owner, Model(), clean_chunk_latents=latents,
                                          chunk_temporal_ids=torch.zeros(1, 1), prompt_embeds=None,
                                          active_chunk_id=3, history_chunk_ids=[])
        self.assertEqual(seen["timestep"].tolist(), [0.0])  # default unchanged


class GraphStoreFormulaTests(unittest.TestCase):
    """The captured graph computes lat*(1-s) + noise*s from runtime buffers."""

    def test_zero_sigma_buffer_is_bit_identical(self):
        lat32 = torch.randn(1, 4, 1, 6, 8, dtype=torch.float32)
        noise = torch.randn_like(lat32).to(torch.bfloat16)
        sigma = torch.zeros((), dtype=torch.float32)
        mixed = (lat32 * (1 - sigma) + noise.to(torch.float32) * sigma).to(torch.bfloat16)
        self.assertTrue(torch.equal(mixed, lat32.to(torch.bfloat16)))

    def test_graph_source_uses_the_buffers(self):
        src = (ROOT / "deploy/xvideo/serving/graph_runner.py").read_text(encoding="utf-8")
        self.assertIn("self.in_store_noise.to(torch.float32) * self.in_store_sigma", src)
        session = (ROOT / "deploy/xvideo/serving/joyomni_streaming.py").read_text(encoding="utf-8")
        self.assertIn("runner.in_store_timestep.fill_(sigma * 1000.0)", session)

    def test_scene_start_sink_is_always_stored_clean(self):
        session = (ROOT / "deploy/xvideo/serving/joyomni_streaming.py").read_text(encoding="utf-8")
        self.assertIn("sigma = 0.0 if active_chunk_id == 0 else self.history_noise.sigma", session)


if __name__ == "__main__":
    unittest.main()
