"""CPU tests of gradual sink decay, including the actual session method, without model weights."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "deploy"))
from xvideo.serving.cache_decay import (CacheDecay, chunk_alpha, decay_sink, load_decay_config,
                                        own_sink, rope_compatible)

source = ROOT / "deploy/xvideo/serving/joyomni_streaming.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "JoyOmniV2VStreamingSession")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_decay_sink")
namespace = {}
module = ast.Module(body=[ast.ImportFrom(module="xvideo.serving.cache_decay", level=0,
                                         names=[ast.alias(name=n) for n in ("chunk_alpha", "decay_sink", "own_sink", "rope_compatible")]),
                          method], type_ignores=[])
# The method uses a relative import; rewrite it to the absolute package for exec.
method.body = [s for s in method.body if not isinstance(s, ast.ImportFrom)]
exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)


def entry(value, layers=2, tokens=6, pre_rope=True):
    return {li: {"key": torch.full((1, tokens, 2, 4), float(value)),
                 "value": torch.full((1, tokens, 2, 4), float(value)) * 2, "pre_rope": pre_rope}
            for li in range(layers)}


class Session:
    _decay_sink = namespace["_decay_sink"]

    def __init__(self, half_life_s, mode="blend", every_chunks=1):
        self.cache_decay = CacheDecay(mode, half_life_s, native_fps=12, every_chunks=every_chunks)
        self.frames_per_next_chunk = 8
        self._sink_decay_chunks = 0
        self._sink_decay_owned = None
        self.runner = SimpleNamespace(pool_dirty=False)
        self.runtime = SimpleNamespace(graph_runners={"g": self.runner})
        self._graph_key = lambda: "g"
        self.pipeline = SimpleNamespace(_kv_cache_memory_id=lambda kind, cid=None: (kind, cid))


class DecayMath(unittest.TestCase):
    def test_alpha_halves_old_share_every_half_life(self):
        alpha = chunk_alpha(60, 8 / 12)
        chunks = round(60 / (8 / 12))
        self.assertAlmostEqual((1 - alpha) ** chunks, 0.5, places=6)
        self.assertEqual(chunk_alpha(0, 0.66), 0.0)

    def test_validation_and_config(self):
        with self.assertRaises(ValueError):
            CacheDecay("smear", 10)
        with self.assertRaises(ValueError):
            CacheDecay("blend", -1)
        with self.assertRaises(ValueError):
            CacheDecay("blend", float("nan"))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache-decay.json"
            self.assertEqual(load_decay_config(path), {"mode": "blend", "half_life_s": 0.0})
            path.write_text(json.dumps({"mode": "dissolve", "half_life_s": 90}))
            self.assertEqual(load_decay_config(path), {"mode": "dissolve", "half_life_s": 90.0})

    def test_live_set_keeps_unspecified_fields(self):
        d = CacheDecay("dissolve", 30)
        state = d.set(half_life_s=120)
        self.assertEqual((state["mode"], state["half_life_s"], state["revision"]), ("dissolve", 120.0, 1))

    def test_blend_moves_toward_current(self):
        sink, cur = entry(0), entry(10)
        decay_sink(sink, cur, 0.25, "blend")
        self.assertTrue(torch.allclose(sink[1]["key"], torch.full_like(sink[1]["key"], 2.5)))
        self.assertTrue(torch.allclose(sink[0]["value"], torch.full_like(sink[0]["value"], 5.0)))
        self.assertTrue(torch.equal(cur[0]["key"], torch.full_like(cur[0]["key"], 10.0)))

    def test_dissolve_swaps_whole_tokens_consistently_across_layers(self):
        sink, cur = entry(0, tokens=4000), entry(1, tokens=4000)
        decay_sink(sink, cur, 0.1, "dissolve", generator=torch.Generator().manual_seed(3))
        swapped = sink[0]["key"][0, :, 0, 0] == 1
        self.assertTrue(0.07 < swapped.float().mean().item() < 0.13)
        for li in sink:
            for name in ("key", "value"):
                tok = sink[li][name][0, :, 0, 0] != 0
                self.assertTrue(torch.equal(tok, swapped))
                self.assertTrue(torch.all(sink[li][name].eq(0).all(-1).all(-1)[0] | tok))

    def test_own_sink_breaks_aliasing(self):
        sink = entry(0)
        owned = own_sink(sink)
        owned[0]["key"].add_(1)
        self.assertEqual(sink[0]["key"].max().item(), 0)
        self.assertTrue(owned[0]["pre_rope"])


class SessionMethod(unittest.TestCase):
    def test_disabled_is_a_noop(self):
        s = Session(0)
        store = {("clean", 0): entry(0), ("clean", 5): entry(1)}
        original = store[("clean", 0)]
        state = s._decay_sink(store, 5)
        self.assertIs(store[("clean", 0)], original)
        self.assertEqual((state["alpha"], state["chunks"], s.runner.pool_dirty), (0.0, 0, False))

    def test_enabled_owns_decays_and_dirties_graph_pool(self):
        s = Session(60)
        original, cur = entry(0), entry(10)
        store = {("clean", 0): original, ("clean", 5): cur}
        state = s._decay_sink(store, 5)
        sink = store[("clean", 0)]
        self.assertIsNot(sink, original)
        self.assertEqual(original[0]["key"].max().item(), 0)  # never mutates possibly shared storage
        self.assertAlmostEqual(sink[0]["key"][0, 0, 0, 0].item(), 10 * state["alpha"], places=5)
        self.assertTrue(s.runner.pool_dirty)
        store[("clean", 6)] = entry(10)
        s._decay_sink(store, 6)
        self.assertIs(store[("clean", 0)], sink)  # reuses owned storage after the first chunk
        self.assertEqual(s._sink_decay_chunks, 2)

    def test_old_share_halves_after_half_life(self):
        s = Session(20)
        store = {("clean", 0): entry(1)}
        chunks = round(20 / (8 / 12))
        for idx in range(1, chunks + 1):
            store[("clean", idx)] = entry(0)
            s._decay_sink(store, idx)
            del store[("clean", idx)]
        self.assertAlmostEqual(store[("clean", 0)][0]["key"][0, 0, 0, 0].item(), 0.5, places=4)

    def test_stride_keeps_half_life_and_skips_between_steps(self):
        s = Session(16, every_chunks=4)  # 24 chunks = 6 whole steps
        store = {("clean", 0): entry(1)}
        chunks = round(16 / (8 / 12))
        for idx in range(1, chunks + 1):
            store[("clean", idx)] = entry(0)
            s._decay_sink(store, idx)
            del store[("clean", idx)]
        self.assertEqual(s._sink_decay_chunks, chunks // 4)
        self.assertAlmostEqual(store[("clean", 0)][0]["key"][0, 0, 0, 0].item(), 0.5, places=4)
        with self.assertRaises(ValueError):
            CacheDecay("blend", 10, every_chunks=0)
        self.assertEqual(CacheDecay().set(every_chunks=2)["every_chunks"], 2)

    def test_skips_mismatched_rope_and_missing_entries(self):
        s = Session(60)
        store = {("clean", 0): entry(0, pre_rope=False), ("clean", 5): entry(1)}
        self.assertEqual(s._decay_sink(store, 5)["skipped"], "rope")
        self.assertEqual(s._decay_sink({("clean", 0): entry(0)}, 5)["skipped"], "missing")
        self.assertFalse(s.runner.pool_dirty)
        self.assertFalse(rope_compatible(entry(0, pre_rope=False), entry(0)))


if __name__ == "__main__":
    unittest.main()
