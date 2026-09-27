"""CPU tests of actual session-control methods, without loading model weights."""
import ast
import json
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

source = Path(__file__).resolve().parents[1] / "deploy/xvideo/serving/joyomni_streaming.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "JoyOmniV2VStreamingSession")
names = {"queue_prompt", "_prepare_prompt_worker", "_apply_prepared_prompt", "_refresh_sink_and_measure", "_encode_streaming_prompt", "_prepare_prompt_graph"}
from PIL import Image
namespace = {"threading": threading, "time": time, "torch": torch, "Image": Image, "graph_env_enabled": lambda: True, "fixed_prompt_graph_enabled": lambda: False}
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
Control = type("Control", (), {**{n: namespace[n] for n in names},
                                 "_decay_sink": lambda self, store, chunk_idx: {"enabled": False}})  # covered in test_cache_decay


def control():
    c = Control()
    c._prompt_lock = threading.Lock()
    c._prompt_thread = c._pending_prompt = c._prepared_prompt = None
    c._prompt_request = c._prompt_revision = 0
    c._closing = False
    c.initialized = True
    c._latest_source_frame = {"anchor": 1}
    c._encode_prompt_value = lambda text, anchor: torch.ones(1, len(text), 2)
    c.prompt = c.raw_prompt = "old"
    c.device = "cpu"
    c.target_dtype = torch.float32
    c._prompt_status = "idle"
    c._prompt_error = None
    c._prompt_encode_s = 0
    c._session_serial = 77
    c._graph_key = lambda: "graph"
    c.runtime = SimpleNamespace(graph_runners={})
    c._maybe_prepare_graph_runner = lambda: None
    c._prepare_prompt_graph = lambda embeds, revision: None
    c._sink_refresh_chunks = 32
    c._sink_chunk = c._sink_refresh_count = 0
    c.settings = SimpleNamespace(max_temporal_ids=128)
    return c


class PromptGraphRefresh(unittest.TestCase):
    def make_control(self):
        c = control()
        c._submit_lock = threading.RLock()
        c._debug_lock = threading.Lock()
        c._debug_counters = dict(submitted_chunks=3, postprocessed_chunks=3, pseudo_encoded_chunks=3)
        c._raise_worker_error_if_needed = lambda: None
        c.runtime.dit_lock = threading.RLock()
        c.runtime.presentation_lock = threading.RLock()
        c.runtime.interpolation_lock = threading.RLock()
        c.streaming_cond_embeds = torch.zeros(1, 2, 2)
        c._graph_key = lambda embeds=None: (embeds if embeds is not None else c.streaming_cond_embeds).shape[1]
        c._prompt_request = 1
        c._prepare_prompt_graph = Control._prepare_prompt_graph.__get__(c)
        return c

    def test_new_shape_waits_for_all_stages_and_preserves_current_prompt(self):
        c = self.make_control()
        old = c.streaming_cond_embeds
        c._debug_counters['pseudo_encoded_chunks'] = 2
        captured = threading.Event()
        def capture():
            self.assertEqual(c.streaming_cond_embeds.shape[1], 9)
            self.assertEqual(c._debug_counters['pseudo_encoded_chunks'], 3)
            c.runtime.graph_runners[9] = object()
            captured.set()
        c._maybe_prepare_graph_runner = capture
        with patch.object(torch.cuda, 'device_count', return_value=0):
            t = threading.Thread(target=c._prepare_prompt_graph, args=(torch.ones(1, 9, 2), 1))
            t.start()
            self.assertFalse(captured.wait(.05))
            # Input cannot enter until the old pseudo stage finishes and capture completes.
            self.assertFalse(c._submit_lock.acquire(blocking=False))
            c._debug_counters['pseudo_encoded_chunks'] = 3
            t.join(2)
        self.assertFalse(t.is_alive())
        self.assertTrue(captured.is_set())
        self.assertIs(c.streaming_cond_embeds, old)
        self.assertEqual(c.prompt, 'old')
        self.assertEqual(c._prompt_revision, 0)


    def test_existing_shape_needs_no_drain_or_capture(self):
        c = self.make_control()
        c.runtime.graph_runners[9] = object()
        c._maybe_prepare_graph_runner = lambda: self.fail('recaptured cached shape')
        c._submit_lock = None
        c._prepare_prompt_graph(torch.ones(1, 9, 2), 1)

    def test_stale_request_does_not_capture(self):
        c = self.make_control()
        c._prompt_request = 2
        c._maybe_prepare_graph_runner = lambda: self.fail('captured stale request')
        c._prepare_prompt_graph(torch.ones(1, 9, 2), 1)

    def test_capture_failure_restores_embeddings_and_unlocks_input(self):
        c = self.make_control()
        old = c.streaming_cond_embeds
        def fail(): raise RuntimeError('capture failed')
        c._maybe_prepare_graph_runner = fail
        with patch.object(torch.cuda, 'device_count', return_value=0), self.assertRaises(RuntimeError):
            c._prepare_prompt_graph(torch.ones(1, 9, 2), 1)
        self.assertIs(c.streaming_cond_embeds, old)
        self.assertTrue(c._submit_lock.acquire(blocking=False))
        c._submit_lock.release()


class StreamControls(unittest.TestCase):
    def test_prepared_scene_initialization_skips_encoder_and_consumes_embeddings(self):
        c = control()
        embeds = torch.ones(1, 9, 2)
        c._initial_prompt_embeds = embeds
        c._encode_prompt_value = lambda *args: self.fail("New scene re-encoded prepared prompt")
        c._encode_streaming_prompt(None)
        self.assertIs(c.streaming_cond_embeds, embeds)
        self.assertIsNone(c._initial_prompt_embeds)

    def test_hot_prompt_never_captures_with_active_vae_workers(self):
        c = control()
        c._prompt_request = 1
        c._prepared_prompt = (1, "new shape", torch.ones(1, 9, 2))
        c._maybe_prepare_graph_runner = lambda: self.fail("Unsafe live graph capture")
        profile = {}
        c._apply_prepared_prompt(95, profile)
        self.assertTrue(profile["prompt_graph_capture_deferred"])
        self.assertEqual(c._prompt_revision, 1)
        self.assertEqual(c._session_serial, 77)

    def test_hot_prompt_reuses_existing_graph_without_recapture(self):
        c = control()
        c._prompt_request = 1
        c._prepared_prompt = (1, "same shape", torch.ones(1, 2, 2))
        calls = []
        c.runtime.graph_runners["graph"] = SimpleNamespace(bind_session=lambda *args: calls.append(args))
        c._maybe_prepare_graph_runner = lambda: self.fail("Unnecessary capture")
        profile = {}
        c._apply_prepared_prompt(95, profile)
        self.assertTrue(profile["prompt_graph_reused"])
        self.assertEqual(calls[0][0], 77)

    def test_continuous_preset_bounds_positions_without_periodic_reset(self):
        preset = json.loads((source.parents[3] / 'tulip/presets/etherea-continuous-v2.json').read_text())
        self.assertEqual(preset['start']['max_temporal_ids'], 8)
        self.assertEqual(preset['start']['kv_reset_frames'], 0)
        self.assertEqual(preset['start']['num_inference_steps'], 2)

    def test_latest_wins_bounded_and_does_not_apply_on_encoder_thread(self):
        c = control()
        entered, release = threading.Event(), threading.Event()
        calls = []
        def encode(text, anchor):
            calls.append(text)
            entered.set()
            self.assertTrue(release.wait(3))
            return torch.ones(1, 2, 2)
        c._encode_prompt_value = encode
        c.queue_prompt("first")
        self.assertTrue(entered.wait(3))
        for i in range(100):
            c.queue_prompt(f"latest-{i}")
        thread = c._prompt_thread
        release.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(calls, ["first", "latest-99"])
        self.assertEqual(c.prompt, "old")
        c._apply_prepared_prompt(200, {})
        self.assertEqual(c.prompt, "latest-99")
        self.assertEqual(c._prompt_revision, 101)
        self.assertEqual(c._session_serial, 77)

    def test_encoding_error_retains_old_prompt(self):
        c = control()
        c._encode_prompt_value = lambda *args: (_ for _ in ()).throw(ValueError("encoder test failure"))
        c.queue_prompt("new")
        thread = c._prompt_thread
        if thread:
            thread.join(3)
        self.assertEqual(c.prompt, "old")
        self.assertEqual(c._prompt_status, "error")
        self.assertIsNone(c._prepared_prompt)

    def test_invalid_and_uninitialized_requests_rejected(self):
        c = control()
        for text in ("", " " * 8, "x" * 12001):
            with self.assertRaises(ValueError):
                c.queue_prompt(text)
        c.initialized = False
        with self.assertRaises(ValueError):
            c.queue_prompt("new")

    def test_stale_prepared_prompt_not_applied(self):
        c = control()
        c._prompt_request = 2
        c._prepared_prompt = (1, "obsolete", torch.ones(1, 2, 2))
        c._apply_prepared_prompt(200, {})
        self.assertEqual(c.prompt, "old")

    def test_rolling_sink_clones_publish_buffer_and_does_not_grow_cache(self):
        c = control()
        entry = lambda value: {0: {"key": torch.full((2, 2), value), "value": torch.ones(2, 2), "pre_rope": True}}
        store = {0: entry(0), 32: entry(3)}
        c.pipeline = SimpleNamespace(transformer=SimpleNamespace(_inference_kv_cache={"cond": store}),
                                     _kv_cache_memory_id=lambda kind, i: i)
        runner = SimpleNamespace(pool_dirty=False)
        c.runtime.graph_runners["graph"] = runner
        with patch.object(torch.cuda, "memory_allocated", return_value=1024), patch.object(torch.cuda, "memory_reserved", return_value=2048):
            c._refresh_sink_and_measure(32, {})
        store[32][0]["key"].zero_()
        self.assertEqual(store[0][0]["key"].sum().item(), 12)
        self.assertEqual(len(store), 2)
        self.assertTrue(runner.pool_dirty)
        self.assertEqual(c._cache_stats["sinkRefreshes"], 1)


if __name__ == "__main__":
    unittest.main()
