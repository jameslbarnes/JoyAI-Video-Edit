"""CPU-only migration, write-failure and control ownership regressions."""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy'))
from xvideo.serving import shader_strength as strength
from xvideo.serving.shader_control_queue import ShaderControlQueue


def migrated(state):
    config = strength.migrate_strength_state(state)
    return strength.ShaderStrength(config['value'], mode=config['mode'], output_mix=config['outputMix'])


class InfluenceTests(unittest.TestCase):
    def test_migration_preserves_actual_inputs_including_old_high_end_overlay(self):
        for mode, values in [('combined', [i/100 for i in range(101)]),
                             ('reference_only', [0, .1, .5, 1]),
                             ('direct_latent_blend', [0, .001, .03, .1])]:
            for value in values:
                before = strength.ShaderStrength(value, mode=mode).snapshot()
                after = migrated(before).snapshot()
                for key in ('blend', 'referenceGain', 'outputMix'):
                    self.assertAlmostEqual(after[key], before[key], places=12, msg=(mode, value, key))
                self.assertTrue(after['settled'])

    def test_original_show_setting_maps_without_changing_inputs(self):
        old = strength.ShaderStrength(.2964197530864196, mode='combined').snapshot()
        new = migrated(old).snapshot()
        self.assertAlmostEqual(new['target'], .24701646090534968)
        self.assertEqual(new['referenceGain'], old['referenceGain'])
        self.assertEqual(new['blend'], 0)
        self.assertEqual(new['outputMix'], 0)

    def test_migration_rejects_unsettled_or_unrepresentable_settings(self):
        old = strength.ShaderStrength(.2, mode='combined')
        old.set(.9)
        with self.assertRaises(ValueError): migrated(old.snapshot())
        with self.assertRaises(ValueError): migrated(strength.ShaderStrength(.3).snapshot())

    def test_dial_never_changes_independent_overlay_and_has_no_high_end_cliff(self):
        dial = strength.ShaderStrength(.5, mode='influence', output_mix=.25)
        previous = (0, 0)
        for i in range(101):
            blend, gain, implicit_overlay = strength.strength_inputs('influence', i/100)
            self.assertGreaterEqual(blend, previous[0])
            self.assertGreaterEqual(gain, previous[1])
            self.assertEqual(implicit_overlay, 0)
            previous = blend, gain
        for value in [0, .6, 1, .5, .95]:
            dial.set(value)
            for _ in range(16): dial.advance()
            self.assertTrue(dial.snapshot()['settled'])
            self.assertEqual(dial.snapshot()['outputMix'], .25)
        self.assertAlmostEqual(strength.strength_inputs('influence', .6)[0], .004)
        self.assertAlmostEqual(strength.strength_inputs('influence', .95)[0], .081)

    def test_overlay_changes_no_inference_inputs_or_history(self):
        dial = strength.ShaderStrength(.8, mode='influence')
        before = dial.snapshot()
        dial.set_overlay(1)
        for _ in range(2): after = dial.advance()
        self.assertEqual(after['outputMix'], 1)
        for key in ('target', 'value', 'blend', 'referenceGain'):
            self.assertEqual(after[key], before[key])
        self.assertTrue(after['cachePreserved'])
        with self.assertRaises(ValueError): strength.ShaderStrength(.6, mode='combined').set_overlay(.5)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'strength.json'
        self.dial = strength.ShaderStrength(.5, mode='influence', config_path=self.path)
        self.dial.set(.5)

    def test_acknowledged_strength_and_overlay_survive_cold_reload(self):
        self.dial.set(.72)
        ack = self.dial.set_overlay(.18)
        reloaded = strength.ShaderStrength(**strength.load_strength_config(self.path), config_path=self.path).snapshot()
        self.assertEqual(reloaded['target'], ack['target'])
        self.assertEqual(reloaded['outputMix'], ack['outputMixTarget'])
        self.assertEqual(reloaded['mode'], 'influence')
        self.assertTrue(reloaded['persistent'])
        self.assertTrue(reloaded['settled'])

    def test_failed_flush_or_replace_preserves_last_accepted_file_and_live_state(self):
        for name in ('fsync', 'replace'):
            before, file_before = self.dial.snapshot(), self.path.read_bytes()
            with patch.object(strength.os, name, side_effect=OSError('disk full')):
                with self.assertRaises(OSError): self.dial.set(.95)
            self.assertEqual(self.dial.snapshot(), before)
            self.assertEqual(self.path.read_bytes(), file_before)
            self.assertEqual(list(self.path.parent.glob('*.tmp')), [])

    def test_invalid_commands_never_write_or_change_state(self):
        before = self.dial.snapshot()
        with patch.object(strength, 'save_strength_config', side_effect=AssertionError('write')):
            for value in (True, None, '0.5', -1, 2, float('nan'), float('inf')):
                with self.assertRaises(ValueError): self.dial.set(value)
                with self.assertRaises(ValueError): self.dial.set_overlay(value)
            with self.assertRaises(ValueError): self.dial.set(.2, mode='typo')
            with self.assertRaises(ValueError): self.dial.set(.2, mode='combined', output_mix=.5)
        self.assertEqual(self.dial.snapshot(), before)

    def test_slow_disk_does_not_block_frame_advance_or_lose_concurrent_overlay(self):
        entered, release = threading.Event(), threading.Event()
        real_save = strength.save_strength_config
        def blocked(path, config):
            entered.set()
            if not release.wait(3): raise TimeoutError('fixture release missing')
            real_save(path, config)
        with ThreadPoolExecutor(max_workers=3) as pool:
            with patch.object(strength, 'save_strength_config', side_effect=blocked):
                writer = pool.submit(self.dial.set, .8)
                try:
                    self.assertTrue(entered.wait(1))
                    overlay = pool.submit(self.dial.set_overlay, .3)
                    frame = pool.submit(self.dial.advance).result(timeout=.5)
                    self.assertEqual(frame['target'], .5, 'uncommitted write must not affect a frame')
                finally:
                    release.set()
                writer.result(timeout=2)
                overlay.result(timeout=2)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved, {'mode':'influence', 'value':.8, 'outputMix':.3})


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocked_write_keeps_event_loop_and_dial_queue_bounded(self):
        dial = strength.ShaderStrength(.5, mode='influence')
        entered, release = threading.Event(), threading.Event()
        original = dial.set
        notices = []
        async def notify(response): notices.append(response)
        def blocked(**command):
            entered.set()
            if not release.wait(3): raise TimeoutError('fixture release missing')
            return original(**command)
        queue = ShaderControlQueue(dial, notify)
        with patch.object(dial, 'set', side_effect=blocked):
            queue.submit('set_strength', {'value':.6, 'requestId':1})
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                for i in range(1000): queue.submit('set_strength', {'value':i/999, 'requestId':i+2})
                queue.submit('set_shader_overlay', {'value':.2})
                self.assertEqual(len(queue.pending), 2)
                self.assertFalse(queue.task.done())
                self.assertEqual(dial.advance()['target'], .5)
                self.assertEqual(notices, [])
            finally:
                release.set()
            await queue.task
        self.assertEqual(dial.snapshot()['target'], 1)
        self.assertEqual(dial.snapshot()['outputMixTarget'], .2)
        self.assertEqual([n['requestId'] for n in notices], [1, 1001, None])
        await queue.close()

    async def test_disk_failure_reports_error_and_next_command_can_succeed(self):
        dial = strength.ShaderStrength(.5, mode='influence')
        notices = []
        async def notify(response): notices.append(response)
        queue = ShaderControlQueue(dial, notify)
        with patch.object(dial, 'set', side_effect=OSError('private path')):
            queue.submit('set_strength', {'value':.9, 'requestId':7})
            await queue.task
        self.assertEqual(notices[0]['type'], 'strength_error')
        self.assertNotIn('private path', notices[0]['message'])
        self.assertEqual(notices[0]['requestId'], 7)
        self.assertEqual(dial.snapshot()['target'], .5)
        queue.submit('set_strength', {'value':.7})
        await queue.task
        self.assertEqual(dial.snapshot()['target'], .7)
        await queue.close()

    async def test_close_owns_active_write_and_discards_unacknowledged_gestures(self):
        dial = strength.ShaderStrength(.5, mode='influence')
        entered, release = threading.Event(), threading.Event()
        original = dial.set
        notices = []
        async def notify(response): notices.append(response)
        def blocked(**command):
            entered.set()
            if not release.wait(3): raise TimeoutError('fixture release missing')
            return original(**command)
        queue = ShaderControlQueue(dial, notify)
        with patch.object(dial, 'set', side_effect=blocked):
            queue.submit('set_strength', {'value':.6})
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                queue.submit('set_strength', {'value':.9})
                closing = asyncio.create_task(queue.close())
                await asyncio.sleep(0)
                self.assertFalse(closing.done())
            finally:
                release.set()
            await closing
        self.assertEqual(dial.snapshot()['target'], .6)
        self.assertEqual(notices, [])
        with self.assertRaises(ValueError): queue.submit('set_strength', {'value':.2})


class RealHandlerTests(unittest.TestCase):
    def test_handlers_reject_inactive_invalid_or_unsaved_updates(self):
        from starlette.responses import JSONResponse
        path = ROOT / 'deploy/xvideo/serving/serve_joyomni_streaming.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        names = ('strength_set', 'shader_overlay_set')
        handlers = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
        for handler in handlers: handler.decorator_list = []
        app = SimpleNamespace(state=SimpleNamespace(active_session=None,
            shader_strength=strength.ShaderStrength(.5, mode='influence')))
        ns = {'app':app, 'JSONResponse':JSONResponse}
        exec(compile(ast.Module(body=handlers, type_ignores=[]), str(path), 'exec'), ns)
        for name, setter in zip(names, ('set', 'set_overlay')):
            self.assertEqual(ns[name]({'value':.7}).status_code, 409)
            app.state.active_session = object()
            self.assertEqual(ns[name]({'value':True}).status_code, 400)
            with patch.object(app.state.shader_strength, setter, side_effect=OSError('disk full')):
                response = ns[name]({'value':.7})
                self.assertEqual(response.status_code, 503)
                self.assertFalse(json.loads(response.body)['ok'])
            response = ns[name]({'value':.7})
            self.assertEqual(response.status_code, 200)
            app.state.active_session = None


if __name__ == '__main__': unittest.main()
