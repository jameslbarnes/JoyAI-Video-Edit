"""Run the real prepare_scene handler with injected native allocation failures."""
import ast
import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class SceneBridgeFailureTests(unittest.IsolatedAsyncioTestCase):
    async def run_handler(self, factory, previous=None, initialized=True):
        path = Path(__file__).resolve().parents[1] / "deploy/xvideo/serving/serve_joyomni_streaming.py"
        source = ast.parse(path.read_text(encoding="utf-8"))
        branch = next(n for n in ast.walk(source) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "msg_type == 'prepare_scene'")
        branch = copy.deepcopy(branch)
        branch.orelse = []
        harness = ast.parse('''async def exercise(session, app, spout_input, spout_bridge):
    msg_type = "prepare_scene"
    payload = {"prompt": "unchanged scene"}
    scene_task = None
    scene_serial = 0
    prepared_scene = None
    live_scene_transition = False
    for _ in range(1):
        pass
    if scene_task is not None:
        await scene_task
    return spout_bridge
''')
        loop = next(n for n in ast.walk(harness) if isinstance(n, ast.For))
        loop.body = [branch]
        events, messages, debug = [], [], {}
        async def prepare(session, prompt, serial):
            events.append((session, prompt, serial))
        async def send(message):
            messages.append(message)
        ns = dict(asyncio=asyncio, __package__="xvideo.serving", ws_debug=debug,
                  _prepare_scene=prepare, _send_json=send)
        exec(compile(ast.fix_missing_locations(harness), str(path), "exec"), ns)
        frame = object()
        session = SimpleNamespace(initialized=initialized, _latest_source_frame=frame)
        upscaler = SimpleNamespace(direct_requested=True, begin_transition=Mock())
        app = SimpleNamespace(state=SimpleNamespace(shader_bridges={}, upscaler=upscaler))
        reader = SimpleNamespace(source="shader", device="cuda:1", close=Mock())
        with patch("xvideo.serving.spout_input.SpoutShaderBridge", factory):
            bridge = await ns["exercise"](session, app, reader, previous)
        reader.close.assert_not_called()
        return SimpleNamespace(bridge=bridge, events=events, messages=messages,
                               debug=debug, upscaler=upscaler, frame=frame)

    async def test_d3d_allocation_failure_prepares_scene_without_disconnect(self):
        result = await self.run_handler(Mock(side_effect=RuntimeError("D3D11 input failure: -2147024882")))
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.messages, [])
        self.assertIsNone(result.bridge)
        self.assertEqual(result.debug["scene_bridge_fallbacks"], 1)
        call = result.upscaler.begin_transition.call_args
        self.assertFalse(call.kwargs["moving"])
        self.assertIs(call.args[0](), result.frame)

    async def test_capture_failure_also_falls_back_after_closing_old_receiver(self):
        old = Mock()
        result = await self.run_handler(Mock(side_effect=TimeoutError("no fresh shader frame")), old)
        old.close.assert_called_once_with()
        self.assertEqual(len(result.events), 1)
        self.assertIsNone(result.bridge)

    async def test_unstoppable_old_receiver_keeps_ownership_and_current_scene(self):
        old = Mock()
        old.close.side_effect = RuntimeError("receiver thread did not stop")
        factory = Mock()
        result = await self.run_handler(factory, old)
        factory.assert_not_called()
        self.assertIs(result.bridge, old)
        self.assertEqual(result.events, [])
        self.assertEqual(result.messages[0]["type"], "scene_error")
        result.upscaler.begin_transition.assert_not_called()

    async def test_healthy_moving_bridge_is_preserved(self):
        bridge = Mock()
        bridge.latest.return_value = object()
        result = await self.run_handler(Mock(return_value=bridge))
        self.assertIs(result.bridge, bridge)
        self.assertEqual(len(result.events), 1)
        call = result.upscaler.begin_transition.call_args
        self.assertTrue(call.kwargs["moving"])
        self.assertIs(call.args[0](), bridge.latest.return_value)

    async def test_invalid_session_never_allocates_receiver(self):
        factory = Mock()
        result = await self.run_handler(factory, initialized=False)
        factory.assert_not_called()
        self.assertEqual(result.events, [])
        self.assertEqual(result.messages[0]["type"], "scene_error")


if __name__ == "__main__":
    unittest.main()
