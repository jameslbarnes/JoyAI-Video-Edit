"""Exercise the actual ASGI preset wrapper without loading model checkpoints."""
import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from urllib.parse import urlparse


class PresetPreviewProtocol(unittest.IsolatedAsyncioTestCase):
    async def through_wrapper(self, payload):
        path = Path(__file__).with_name("serve.py")
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LocalPreset")
        seen = []
        async def receive():
            return {"type": "websocket.receive", "text": json.dumps(payload)}
        async def application(scope, receive, send):
            seen.append(json.loads((await receive())["text"]))
        async def send(message):
            self.fail(f"Local request unexpectedly rejected: {message}")
        with tempfile.TemporaryDirectory() as directory:
            native = dict(width=1152, height=648, fps=12, rife_multiplier=2)
            namespace = dict(json=json, urlparse=urlparse, ROOT=Path(directory),
                             load_native_config=lambda: native.copy(), native_quality=native,
                             load_input_config=lambda: {"transport": "spout"},
                             args=SimpleNamespace(), preset={"prompt": "default", "start":
                             {"type": "start", "width": 1, "height": 1, "fps": 1}})
            exec(compile(ast.Module(body=[wrapper], type_ignores=[]), str(path), "exec"), namespace)
            await namespace["LocalPreset"](application)(
                {"type": "websocket", "headers": [(b"origin", b"http://127.0.0.1:7860")]}, receive, send)
        return seen[0]

    async def test_negotiated_preview_survives_the_production_preset_wrapper(self):
        payload = await self.through_wrapper(dict(type="start", spout_input=True,
            independent_preview=True, prompt="current art", width=4096, fps=120, rife_multiplier=2))
        self.assertIs(payload["independent_preview"], True)
        self.assertEqual(payload["input_codec"], "spout")
        self.assertEqual(payload["prompt"], "current art")
        self.assertEqual((payload["width"], payload["height"], payload["fps"]), (1152, 648, 12))

    async def test_legacy_and_untyped_requests_keep_the_legacy_contract(self):
        for value in (None, False, "true", 1):
            payload = await self.through_wrapper(dict(type="start", independent_preview=value))
            self.assertIs(payload["independent_preview"], False)
            self.assertEqual(payload["input_codec"], "mjpeg")


if __name__ == "__main__":
    unittest.main()
