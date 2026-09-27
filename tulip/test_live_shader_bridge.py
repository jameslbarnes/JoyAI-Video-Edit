import io
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from xvideo.serving.live_shader_bridge import LiveShaderBridge

class ShaderBridge(unittest.TestCase):
    def jpeg(self, value, size=(648,360)):
        out=io.BytesIO();Image.new('RGB',size,(value,)*3).save(out,format='JPEG');return out.getvalue()

    def test_replaces_single_frame_and_expires_without_model_input(self):
        bridge=LiveShaderBridge()
        with patch('xvideo.serving.live_shader_bridge.time.perf_counter',return_value=10):
            bridge.publish(self.jpeg(20));first=bridge.latest()
        with patch('xvideo.serving.live_shader_bridge.time.perf_counter',return_value=10.03):
            bridge.publish(self.jpeg(80))
            self.assertEqual(int(bridge.latest()[0,0,0]),80)
            self.assertEqual(int(first[0,0,0]),20)
            self.assertEqual(bridge.frames,2)
            self.assertAlmostEqual(bridge.max_gap_ms,30)
        with patch('xvideo.serving.live_shader_bridge.time.perf_counter',return_value=11):
            self.assertIsNone(bridge.latest())

    def test_rejects_invalid_size_and_payload_without_replacing_frame(self):
        bridge=LiveShaderBridge();bridge.publish(self.jpeg(20));first=bridge.frame
        for data in [b'',b'x'*(512*1024+1),self.jpeg(80,(16,16))]:
            with self.assertRaises(ValueError):bridge.publish(data)
            self.assertIs(bridge.frame,first)
