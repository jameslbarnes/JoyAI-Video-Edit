import ctypes
from collections import OrderedDict
from types import SimpleNamespace
import threading
import unittest
from xvideo.serving.spout_output import GpuSpoutPublisher, NativeFrameSample


class SpoutFrameOriginTests(unittest.TestCase):
    def publisher(self, fill=None):
        p = GpuSpoutPublisher.__new__(GpuSpoutPublisher)
        p.native = 1; p.name = 'fixture'; p.publisher_id = 'publisher'
        p.condition = threading.Condition()
        p.frame_origins = OrderedDict({7: {'session': 'scene', 'revision': 2}})
        p.dll = SimpleNamespace(**({'jsp_frame_trace':fill} if fill else {}))
        return p

    def test_real_native_token_maps_to_origin_not_public_frame_counter(self):
        def fill(native, rows, capacity, size):
            self.assertEqual(size,40); self.assertEqual(capacity,256)
            rows[0] = NativeFrameSample(99,7,34,1234,500)
            return 1
        p = self.publisher(fill)
        value = p.frame_trace()
        self.assertEqual(value['publisher_id'],'publisher')
        self.assertEqual(value['frames'][0]['sender_frame'],34)
        self.assertEqual(value['frames'][0]['origin']['revision'],2)
        value['frames'][0]['origin']['revision'] = 80
        self.assertEqual(p.frame_origins[7]['revision'],2)

    def test_unknown_or_shader_only_token_is_never_relabelled(self):
        def fill(native, rows, capacity, size):
            rows[0] = NativeFrameSample(1,8,35,1234,500)
            return 1
        self.assertIsNone(self.publisher(fill).frame_trace()['frames'][0]['origin'])

    def test_rollback_dll_retains_output_but_reports_missing_telemetry(self):
        value = self.publisher().frame_trace()
        self.assertFalse(value['supported']); self.assertEqual(value['frames'],[])

    def test_abi_failure_is_explicit(self):
        with self.assertRaisesRegex(RuntimeError,'ABI'):
            self.publisher(lambda *args:-1).frame_trace()
