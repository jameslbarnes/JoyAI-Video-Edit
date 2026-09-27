import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from xvideo.serving.stage_timings import StageTimings


class Event:
    ready = False
    def __init__(self, **kw): self.stream = None
    def record(self, stream): self.stream = stream
    def query(self): return self.ready
    def elapsed_time(self, other):
        assert other.ready
        return 12.5
    def synchronize(self): raise AssertionError('Profiling must not synchronize')


class StageTimingTests(unittest.TestCase):
    def test_deferred_completion_bounded_storage_and_live_disable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profile.json'
            path.write_text('{"enabled":true}')
            profiler = StageTimings(path, capacity=2)
            with patch('torch.cuda.Event', Event), patch('torch.cuda.current_stream', return_value='stream-1'):
                with profiler.measure('encode', 'cuda:1', 4): pass
                self.assertEqual(profiler.snapshot()['summary'], {})
                self.assertEqual(len(profiler.pending), 1)
                start, end, row = profiler.pending[0]
                self.assertEqual(start.stream, 'stream-1')
                end.ready = True
                self.assertEqual(profiler.snapshot()['summary']['encode@cuda:1']['medianMs'], 12.5)
                for chunk in [5,6,7]:
                    with profiler.measure('encode', 'cuda:1', chunk): pass
                self.assertEqual(len(profiler.pending),2)
                self.assertEqual(profiler.skipped,1)
                path.write_text('{"enabled":false}');profiler.checked_at=0
                with profiler.measure('decode','cuda:1',8): pass
                self.assertFalse(profiler.snapshot()['enabled'])
                self.assertEqual(len(profiler.pending),2)

    def test_disabled_startup_and_malformed_config_do_not_touch_cuda(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'profile.json';p=StageTimings(path)
            with patch('torch.cuda.Event', side_effect=AssertionError('Unexpected CUDA event')):
                with p.measure('encode','cuda:0',5): pass
                path.write_text('{"enabled":true}');p.checked_at=0
                with p.measure('encode','cuda:0',0): pass
                path.write_text('{"enabled":"yes"}');p.checked_at=0
                with p.measure('encode','cuda:0',5): pass
                self.assertFalse(p.snapshot()['enabled'])
                self.assertIsNotNone(p.error)

if __name__ == '__main__': unittest.main()
