import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from xvideo.serving.memory_status import memory_status, thread_stacks


class MemoryStatusTest(unittest.TestCase):
    def test_stacks_are_bounded_and_exclude_locals(self):
        secret_local = 'must not be exposed'
        stacks = thread_stacks()
        self.assertLessEqual(len(stacks), 128)
        self.assertNotIn(secret_local, str(stacks))
        self.assertTrue(any(any(f['function'] == 'test_stacks_are_bounded_and_exclude_locals'
                               for f in t['stack']) for t in stacks))
        self.assertTrue(all(len(t['stack']) <= 32 for t in stacks))

    def test_unloaded_runtime_never_initializes_cuda(self):
        cuda = SimpleNamespace(is_initialized=lambda: False)
        self.assertEqual(memory_status(cuda=cuda)['devices'], [])

    def test_active_tensors_and_reserved_cache_are_separate_on_each_gpu(self):
        cuda = SimpleNamespace(is_initialized=lambda: True, device_count=lambda: 2,
            memory_stats=lambda i: {'allocated_bytes.all.current': (i+1)*2**20,
                                    'reserved_bytes.all.current': (i+4)*2**20,
                                    'inactive_split_bytes.all.current': 2**19})
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1,0'}):
            result = memory_status(SimpleNamespace(graph_runners={'one': object()}), cuda=cuda)
        self.assertEqual(result['graphRunners'], 1)
        self.assertEqual(result['devices'][0]['physicalGpu'], '1')
        self.assertEqual(result['devices'][1]['physicalGpu'], '0')
        self.assertEqual(result['devices'][1]['allocatedMiB'], 2)
        self.assertEqual(result['devices'][1]['reservedMiB'], 5)
        self.assertEqual(result['devices'][1]['inactiveSplitMiB'], .5)


if __name__ == '__main__':
    unittest.main()
