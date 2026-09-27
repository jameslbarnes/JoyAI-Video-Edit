"""Cache teardown must wait for GPU completion, not merely CPU worker exit."""
import ast
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / 'deploy/xvideo/serving/joyomni_streaming.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
session = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'JoyOmniV2VStreamingSession')
close = next(n for n in session.body if isinstance(n, ast.FunctionDef) and n.name == 'close')


class SessionCacheTeardown(unittest.TestCase):
    def fixture(self, cuda=True, fail_device=None):
        calls = []
        pending = {'cuda:0', 'cuda:1'} if cuda else set()

        def synchronize(device):
            calls.append(('synchronize', str(device)))
            if device == fail_device:
                raise RuntimeError('GPU work failed')
            pending.discard(str(device))

        def clear():
            self.assertFalse(pending, 'Cleared shared caches with unfinished GPU work')
            calls.append('clear-vae')

        ns = {'torch': SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda, synchronize=synchronize),
            device=lambda value: SimpleNamespace(type=str(value).split(':')[0])),
            '_module_device': lambda module: module.device}
        exec(compile(ast.Module(body=[close], type_ignores=[]), str(SOURCE), 'exec'), ns)
        model_device = 'cuda:0' if cuda else 'cpu'
        output_device = 'cuda:1' if cuda else 'cpu'
        c = SimpleNamespace(
            _prompt_lock=threading.Lock(), _closing=False,
            _pending_prompt=object(), _prepared_prompt=object(),
            _prompt_thread=SimpleNamespace(join=lambda: calls.append('join-prompt')),
            # Worker exit deliberately leaves simulated CUDA work pending.
            _stop_async_workers=lambda: calls.append('join-workers'),
            _clear_vae_feature_caches=clear,
            device=model_device, postprocess_device=output_device,
            pipeline=SimpleNamespace(vae=SimpleNamespace(device=model_device),
                transformer=SimpleNamespace(reset_inference_kv_cache=lambda: calls.append('clear-kv'))),
            decode_vae=SimpleNamespace(device=model_device),
            pseudo_encode_vae=SimpleNamespace(device=output_device))
        return c, calls, ns['close']

    def test_all_used_devices_finish_before_any_shared_cache_is_cleared(self):
        c, calls, close_session = self.fixture()
        close_session(c)
        self.assertEqual(calls, ['join-prompt', 'join-workers',
            ('synchronize', 'cuda:0'), ('synchronize', 'cuda:1'), 'clear-vae', 'clear-kv'])
        self.assertTrue(c._closing)
        self.assertIsNone(c._pending_prompt)
        self.assertIsNone(c._prepared_prompt)

    def test_gpu_failure_does_not_clear_caches_as_if_work_completed(self):
        c, calls, close_session = self.fixture(fail_device='cuda:1')
        with self.assertRaisesRegex(RuntimeError, 'GPU work failed'):
            close_session(c)
        self.assertNotIn('clear-vae', calls)
        self.assertNotIn('clear-kv', calls)

    def test_cpu_session_needs_no_cuda_operations(self):
        c, calls, close_session = self.fixture(cuda=False)
        close_session(c)
        self.assertEqual(calls, ['join-prompt', 'join-workers', 'clear-vae', 'clear-kv'])


if __name__ == '__main__':
    unittest.main()
