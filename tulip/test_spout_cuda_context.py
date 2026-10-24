import ctypes
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from xvideo.serving.spout_output import tensor_cuda_context


class ContextTests(unittest.TestCase):
    def driver(self):
        def pointer(out, attribute, pointer):
            self.assertEqual((attribute, pointer), (1, 12345))
            ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = 67890
            return 0
        return SimpleNamespace(cuPointerGetAttribute=Mock(side_effect=pointer),
                               cuCtxPushCurrent_v2=Mock(return_value=0),
                               cuCtxPopCurrent_v2=Mock(return_value=0))

    def test_binds_allocation_context_and_restores_after_native_failure(self):
        driver = self.driver()
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            with tensor_cuda_context(SimpleNamespace(data_ptr=lambda: 12345), driver):
                self.assertEqual(driver.cuCtxPushCurrent_v2.call_args.args[0].value, 67890)
                driver.cuCtxPopCurrent_v2.assert_not_called()
                raise RuntimeError('native failure')
        driver.cuCtxPopCurrent_v2.assert_called_once()

    def test_failed_lookup_never_pushes_or_pops(self):
        driver = self.driver()
        driver.cuPointerGetAttribute.side_effect = None
        driver.cuPointerGetAttribute.return_value = 1
        with self.assertRaisesRegex(RuntimeError, 'resolve'):
            with tensor_cuda_context(SimpleNamespace(data_ptr=lambda: 12345), driver):
                self.fail('must not create a sender')
        driver.cuCtxPushCurrent_v2.assert_not_called()
        driver.cuCtxPopCurrent_v2.assert_not_called()

    def test_failed_push_does_not_pop_someone_elses_context(self):
        driver = self.driver()
        driver.cuCtxPushCurrent_v2.return_value = 1
        with self.assertRaisesRegex(RuntimeError, 'bind'):
            with tensor_cuda_context(SimpleNamespace(data_ptr=lambda: 12345), driver):
                self.fail('must not create a sender')
        driver.cuCtxPopCurrent_v2.assert_not_called()
