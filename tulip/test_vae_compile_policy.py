import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from xvideo.models.vae import vae_compile

class ReferenceCompilePolicy(unittest.TestCase):
    def setUp(self):
        vae_compile._configured_encode_dynamic.clear()

    def test_eager_reference_unwraps_native_compiled_encoder(self):
        original=lambda x:x
        vae=SimpleNamespace(_encode=SimpleNamespace(_torchdynamo_orig_callable=original))
        with patch.dict(os.environ,{'JOYOMNI_COMPILE_REFERENCE':'0'}), patch.object(vae_compile.torch,'compile',side_effect=AssertionError('unexpected compilation')):
            vae_compile.maybe_setup_encode_dynamic(vae)
        self.assertIs(vae._encode_dynamic,original)
        self.assertEqual(vae._encode_dynamic('reference'),'reference')

    def test_explicit_reference_compilation_remains_available(self):
        vae=SimpleNamespace(_encode=lambda x:x)
        with patch.dict(os.environ,{'JOYOMNI_COMPILE_REFERENCE':'1'}), patch.object(vae_compile.torch,'compile') as compile:
            vae_compile.maybe_setup_encode_dynamic(vae)
            compile.assert_called_once_with(vae._encode,mode='max-autotune-no-cudagraphs',dynamic=True)

if __name__=='__main__':unittest.main()
