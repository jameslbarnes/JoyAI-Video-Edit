import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from xvideo.models.vae import vae_compile

class DecodeWarmupTests(unittest.TestCase):
    def test_graph_latents_remain_fp32_with_bf16_compute(self):
        seen=[]
        def decode(z,**kwargs):
            seen.append((tuple(z.shape),z.dtype,z.is_contiguous(memory_format=torch.channels_last_3d),
                         torch.is_autocast_enabled('cpu'),torch.get_autocast_dtype('cpu')))
            return (z,)
        with patch.object(vae_compile,'maybe_setup_decode'),patch.object(torch.cuda,'is_available',return_value=False):
            vae_compile.warmup_decode(SimpleNamespace(decode=decode),64,25,44,
                device=torch.device('cpu'),dtype=torch.bfloat16,input_dtype=torch.float32)
        self.assertEqual(seen,[((1,64,t,25,44),torch.float32,True,True,torch.bfloat16) for t in (1,2)])
    def test_default_preserves_eager_input_precision(self):
        seen=[]
        def decode(z,**kwargs):seen.append(z.dtype);return(z,)
        with patch.object(vae_compile,'maybe_setup_decode'),patch.object(torch.cuda,'is_available',return_value=False):
            vae_compile.warmup_decode(SimpleNamespace(decode=decode),2,2,2,
                device=torch.device('cpu'),dtype=torch.bfloat16)
        self.assertEqual(seen,[torch.bfloat16,torch.bfloat16])

if __name__=='__main__':unittest.main()
