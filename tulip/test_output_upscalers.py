"""Real-GPU shape/determinism tests; run with CUDA_VISIBLE_DEVICES=0.
Only SR / tiny VAE weights load. No live model or show state is changed.
"""
import unittest
import cv2
import numpy as np
import torch
from upscale_trial import Upscaler, FluxLatentUpscaler
from upscale_live_trial import bridge_request


class OutputUpscalersTest(unittest.TestCase):
    def test_bridge_hello_is_not_a_command_ack(self):
        class Socket:
            def __init__(self):self.messages=iter(['{"type":"hello"}','{"id":1,"result":{"ok":true}}'])
            def send(self,_):pass
            def recv(self,timeout):return next(self.messages)
        self.assertEqual(bridge_request(Socket(),1,'video_sender_stop',{}),{'ok':True})

    def test_denoise_validation(self):
        for value in (-1,2,float('nan')):
            with self.assertRaises(ValueError):Upscaler(value)

    def test_pixel_shape_finite_deterministic_and_nontrivial(self):
        model=Upscaler(.5)
        rng=np.random.default_rng(52)
        frame=rng.integers(0,256,(40,56,3),dtype=np.uint8)
        first,profile=model.process(frame)
        second,_=model.process(frame)
        self.assertEqual(first.shape,(80,112,3))
        self.assertEqual(first.dtype,np.uint8)
        self.assertTrue(np.array_equal(first,second))
        self.assertGreater(profile['pipeline_ms'],0)
        self.assertGreater(np.abs(first.astype(float)-cv2.resize(frame,(112,80),interpolation=cv2.INTER_LANCZOS4)).mean(),1)

    def test_latent_roundtrip_preserves_output_geometry(self):
        model=FluxLatentUpscaler()
        frame=np.zeros((40,56,3),np.uint8)
        frame[8:32,12:44]=[210,100,40]
        first,_=model.process(frame)
        second,_=model.process(frame)
        self.assertEqual(first.shape,(80,112,3))
        self.assertEqual(first.dtype,np.uint8)
        self.assertTrue(np.array_equal(first,second),'Fixed noise must not introduce random frame flicker')
        self.assertGreater(first.std(),1)


if __name__=='__main__':
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.12,0)
    unittest.main()
