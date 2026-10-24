"""Color/packing contract checks against an independent CPU reference."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy'))
from xvideo.serving.ndi_pixels import ndi_coefficients, pack_uyvy


def reference(rgb):
    h,w,_=rgb.shape
    kr,kb=ndi_coefficients(w,h)
    p=rgb.astype(np.float64).reshape(h,w//2,2,3)
    y=kr*p[:,:,:,0]+(1-kr-kb)*p[:,:,:,1]+kb*p[:,:,:,2]
    u=128+(p[:,:,:,2]-y).mean(axis=2)*(112/(255*(1-kb)))
    v=128+(p[:,:,:,0]-y).mean(axis=2)*(112/(255*(1-kr)))
    out=np.stack([u,16+y[:,:,0]*219/255,v,16+y[:,:,1]*219/255],axis=2)
    return np.floor(out+.5).clip(0,255).astype(np.uint8).reshape(h,w,2)


class NdiPixels(unittest.TestCase):
    def test_resolution_matrix_boundaries(self):
        self.assertEqual(ndi_coefficients(720,576),(.299,.114))
        self.assertEqual(ndi_coefficients(1920,1080),(.2126,.0722))
        self.assertEqual(ndi_coefficients(1944,1080),(.2627,.0593))
        self.assertEqual(ndi_coefficients(1920,1082),(.2627,.0593))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_color_bars_and_random_pairs(self):
        colors=np.array([[0,0,0],[255,255,255],[255,0,0],[0,255,0],
                         [0,0,255],[0,255,255],[255,0,255],[255,255,0]],np.uint8)
        rng=np.random.default_rng(654)
        for h,w in [(360,648),(720,1296),(1080,1944),(1440,2592)]:
            with self.subTest(size=(w,h)):
                rgb=rng.integers(0,256,(h,w,3),dtype=np.uint8)
                rgb[:8,:,:]=colors[:,None,:]
                tensor=torch.from_numpy(rgb).to('cuda:0')
                got=pack_uyvy(tensor).cpu().numpy()
                expected=reference(rgb)
                self.assertLessEqual(np.abs(got.astype(int)-expected.astype(int)).max(),1)
                self.assertTrue(np.array_equal(got[0,0:2].reshape(-1),[128,16,128,16]))
                self.assertTrue(np.array_equal(got[1,0:2].reshape(-1),[128,235,128,235]))
                self.assertEqual(got.nbytes,w*h*2)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_reused_destination_and_validation(self):
        rgb=torch.zeros((2,4,3),dtype=torch.uint8,device='cuda:0')
        dest=torch.empty((2,4,2),dtype=torch.uint8,device='cuda:0')
        self.assertEqual(pack_uyvy(rgb,dest).data_ptr(),dest.data_ptr())
        rgb.fill_(255)
        pack_uyvy(rgb,dest)
        self.assertEqual(dest.cpu()[0,0].tolist(),[128,235])
        with self.assertRaises(ValueError):pack_uyvy(rgb[:,:3])
        with self.assertRaises(ValueError):pack_uyvy(rgb.cpu())


if __name__=='__main__':unittest.main()
