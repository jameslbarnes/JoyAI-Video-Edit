"""Exercise actual upload/packing methods without constructing model weights."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from typing import Any
import numpy as np
import torch
from PIL import Image
from einops import rearrange

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'deploy'))
from xvideo.serving.shader_strength import ShaderStrength
source=root/'deploy/xvideo/serving/joyomni_streaming.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JoyOmniV2VStreamingSession')
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_pack_decoded_pixels','_frames_to_tensor')]
ns={'torch':torch,'np':np,'Image':Image,'rearrange':rearrange,'Any':Any,
    '__package__':'xvideo.serving','_module_device':lambda _:torch.device('cpu')}
exec(compile(ast.Module(body=methods,type_ignores=[]),str(source),'exec'),ns)
Session=type('Session',(),{m.name:ns[m.name] for m in methods})

class ShaderPresentation(unittest.TestCase):
    def test_actual_upload_survives_to_raw_packing_for_first_and_streaming_chunks(self):
        for count in (1,8):
            s=Session();s.pipeline=SimpleNamespace(vae=None);s._resize_frame=lambda f:f
            s.postprocess_device='cpu';s.runtime=SimpleNamespace(lossless_output=True)
            s.settings=SimpleNamespace(output_codec='mjpeg')
            arrays=[np.random.default_rng(i).integers(0,256,(6,8,3),dtype=np.uint8) for i in range(count)]
            job=SimpleNamespace(source_pixels=None)
            pixels=s._frames_to_tensor([Image.fromarray(a) for a in arrays],source_job=job)
            self.assertEqual(tuple(pixels.shape),(1,3,count,6,8))
            np.testing.assert_array_equal(job.source_pixels.numpy(),np.stack(arrays))
            decoded=torch.full_like(pixels,-1)
            profile={'shader_strength':ShaderStrength(1,mode='combined').snapshot()}
            packed=s._pack_decoded_pixels(decoded,profile=profile,chunk_idx=0,source_pixels=job.source_pixels)
            np.testing.assert_array_equal(np.stack(packed),np.stack(arrays))
            self.assertTrue(profile['shader_presentation']['rawSource'])
            self.assertEqual(profile['shader_presentation']['additionalDownloadBytes'],0)
            self.assertTrue(torch.all(decoded==-1)) # pseudo encoder receives unmodified generated pixels
            profile={'shader_strength':ShaderStrength(.6,mode='combined').snapshot()}
            generated=s._pack_decoded_pixels(decoded,profile=profile,chunk_idx=0,source_pixels=job.source_pixels)
            self.assertFalse(np.any(np.stack(generated)))

if __name__=='__main__':unittest.main()
