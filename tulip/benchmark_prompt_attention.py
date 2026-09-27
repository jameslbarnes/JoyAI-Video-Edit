"""Bounded attention probe; does not mutate the live runtime."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from sageattention import sageattn
from sageattention.core import sageattn_qk_int8_pv_fp16_triton
from probe_prefix_sage import prefix_sage

torch.set_num_threads(1)
torch.manual_seed(42)
reports = []
# 840x480 -> 35x20 spatial latent, image + input queries and two cached chunks.
for capacity in (512, 640, 1024):
    ct = 700
    q = torch.randn(1, 2*ct+capacity, 32, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, 4*ct+capacity, 32, 128, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = (torch.arange(k.shape[1], device='cuda') < 4*ct+475)[None,None,None,:]
    def cudnn():
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), attn_mask=mask).transpose(1,2)
    funcs = {
        'cudnn': cudnn,
        'sage-masked-triton': lambda: sageattn_qk_int8_pv_fp16_triton(q,k,v,tensor_layout='NHD',attn_mask=mask),
        'sage-prefix': lambda: prefix_sage(q,k,v,mask),
        'sage-unmasked': lambda: sageattn(q,k,v,tensor_layout='NHD'),
    }
    for name, fn in funcs.items():
        try:
            for _ in range(3): fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): out = fn()
            a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(100): g.replay()
            b.record(); b.synchronize()
            row = {'capacity':capacity,'backend':name,'graphMs':a.elapsed_time(b)/100}
            if name != 'sage-unmasked':
                # Replay must use new Q/K/V and a different valid length.
                q.normal_();k.normal_();v.normal_()
                valid = 4*ct+177
                mask.copy_((torch.arange(k.shape[1],device='cuda') < valid)[None,None,None,:])
                g.replay();torch.cuda.synchronize()
                ref = F.scaled_dot_product_attention(q.transpose(1,2), k[:,:valid].transpose(1,2), v[:,:valid].transpose(1,2)).transpose(1,2)
                row['relativeL2'] = float((out-ref).float().norm()/ref.float().norm())
                assert row['relativeL2'] < .03, row
                mask.copy_((torch.arange(k.shape[1],device='cuda') < 4*ct+475)[None,None,None,:])
            del g,out
        except Exception as exc:
            row = {'capacity':capacity,'backend':name,'error':repr(exc)}
        reports.append(row); print(json.dumps(row),flush=True)
Path('tulip/prompt-attention-probe-20260918.json').write_text(json.dumps(reports,indent=2))
