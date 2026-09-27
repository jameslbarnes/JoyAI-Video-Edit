"""Compare launch policies for the existing Sage varlen attention kernel."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json, statistics
from pathlib import Path
import torch
from sageattention.triton.attn_qk_int8_block_varlen import _attn_fwd
from xvideo.models.dit.prefix_attention import prepare_prefix_mask,set_prefix_length,prefix_attention
torch.set_num_threads(1);torch.manual_seed(42)

def policy(warps,stages):
    def forward(q,k,v,cq,ck,nq,qs,ks,sq,sk,output_dtype):
        out=torch.empty_like(q,dtype=output_dtype)
        h,d=q.shape[1:]
        _attn_fwd[((nq+127)//128,h,1)](
            q,k,v,cq,ck,qs,ks,sq,sk,out,
            q.stride(1),q.stride(0),k.stride(1),k.stride(0),
            v.stride(1),v.stride(0),out.stride(1),out.stride(0),h,h//k.shape[1],
            BLOCK_M=128,BLOCK_N=64,HEAD_DIM=d,STAGE=1,num_warps=warps,num_stages=stages)
        return out
    return forward

q=torch.randn(1,1912,32,128,device='cuda',dtype=torch.bfloat16)
k=torch.randn(1,3312,32,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
mask=(torch.arange(3312,device='cuda')<3275)[None,None,None,:]
prepare_prefix_mask(mask,1912);set_prefix_length(mask,3275)
graphs=[]
for warps,stages in ((8,4),(8,2),(8,3),(4,2),(4,3),(4,4)):
    mask._sage_prefix=(*mask._sage_prefix[:-1],policy(warps,stages))
    for _ in range(3):prefix_attention(q,k,v,mask)
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=prefix_attention(q,k,v,mask)
    graphs.append((warps,stages,g,out,[]))
for repeat in range(7):
    for warps,stages,g,out,times in graphs:
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(20):g.replay()
        b.record();b.synchronize();times.append(a.elapsed_time(b)/20)
q.normal_();k.normal_();v.normal_();mask.fill_(False);mask[...,:2877]=True;set_prefix_length(mask,2877)
ref=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k[:,:2877].transpose(1,2),v[:,:2877].transpose(1,2)).transpose(1,2)
rows=[]
for warps,stages,g,out,times in graphs:
    g.replay();torch.cuda.synchronize()
    relative=float((out-ref).float().norm()/ref.float().norm())
    assert relative<.03,relative
    rows.append({'warps':warps,'stages':stages,'medianMs':statistics.median(times),'minMs':min(times),'relativeL2':relative})
print(json.dumps(rows),flush=True)
Path('tulip/prefix-kernel-tuning-20260918.json').write_text(json.dumps(rows,indent=2))
