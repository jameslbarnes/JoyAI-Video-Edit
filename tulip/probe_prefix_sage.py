"""Prototype graph-safe single-sequence Sage prefix attention."""
import torch
from sageattention.triton.quant_per_block_varlen import quant_per_block_int8_kernel
from sageattention.triton.attn_qk_int8_block_varlen import forward

def prefix_sage(q, k, v, mask):
    q,k,v = (x.squeeze(0) for x in (q,k,v))
    nq,h,d = q.shape
    nk = k.shape[0]
    cq = torch.zeros(2,device=q.device,dtype=torch.int32);cq[1].fill_(nq)
    ck = torch.cat((torch.zeros(1,device=q.device,dtype=torch.int32), mask.sum(dtype=torch.int32).reshape(1)))
    sq = torch.zeros(2,device=q.device,dtype=torch.int32);sq[1].fill_((nq+127)//128)
    sk = torch.zeros(2,device=q.device,dtype=torch.int32);sk[1].fill_((nk+63)//64)
    valid = mask.reshape(nk,1,1)
    km = torch.where(valid,k,0).sum(dim=0,keepdim=True)/ck[1]
    k = k-km
    qi,ki = torch.empty_like(q,dtype=torch.int8),torch.empty_like(k,dtype=torch.int8)
    qs = torch.empty(((nq+127)//128,h),device=q.device,dtype=torch.float32)
    ks = torch.empty(((nk+63)//64,h),device=q.device,dtype=torch.float32)
    quant_per_block_int8_kernel[((nq+127)//128,h,1)](q,qi,qs,cq,sq,q.stride(1),q.stride(0),qi.stride(1),qi.stride(0),sm_scale=d**-.5*1.44269504,H=h,C=d,BLK=128)
    quant_per_block_int8_kernel[((nk+63)//64,h,1)](k,ki,ks,ck,sk,k.stride(1),k.stride(0),ki.stride(1),ki.stride(0),sm_scale=1.,H=h,C=d,BLK=64)
    return forward(qi,ki,v.to(torch.float16),cq,ck,nq,qs,ks,sq,sk,output_dtype=q.dtype).unsqueeze(0)
