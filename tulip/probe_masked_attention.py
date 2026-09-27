import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch,json,time
from torch.nn.attention import sdpa_kernel,SDPBackend
from sageattention import sageattn
import torch.nn.functional as F
torch.set_num_threads(1)
d='cuda';torch.manual_seed(4)
q=torch.randn(1,1834,32,128,device=d,dtype=torch.bfloat16)
k=torch.randn(1,2644,32,128,device=d,dtype=torch.bfloat16);v=torch.randn_like(k)
mask=torch.arange(k.shape[1],device=d)<(1620+400)
qt,kt,vt=[x.transpose(1,2) for x in (q,k,v)]
results=[]
for name,fn in [('sage',lambda:sageattn(q,k,v,tensor_layout='NHD'))]+[(b.name,lambda b=b:masked(b)) for b in (SDPBackend.CUDNN_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.FLASH_ATTENTION)]:
 def masked(b):
  with sdpa_kernel([b]):return F.scaled_dot_product_attention(qt,kt,vt,attn_mask=mask[None,None,None,:])
 try:
  for _ in range(3):out=fn()
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=fn()
  a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  a.record()
  for _ in range(20):g.replay()
  z.record();z.synchronize();ms=a.elapsed_time(z)/20
  entry={'backend':name,'graphMs':ms,'finite':bool(torch.isfinite(out).all())}
  if name!='sage':
   valid=1620+177;mask.copy_(torch.arange(k.shape[1],device=d)<valid);g.replay();torch.cuda.synchronize()
   reference=F.scaled_dot_product_attention(qt[:,:,:900],kt[:,:,:valid],vt[:,:,:valid])
   actual=out[:,:,:900];entry['maxAbsError']=float((actual-reference).abs().max());entry['relativeL2']=float((actual-reference).float().norm()/reference.float().norm())
  results.append(entry);print(json.dumps(entry),flush=True)
 except Exception as e:results.append({'backend':name,'error':str(e)[:500]});print(json.dumps(results[-1]),flush=True)
from pathlib import Path
Path('tulip/masked-attention-probe-20260915.json').write_text(json.dumps(results,indent=2))
