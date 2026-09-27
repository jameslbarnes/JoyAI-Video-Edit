import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch,json
from pathlib import Path
from xvideo.models.dit.dit import Transformer3DModel
from xvideo.serving.graph_runner import StreamingGraphRunner
from contextlib import nullcontext
torch.set_num_threads(1);torch.manual_seed(5)
d=torch.device('cuda:0');dtype=torch.bfloat16
model=Transformer3DModel(patch_size=[1,1,1],in_channels=4,out_channels=4,hidden_size=128,heads_num=1,text_states_dim=64,mlp_width_ratio=2,mm_double_blocks_depth=2).to(device=d,dtype=dtype).eval()
shape=(1,4,1,2,3);cap=16
image=torch.randn(shape,device=d,dtype=dtype);ref=torch.randn_like(image);time=torch.ones(1,device=d,dtype=dtype)
emb=torch.randn(1,cap,64,device=d,dtype=dtype);mask=torch.ones(1,1,1,12+cap,device=d,dtype=torch.bool)
def run():return model(image,time,emb,ref_video_latent=ref,attention_mask=mask)[0]
report=[]
with torch.no_grad():
 for _ in range(3):run()
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):out=run()
 for n in (3,16,1,11,4):
  source=torch.randn(1,n,64,device=d,dtype=dtype)
  emb.fill_(1000);emb[:,:n].copy_(source);mask.fill_(False);mask[...,:12+n]=True
  expected=model(image,time,source,ref_video_latent=ref)[0]
  g.replay();torch.cuda.synchronize()
  diff=(out-expected).float();rel=float(diff.norm()/expected.float().norm());err=float(diff.abs().max())
  report.append({'tokens':n,'relativeL2':rel,'maxAbsError':err})
  assert torch.isfinite(out).all() and rel<.03,(n,rel,err)
 # Actual streaming runner: warm once, bind changing lengths, replay one graph.
 ids=model._get_token_frame_ids((8,2,3),d,temporal_ids=torch.arange(8,device=d))
 cos,sin=model.get_rotary_pos_embed_from_ids(frame_ids=ids,spatial_shape=(2,3))
 runner=StreamingGraphRunner(model,chunk_tokens=6,ref_tokens=0,latent_shape=shape,txt_len=cap,max_temporal_ids=8,pos_freqs=(cos.unsqueeze(0),sin.unsqueeze(0)),device=d,dtype=dtype,autocast_ctx=nullcontext,mask_prompt=True)
 runner.capture(timesteps=torch.tensor([1.],device=d),sigmas=torch.tensor([1.,0.],device=d))
 graph=runner.full_graph;addr=runner.in_prompt.data_ptr();maskaddr=runner.prompt_key_mask.data_ptr()
 for n in (3,16,1,11):
  runner.bind_session(10,torch.randn(1,n,64,device=d,dtype=dtype));runner.in_noise.copy_(image);runner.in_ref_latent.copy_(ref);graph.replay();torch.cuda.synchronize()
  assert runner.full_graph is graph and runner.in_prompt.data_ptr()==addr and runner.prompt_key_mask.data_ptr()==maskaddr
  assert torch.isfinite(runner.out_latents).all()
print(json.dumps({'fullTransformerPaddingComparisons':report,'streamingRunnerSingleGraphReused':True}),flush=True)
Path('tulip/fixed-prompt-gpu-proof-20260915.json').write_text(json.dumps(report,indent=2))
