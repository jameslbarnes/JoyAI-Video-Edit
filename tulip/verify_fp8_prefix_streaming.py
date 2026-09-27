"""Paired complete streaming graphs: denoise, clean-KV store, commit and publication."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
args=p.parse_args()
if args.output.exists(): raise RuntimeError('Evidence already exists')
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'deploy'))
os.environ.update(CUDA_VISIBLE_DEVICES='0',JOYOMNI_FP8_IMG='1',JOYOMNI_FP8_TXT='1',
                  JOYOMNI_FP8_FAST_ACCUM='1',JOYOMNI_SAGE_ATTN='1',
                  CUDA_HOME=str(root/'.toolchain/Library'),
                  TRITON_CACHE_DIR=str(root/'tulip/cache/fp8-prefix-proof'))
os.environ['PATH']=str(root/'.toolchain/Library/bin')+os.pathsep+os.environ['PATH']

def publisher():
    with urllib.request.urlopen('http://127.0.0.1:7873/health',timeout=10) as r:
        return json.load(r)['upscale']['gpu_video']
before=publisher()
def guard():
    after=publisher()
    if (after.get('error') or after['underruns']>before['underruns'] or
            after['dropped']>before['dropped'] or after['mux_pid']!=before['mux_pid']):
        raise RuntimeError('Publisher changed; stop only this probe')

import torch
from xvideo.models.dit.dit import Transformer3DModel
from xvideo.serving.graph_runner import StreamingGraphRunner
torch.set_num_threads(1);torch.manual_seed(735)
torch.cuda.set_per_process_memory_fraction(.035)
d=torch.device('cuda:0');dtype=torch.bfloat16
model=Transformer3DModel(patch_size=[1,1,1],in_channels=4,out_channels=4,
    hidden_size=128,heads_num=1,text_states_dim=64,mlp_width_ratio=2,
    mm_double_blocks_depth=3).to(device=d,dtype=dtype).eval()
shape=(1,4,1,2,3)
ids=model._get_token_frame_ids((8,2,3),d,temporal_ids=torch.arange(8,device=d))
cos,sin=model.get_rotary_pos_embed_from_ids(frame_ids=ids,spatial_shape=(2,3))
reports=[]
with torch.no_grad():
    for reference_tokens in (0,6):
        families=[]
        for trim in (False,True):
            os.environ['JOYOMNI_TRIM_PADDED_TEXT']='1' if trim else '0'
            family={}
            for capacity in (512,1024):
                runner=StreamingGraphRunner(model,chunk_tokens=6,ref_tokens=reference_tokens,
                    latent_shape=shape,txt_len=capacity,max_temporal_ids=8,
                    pos_freqs=(cos.unsqueeze(0),sin.unsqueeze(0)),device=d,dtype=dtype,
                    autocast_ctx=nullcontext,mask_prompt=True,shared_kv=family.get(512))
                runner.capture(timesteps=torch.tensor([1.,.5],device=d),sigmas=torch.tensor([1.,.5,0.],device=d))
                family[capacity]=runner
            families.append(family)
        # Both families start from identical history, including reference KV.
        for name in ('pool_k','pool_v'):
            base=getattr(families[0][512],name);base.normal_(std=.1)
            getattr(families[1][512],name).copy_(base)
        pointers={id(r):(r.in_prompt.data_ptr(),r.prompt_key_mask.data_ptr(),id(r.full_graph))
                  for family in families for r in family.values()}
        caches=[{},{}]
        for step,length in enumerate((1,497,512,513,517,895,1024,3,517,116,117)):
            guard()
            capacity=512 if length<=512 else 1024
            prompt=torch.randn(1,length,64,device=d,dtype=dtype)
            noise=torch.randn(shape,device=d,dtype=dtype)
            reference=torch.randn_like(noise)
            pair=[family[capacity] for family in families]
            for r in pair:
                r.bind_session(step,prompt);r.set_positions(step+2)
                r.in_noise.copy_(noise);r.in_ref_latent.copy_(reference)
                # Poison padding identically to detect masked-row leakage.
                r.in_prompt[:,length:].fill_(16)
                r.full_graph.replay()
            for i,r in enumerate(pair): r.publish_to_cache(caches[i],step)
            torch.cuda.synchronize()
            exact=torch.equal(pair[0].out_latents,pair[1].out_latents)
            finite=bool(torch.isfinite(pair[1].out_latents).all())
            buffers=('pool_k','pool_v','stage_k','stage_v','pub_k','pub_v')
            equal_buffers={name:torch.equal(getattr(pair[0],name),getattr(pair[1],name)) for name in buffers}
            for r in pair:
                assert pointers[id(r)]==(r.in_prompt.data_ptr(),r.prompt_key_mask.data_ptr(),id(r.full_graph))
            row={'referenceTokens':reference_tokens,'tokens':length,'capacity':capacity,
                 'activeRows':pair[1].active_text_rows,'latentsBitExact':exact,
                 'cacheBuffersBitExact':equal_buffers,'finite':finite}
            reports.append(row)
            args.output.write_text(json.dumps({'complete':False,'cases':reports},indent=2))
            print(json.dumps(row),flush=True)
            assert exact and finite and all(equal_buffers.values()),row
            time.sleep(.1)
guard()
result={'complete':True,'cases':reports,'sameGraphsAndAddresses':True,
        'peakAllocatedMiB':torch.cuda.max_memory_allocated()/2**20,
        'note':'Small three-block model; full graph flow and shared KV bucket switching, not full-model performance.'}
args.output.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
