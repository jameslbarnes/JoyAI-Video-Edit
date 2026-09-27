"""Opt-in padded-row proof on a small transformer; never changes live settings."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output',type=Path)
args=parser.parse_args()
if args.output.exists(): raise RuntimeError('Evidence already exists')
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'deploy'))
os.environ.update(CUDA_VISIBLE_DEVICES='0',JOYOMNI_FP8_IMG='1',JOYOMNI_FP8_TXT='1',
                  JOYOMNI_FP8_FAST_ACCUM='1',JOYOMNI_SAGE_ATTN='1',
                  CUDA_HOME=str(root/'.toolchain/Library'),
                  TRITON_CACHE_DIR=str(root/'tulip/cache/fp8-prefix-proof'))
os.environ['PATH']=str(root/'.toolchain/Library/bin')+os.pathsep+os.environ['PATH']
import torch
from xvideo.models.dit.dit import Transformer3DModel
from xvideo.models.dit.prefix_attention import prepare_prefix_mask,set_prefix_length,protected_text_rows
torch.set_num_threads(1);torch.manual_seed(73)
torch.cuda.set_per_process_memory_fraction(.035)
d=torch.device('cuda:0');dtype=torch.bfloat16
stream=torch.cuda.Stream(device=d)
model=Transformer3DModel(patch_size=[1,1,1],in_channels=4,out_channels=4,
    hidden_size=128,heads_num=1,text_states_dim=64,mlp_width_ratio=2,
    mm_double_blocks_depth=3).to(device=d,dtype=dtype).eval()
shape=(1,4,1,2,3);capacity=1024;query_prefix=12
image=torch.randn(shape,device=d,dtype=dtype);ref=torch.randn_like(image)
timestep=torch.ones(1,device=d,dtype=dtype)
emb=torch.randn(1,capacity,64,device=d,dtype=dtype)
masks=[torch.ones(1,1,1,query_prefix+capacity,device=d,dtype=torch.bool) for _ in range(2)]
prepare_prefix_mask(masks[0],query_prefix+capacity)
prepare_prefix_mask(masks[1],query_prefix+capacity,text_capacity=capacity)
graphs=[];outputs=[];reports=[]
with torch.no_grad(),torch.cuda.stream(stream):
    for mask in masks:
        for _ in range(2): model(image,timestep,emb,ref_video_latent=ref,attention_mask=mask)
        stream.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream,capture_error_mode='thread_local'):
            output=model(image,timestep,emb,ref_video_latent=ref,attention_mask=mask)
        graphs.append(g);outputs.append(output)
    pointers=(emb.data_ptr(),masks[1].data_ptr(),masks[1]._fp8_text_rows.data_ptr())
    for length in (1,116,117,128,497,511,512,513,517,640,895,1024,3,517):
        emb.normal_()
        # Poison padded inputs to catch any accidental leakage; the halo must
        # preserve their contribution to the last real Q quantization block.
        emb[:,length:].mul_(16)
        image.normal_();ref.normal_()
        for mask in masks:
            mask.fill_(False);mask[...,:query_prefix+length]=True
            set_prefix_length(mask,query_prefix+length)
        for g in graphs:g.replay()
        stream.synchronize()
        actual_rows=protected_text_rows(query_prefix,length,capacity)
        exact_image=torch.equal(outputs[0][0],outputs[1][0])
        exact_text=torch.equal(outputs[0][1][:,:actual_rows],outputs[1][1][:,:actual_rows])
        finite=bool(torch.isfinite(outputs[1][0]).all() and torch.isfinite(outputs[1][1]).all())
        row={'tokens':length,'protectedRows':actual_rows,'imageBitExact':exact_image,
             'protectedTextBitExact':exact_text,'finite':finite}
        reports.append(row);print(json.dumps(row),flush=True)
        args.output.write_text(json.dumps({'cases':reports,'complete':False},indent=2))
        assert exact_image and exact_text and finite,row
        assert pointers==(emb.data_ptr(),masks[1].data_ptr(),masks[1]._fp8_text_rows.data_ptr())
        time.sleep(.08)
result={'cases':reports,'complete':True,'sameGraphReplayedAcrossLengths':True,
        'peakAllocatedMiB':torch.cuda.max_memory_allocated()/2**20,
        'note':'Three-block small transformer, same weights and arithmetic; not full-model/video qualification.'}
args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
