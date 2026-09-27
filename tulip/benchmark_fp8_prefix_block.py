"""Matched real-weight DiT block at 648p geometry, with low-duty live-safe pacing."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('evidence',type=Path)
parser.add_argument('--repeats',type=int,default=31)
parser.add_argument('--focus',action='store_true')
args=parser.parse_args()
if not 7<=args.repeats<=61:parser.error('repeats must be 7..61')
args.evidence.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'deploy'))
os.environ.update(CUDA_VISIBLE_DEVICES='0',JOYOMNI_FP8_IMG='1',JOYOMNI_FP8_TXT='1',
                  JOYOMNI_FP8_FAST_ACCUM='1',JOYOMNI_SAGE_ATTN='1',
                  CUDA_HOME=str(root/'.toolchain/Library'),
                  TRITON_CACHE_DIR=str(root/'tulip/cache/fp8-prefix-proof'))
os.environ['PATH']=str(root/'.toolchain/Library/bin')+os.pathsep+os.environ['PATH']
def get(endpoint):
    with urllib.request.urlopen('http://127.0.0.1:7873/'+endpoint,timeout=5) as r:return json.load(r)
before=get('health');base_gpu=before['upscale']['gpu_video']
(args.evidence/'before-health.json').write_text(json.dumps(before,indent=2))
free=float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
if free<4500: raise RuntimeError('Need 4.5 GiB free on output GPU for this isolated block probe')
def guard():
    gpu=get('health')['upscale']['gpu_video']
    if (gpu.get('error') or gpu['underruns']>base_gpu['underruns'] or gpu['dropped']>base_gpu['dropped']
            or gpu['mux_pid']!=base_gpu['mux_pid']):
        raise RuntimeError('Live publisher changed/regressed; stop only this probe')

import torch
from xvideo.models.dit.dit import MMDoubleStreamBlock,_maybe_install_fp8_stream
from xvideo.models.dit.prefix_attention import prepare_prefix_mask,set_prefix_length,protected_text_rows
torch.set_num_threads(1);torch.manual_seed(911)
torch.cuda.set_per_process_memory_fraction(.10)
d=torch.device('cuda:0');dtype=torch.bfloat16;stream=torch.cuda.Stream(device=d)
checkpoint=root/'tulip/models/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth'
state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
prefix='double_blocks.0.'
with torch.device('meta'):block=MMDoubleStreamBlock(4096,32,4)
block.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},assign=True)
block=block.to(device=d,dtype=dtype).eval()
del state
_maybe_install_fp8_stream(block,'img');_maybe_install_fp8_stream(block,'txt')
print(json.dumps({'loadedRealBlock':0,'allocatedMiB':torch.cuda.memory_allocated()/2**20}),flush=True)

ct=27*48;query_prefix=2*ct;key_prefix=4*ct
image=torch.randn(1,query_prefix,4096,device=d,dtype=dtype)
vec=torch.randn(1,6,4096,device=d,dtype=dtype)*.1
cached_key=torch.randn(1,2*ct,32,128,device=d,dtype=dtype)
cached_value=torch.randn_like(cached_key)
cos=torch.ones(1,query_prefix,128,device=d,dtype=dtype)
sin=torch.zeros_like(cos)
flush=torch.empty(256*1024*1024,device=d,dtype=torch.uint8)
def kv(*args,**kwargs):return cached_key,cached_value
reports=[]
def run(mask):
    return block(image,text,vec,vis_freqs_cis=(cos,sin),kv_cache_assembler=kv,
                 layer_idx=0,kv_cache_pre_rope=True,attention_mask=mask)
def time_graph(g):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    flush.zero_();a.record(stream);g.replay();b.record(stream);b.synchronize()
    return a.elapsed_time(b)
try:
    with torch.no_grad(),torch.cuda.stream(stream):
        for capacity,lengths in [(512,(1,350,497,512)),(1024,(513,517,640,895,1024,517))]:
            if args.focus:lengths=(497,) if capacity==512 else (517,1024)
            guard()
            text=torch.randn(1,capacity,4096,device=d,dtype=dtype)
            masks=[torch.ones(1,1,1,key_prefix+capacity,device=d,dtype=torch.bool) for _ in range(2)]
            prepare_prefix_mask(masks[0],query_prefix+capacity)
            prepare_prefix_mask(masks[1],query_prefix+capacity,text_capacity=capacity)
            graphs=[];outputs=[]
            for mask in masks:
                run(mask);stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream,capture_error_mode='thread_local'):
                    output=run(mask)
                graphs.append(graph);outputs.append(output)
            for length in lengths:
                guard();text.normal_();text[:,length:].mul_(4)
                for mask in masks:
                    mask.fill_(False);mask[...,:key_prefix+length]=True
                    set_prefix_length(mask,key_prefix+length)
                for g in graphs:g.replay()
                stream.synchronize()
                active=protected_text_rows(query_prefix,length,capacity)
                exact_image=torch.equal(outputs[0][0],outputs[1][0])
                exact_text=torch.equal(outputs[0][1][:,:active],outputs[1][1][:,:active])
                assert exact_image and exact_text,(capacity,length,exact_image,exact_text)
                assert torch.isfinite(outputs[1][0]).all() and torch.isfinite(outputs[1][1]).all()
                base_ms=[];prefix_ms=[]
                for i in range(args.repeats):
                    if i%2:
                        prefix_ms.append(time_graph(graphs[1]));base_ms.append(time_graph(graphs[0]))
                    else:
                        base_ms.append(time_graph(graphs[0]));prefix_ms.append(time_graph(graphs[1]))
                    time.sleep(.14)
                row={'capacity':capacity,'tokens':length,'protectedRows':active,'imageBitExact':exact_image,
                     'protectedTextBitExact':exact_text,'baselineMs':base_ms,'prefixMs':prefix_ms,
                     'baselineMedianMs':statistics.median(base_ms),'prefixMedianMs':statistics.median(prefix_ms),
                     'pairedRatioMedian':statistics.median(p/b for p,b in zip(prefix_ms,base_ms))}
                reports.append(row)
                (args.evidence/'results.json').write_text(json.dumps(reports,indent=2))
                print(json.dumps({k:v for k,v in row.items() if k not in ('baselineMs','prefixMs')}),flush=True)
            # Drop both captured pools before testing the other capacity.
            del graph,output,g,graphs,outputs,masks,mask,text
            gc.collect();stream.synchronize();torch.cuda.empty_cache();time.sleep(.2)
    guard()
finally:
    result={'reports':reports,'physicalGpu':0,'realWeightBlock':0,'nativeGeometry':[1152,648],
            'peakAllocatedMiB':torch.cuda.max_memory_allocated()/2**20,
            'peakReservedMiB':torch.cuda.max_memory_reserved()/2**20,
            'note':'One trained block, synthetic activations, cold-cache graph timings during live rendering; not a full-model speed claim.'}
    (args.evidence/'summary.json').write_text(json.dumps(result,indent=2))
    try:(args.evidence/'after-health.json').write_text(json.dumps(get('health'),indent=2))
    except Exception:pass
