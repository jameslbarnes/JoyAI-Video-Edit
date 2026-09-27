"""Bounded same-input SR CUDA graph benchmark; no service configuration writes."""
import json, os, sys, time, statistics
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/"deploy"))
from xvideo.serving.upscale_output import PixelUpscaler
assert torch.cuda.mem_get_info(0)[0] > 2*2**30, "Need 2 GiB free for isolated SR probe"
torch.set_num_threads(2)
u=PixelUpscaler("cuda:0",.25)
shape=(1,3,648,1152);size=(1440,2592)
def render(x):
    r=u.model(x).clamp_(0,1)
    r=F.interpolate(r.float(),size=size,mode="bicubic",align_corners=False,antialias=True)
    return r.clamp_(0,1).mul_(255).byte()[0].permute(1,2,0).contiguous()
with torch.inference_mode(),torch.cuda.stream(u.stream):
    x=torch.rand(shape,device="cuda:0",dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    for _ in range(3): y=render(x)
    u.stream.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=u.stream,capture_error_mode="thread_local"):
        captured=render(x)
    reports=[]
    for repeat in range(3):
        for mode in ("eager","graph") if repeat%2==0 else ("graph","eager"):
            times=[]
            for _ in range(12):
                tick=time.perf_counter()
                if mode=="eager": y=render(x)
                else: g.replay();y=captured.clone()
                u.stream.synchronize();times.append((time.perf_counter()-tick)*1000)
            reports.append(dict(mode=mode,medianMs=statistics.median(times),samplesMs=times))
    accuracy=[]
    for seed in (11,22,33):
        torch.manual_seed(seed);x.copy_(torch.rand_like(x));expected=render(x)
        g.replay();actual=captured.clone();u.stream.synchronize()
        delta=(actual.float()-expected.float()).abs()
        accuracy.append(dict(maxError=delta.max().item(),meanError=delta.mean().item()))
report=dict(reports=reports,accuracy=accuracy,peakMiB=torch.cuda.max_memory_allocated()/2**20,
    note="Probe shares output GPU with live feed; paired timing is indicative, not isolated throughput")
(root/"tulip/evidence/shared-vae-20260922/sr-graph-probe.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
