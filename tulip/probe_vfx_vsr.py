"""Isolated official VFX availability/timing probe; never changes live settings."""
import json,sys,time,statistics
from pathlib import Path
import numpy as np
from PIL import Image
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/"tulip/research/vfx-probe"))
from nvvfx import VideoSuperRes, __version__, get_sdk_version
from nvvfx.effects.video_super_res import QualityLevel
out=root/"tulip/evidence/shared-vae-20260922";results=[]
assert torch.cuda.mem_get_info()[0]>2*2**30,"Need 2GiB free"
array=np.asarray(Image.open(root/"tulip/source-frame.png").convert("RGB").resize((1152,648))).copy()
x=torch.from_numpy(array).permute(2,0,1).contiguous().cuda().float()/255
for mode in (QualityLevel.HIGH,QualityLevel.ULTRA,QualityLevel.HIGHBITRATE_ULTRA):
    try:
        tick=time.perf_counter()
        with VideoSuperRes(quality=mode,device=0) as sr:
            sr.output_width=2592;sr.output_height=1440;sr.load()
            samples=[]
            for i in range(16):
                started=time.perf_counter();y=torch.from_dlpack(sr.run(x).image).clone();torch.cuda.synchronize()
                if i>=4:samples.append((time.perf_counter()-started)*1000)
            assert y.shape==(3,1440,2592),y.shape
            assert torch.isfinite(y).all() and y.std()>.005,"Invalid or black output"
            Image.fromarray(y.clamp(0,1).mul(255).byte().permute(1,2,0).cpu().numpy()).save(out/("vsr-"+mode.name+".png"))
            r=dict(mode=mode.name,medianMs=statistics.median(samples),samplesMs=samples,elapsed=time.perf_counter()-tick,mean=float(y.mean()),std=float(y.std()))
    except Exception as e:r=dict(mode=mode.name,error=repr(e))
    results.append(r);print(json.dumps(r),flush=True)
report=dict(package=__version__,sdk=get_sdk_version(),results=results,note="Isolated process on physical GPU0 while JoyAI starts; source is shader screenshot, not native JoyAI output; timings are indicative only")
(out/"vsr-probe.json").write_text(json.dumps(report,indent=2,default=str))
