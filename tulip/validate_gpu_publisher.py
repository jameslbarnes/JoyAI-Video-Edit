"""Eight-second isolated publisher/scene handoff test; no model inference."""
import json,sys,time,subprocess
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deploy'))
from xvideo.serving.gpu_video_output import GpuVideoPublisher,GpuFrame
ffmpeg='C:/Users/Admin/src/easel/external/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe'
p=GpuVideoPublisher('cuda:0',ffmpeg,'rtmp://127.0.0.1:1935/joyai-gpu-validation')
record=None;rows=[];shader=np.zeros((360,648,3),np.uint8)
start=time.perf_counter();next_batch=0;phase=0
try:
    while time.perf_counter()-start<8:
        elapsed=time.perf_counter()-start
        shader=np.full((360,648,3),int(elapsed*30)%256,np.uint8)
        if elapsed>=2 and phase==0:
            assert p.begin_transition(lambda:shader,moving=True);phase=1
        if elapsed>=2.7 and phase==1:
            assert p.reset_scene();phase=2
        if elapsed>=next_batch and (elapsed<2.7 or elapsed>=4.2):
            with torch.cuda.device(0):
                pixels=torch.full((1440,2592,3),int(elapsed*25)%256,device='cuda:0',dtype=torch.uint8)
                frame=GpuFrame.record(pixels)
                assert p.submit([frame]*16,32)
            next_batch=elapsed+.5
        if p.packet_count>=32 and record is None:
            record=subprocess.Popen([ffmpeg,'-v','error','-y','-i',p.url,'-t','3','-c','copy','tulip/gpu-publisher-validation.flv'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        state=p.status();rows.append({'elapsed':elapsed,**state})
        if state['error']:raise RuntimeError(state['error'])
        time.sleep(.025)
finally:
    final=p.status();p.close()
    if record:
        try: _,err=record.communicate(timeout=10)
        except subprocess.TimeoutExpired:record.terminate();_,err=record.communicate(timeout=5)
        if record.returncode:raise RuntimeError(err.decode(errors='replace'))
Path('tulip/gpu-publisher-validation.json').write_text(json.dumps({'final':final,'rows':rows},indent=2))
print(json.dumps(final))
steady=next(r for r in rows if r['elapsed']>=4.5)
steady_fps=(rows[-1]['sent']-steady['sent'])/(rows[-1]['elapsed']-steady['elapsed'])
print(json.dumps({'steadyFps':steady_fps}))
assert not final['error'] and steady_fps>30.5 and final['dropped']==0
assert final['transitions_completed']==1 and final['transition_phase'] is None
assert p.thread is None and p.process is None
