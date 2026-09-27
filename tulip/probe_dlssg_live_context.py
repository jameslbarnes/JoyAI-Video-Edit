"""Read-only-to-renderer bridge probe using production GPU order and installed PyAV."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1,0'
from pathlib import Path
import sys,json,gc,time
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'deploy'),str(root/'tulip')]
import torch,numpy as np,av
from dlssg_gpu_bridge import GpuDlssg,load_bridge
native,_,_=load_bridge()
device=1
# Torch has already retained the CUDA primary context. This RGB-only integration
# does not use FFmpeg AVFrames or require a second FFmpeg-created device context.
def ensure_primary(ordinal):
    with torch.cuda.device(ordinal):
        torch.cuda.init()
        torch.empty(1,device=f'cuda:{ordinal}')
native._MANAGER._ensure_ffmpeg_device=ensure_primary
print(json.dumps({'av':av.__version__,'visible':os.environ['CUDA_VISIBLE_DEVICES']}),flush=True)
model=GpuDlssg(device=device)
print(json.dumps(model.status),flush=True)
inputs=np.load(root/'tulip/evidence/dlssg-rife-20260923/inputs.npy',mmap_mode='r')
held=[]
for i in range(5):
    rgba=np.empty((*inputs[i].shape[:2],4),dtype=np.uint8);rgba[:,:,:3]=inputs[i];rgba[:,:,3]=255
    frames,detail=model.process(rgba,reset=i==0)
    for f in frames: f.ready.synchronize()
    assert torch.equal(frames[-1].wait().cpu(),torch.from_numpy(np.array(inputs[i],copy=True)))
    print(json.dumps({'i':i,'count':len(frames),'device':str(frames[-1].tensor.device),'detail':detail}),flush=True)
    held.append(frames[-1]);time.sleep(.2)
del frames,f,held;gc.collect();torch.cuda.synchronize(device);model.close()
print('SUCCESS',flush=True)
