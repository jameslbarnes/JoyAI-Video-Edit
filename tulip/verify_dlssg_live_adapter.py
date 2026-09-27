"""Production adapter proof without changing the live engine or renderer."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1,0'
from pathlib import Path
import sys,json,gc,time
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'deploy'))
import numpy as np,torch,av
from xvideo.serving.interpolation_output import DlssgOutput
from xvideo.serving.vsr_output import VsrUpscaler
torch.set_num_threads(2)
model=DlssgOutput('cuda:1');vsr=VsrUpscaler('cuda:1')
inputs=np.load(root/'tulip/evidence/dlssg-rife-20260923/inputs.npy',mmap_mode='r')
reports=[];held=[]
original_cpu=torch.Tensor.cpu
def forbidden(*a,**k):raise AssertionError('Intermediate CPU download')
try:
    for start,count in ((0,1),(1,8),(9,8),(17,8)):
        profile={};batch=[np.array(f,copy=True) for f in inputs[start:start+count]]
        torch.Tensor.cpu=forbidden
        try:
            frames,metas=model.process(batch,[{'seq':start+i} for i in range(count)],profile,raw_output=True,gpu_output=True)
            assert len(frames)==(1 if start==0 else count*2)
            # Retain more than the native surface-pool capacity, proving GPU
            # ownership prevents queued output from blocking future frames.
            held.extend(frames)
            for i,meta in enumerate(metas):
                if not meta['interpolated']:
                    expected=torch.from_numpy(batch[meta['seq']-start]).to('cuda:1')
                    assert torch.equal(frames[i].wait(),expected),'Original frame changed'
            first=vsr.process_gpu(frames[0],(1440,2592));first.ready.synchronize()
            assert bool(torch.isfinite(first.tensor).all()) and float(first.tensor.float().std())>1
        finally:torch.Tensor.cpu=original_cpu
        assert profile['dlssg']['download_bytes']==0
        reports.append(profile);print(json.dumps(profile),flush=True);time.sleep(.15)
    snapshot=held[0].wait().clone()
    model.reset();profile={}
    frames,metas=model.process([np.array(inputs[2],copy=True)],[{'seq':101}],profile,raw_output=True,gpu_output=True)
    assert len(frames)==1 and not metas[0]['interpolated']
    assert torch.equal(held[0].wait(),snapshot),'Reset changed queued output'
    # Queued frames own Torch memory even after the native session is closed.
    model.close();assert torch.equal(held[0].wait(),snapshot)
    result={'ok':True,'pyav':av.__version__,'retainedFrames':len(held),
            'checks':['native originals exact','GPU-only DLSSG to VSR','more than 16 retained outputs',
                      'reset starts a new history','retained outputs survive reset and close'],
            'reports':reports}
    (root/'tulip/evidence/dlssg-live-20260924/adapter-proof.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='reports'}),flush=True)
finally:
    torch.Tensor.cpu=original_cpu;held.clear();gc.collect();vsr.close();model.close()
