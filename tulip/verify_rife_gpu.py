"""Check GPU-only RIFE output against the existing CPU path and retained frames."""
from pathlib import Path
import os,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'deploy'))
os.environ['JOYAI_RIFE_FLUX_ROOT']=str(ROOT.parent/'FluxRT')
import numpy as np,torch
from xvideo.serving.rife_output import RifeOutput
torch.set_num_threads(2)
inputs=np.load(ROOT/'tulip/evidence/dlssg-rife-20260923/inputs.npy',mmap_mode='r')
model=RifeOutput('cuda:0');checked=[]
# Pin convolution selection for an exact transport comparison. The existing
# recursive morph can vary slightly between repeated CPU-output runs when
# FluxRT's benchmark/autotuning mode is left on; this is not a transport change.
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
for mult in [2,4]:
    model.reset();expected,meta=model.process(list(inputs[:8]),[{'seq':i} for i in range(8)],{},raw_output=True,multiplier=mult)
    model.reset();original_cpu=torch.Tensor.cpu
    def forbidden(*a,**k):raise AssertionError('Unexpected RIFE GPU output download')
    torch.Tensor.cpu=forbidden
    try:
        actual,actual_meta=model.process(list(inputs[:8]),[{'seq':i} for i in range(8)],{},raw_output=True,multiplier=mult,gpu_output=True)
        saved=[x.wait().clone() for x in actual]
        next_frames,_=model.process(list(inputs[8:16]),[{'seq':i+8} for i in range(8)],{},raw_output=True,multiplier=mult,gpu_output=True)
        torch.cuda.synchronize(0)
        assert all(torch.equal(x.wait(),copy) for x,copy in zip(actual,saved)),'Retained output overwritten'
    finally:torch.Tensor.cpu=original_cpu
    assert actual_meta==meta
    assert len(expected)==len(actual)
    assert all(np.array_equal(x,y.wait().cpu().numpy()) for x,y in zip(expected,actual)),'GPU path changed pixels'
    checked.append({'multiplier':mult,'frames':len(actual),'exactPixels':True,'retainedOwnership':True,'noRawDownload':True})
    del actual,next_frames,saved,expected;torch.cuda.empty_cache()
source=inputs[16].copy();metas=[{'seq':1}]
model.reset(transition=True)
expected,expected_meta=model.process([inputs[0]],metas,{},raw_output=True,transition_source=source)
model.reset(transition=True);original_cpu=torch.Tensor.cpu;torch.Tensor.cpu=forbidden
try:
    actual,actual_meta=model.process([inputs[0]],metas,{},raw_output=True,transition_source=source,gpu_output=True)
    torch.cuda.synchronize(0)
finally:torch.Tensor.cpu=original_cpu
assert len(actual)==16 and actual_meta==expected_meta
assert all(np.array_equal(x,y.wait().cpu().numpy()) for x,y in zip(expected,actual))
checked.append({'sceneMorph':True,'frames':16,'exactPixels':True,'noRawDownload':True})
report={'ok':True,'cases':checked}
(ROOT/'tulip/evidence/dlssg-gpu-20260923/rife-interop-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
