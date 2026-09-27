"""Real-GPU interop/ownership verification, isolated from the live renderer."""
from pathlib import Path
import sys,json,time,gc
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tulip/research/dlssg-python'),str(ROOT/'deploy'),str(ROOT/'tulip')]
from dlssg_gpu_bridge import GpuDlssg,load_bridge
native,_,_=load_bridge()
runtime=native.initialize_bridge(0)
import numpy as np,cv2,torch
from xvideo.serving.vsr_output import VsrUpscaler
torch.set_num_threads(2);cv2.setNumThreads(1)
out=ROOT/'tulip/evidence/dlssg-gpu-20260923';out.mkdir(exist_ok=True)
frames=np.load(ROOT/'tulip/evidence/dlssg-rife-20260923/inputs.npy',mmap_mode='r')
model=GpuDlssg();vsr=VsrUpscaler('cuda:0')
report={'runtime':runtime,'torch':torch.__version__,'checks':[]}
try:
    rgba=cv2.cvtColor(frames[0],cv2.COLOR_RGB2RGBA)
    outputs,detail=model.process(rgba,reset=True)
    first=outputs[-1]
    assert torch.equal(first.wait().cpu(),torch.from_numpy(frames[0].copy()))
    saved=first.wait().clone()
    cpu_reference=vsr.process_gpu(frames[0],(1440,2592));cpu_reference.ready.synchronize()
    original_cpu=torch.Tensor.cpu
    original_download=native.DLSSGCudaSurface.to_host_rgb
    def forbidden(*a,**k):raise AssertionError('Unexpected raw CPU transfer in GPU path')
    torch.Tensor.cpu=forbidden;native.DLSSGCudaSurface.to_host_rgb=forbidden
    try:
        gpu_result=vsr.process_gpu(first,(1440,2592));gpu_result.ready.synchronize()
        assert torch.equal(gpu_result.wait(),cpu_reference.wait()),'Tensor/NumPy VSR mismatch'
        retained=gpu_result.wait().clone()
        samples=[]
        for i in range(1,12):
            outputs,detail=model.process(cv2.cvtColor(frames[i],cv2.COLOR_RGB2RGBA))
            assert len(outputs)==2,(i,detail)
            tick=time.perf_counter()
            for generated in outputs:
                latest=vsr.process_gpu(generated,(1440,2592));latest.ready.synchronize()
            samples.append((time.perf_counter()-tick)*1000)
            assert torch.equal(first.wait(),saved),'DLSSG overwrote retained input surface'
            assert torch.equal(gpu_result.wait(),retained),'VSR overwrote retained output'
            del outputs,generated,latest
            gc.collect();time.sleep(.2)
        report['checks']=['native original exact','VSR tensor/NumPy output exact','no raw CPU transfer','DLSSG retained surface immutable','VSR retained output immutable']
        report['twoOutputVsrMedianMs']=float(np.median(samples))
    finally:
        torch.Tensor.cpu=original_cpu;native.DLSSGCudaSurface.to_host_rgb=original_download
    try:
        vsr.process_gpu(torch.from_numpy(frames[0].copy()))
        raise AssertionError('Accepted CPU tensor')
    except ValueError:report['checks'].append('reject CPU tensor')
    assert int(first.wait().float().std())>1
    report['ok']=True
finally:
    # Release all exported surfaces before releasing the session.
    for name in ['outputs','first','saved','gpu_result','retained','cpu_reference']:
        globals().pop(name,None)
    gc.collect();torch.cuda.synchronize(0)
    vsr.close();model.close()
(out/'interop-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
