"""Small GPU-memory encode probe; writes only a local diagnostic bitstream."""
import ctypes,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
import PyNvVideoCodec as nvc

torch.cuda.set_device(0)
stream=torch.cuda.Stream()
class Frame:
    def __init__(self,t): self.t=t
    def cuda(self):
        h=self.t.shape[0]*2//3;w=self.t.shape[1]
        return [SimpleNamespace(__cuda_array_interface__={'shape':shape,'strides':strides,'typestr':'|u1','data':(ptr,False),'version':3,'stream':stream.cuda_stream}) for shape,strides,ptr in [((h,w,1),(w,1,1),self.t.data_ptr()),((h//2,w//2,2),(w,2,1),self.t.data_ptr()+h*w)]]
with torch.cuda.stream(stream):
    t=torch.full((540,648),128,dtype=torch.uint8,device='cuda:0');t[:360]=100
    stream.synchronize()
    ctx=ctypes.c_void_p();assert ctypes.WinDLL('nvcuda.dll').cuCtxGetCurrent(ctypes.byref(ctx))==0
    encoder=nvc.CreateEncoder(648,360,'NV12',False,gpu_id=0,cudacontext=ctx.value,cudastream=stream.cuda_stream,codec='h264',preset='P4',tuning_info='ultra_low_latency',rc='cbr',bitrate='15000000',fps=32,gop=32,bf=0)
    frame=Frame(t);packets=[];times=[]
    for i in range(16):
        tick=time.perf_counter();packets.append(encoder.Encode(frame));times.append((time.perf_counter()-tick)*1000)
        time.sleep(.032)
    packets.append(encoder.EndEncode())
    first=next(p for batch in packets for p in batch)
    print('packet',type(first),[(k,type(v),len(v) if hasattr(v,'__len__') else v) for k,v in first.items()] if isinstance(first,dict) else repr(first)[:500], 'batch counts',[len(p) for p in packets], 'methods',dir(encoder))
    packets=[p['data'] for batch in packets for p in batch]
    Path('tulip/gpu-encoder-probe.h264').write_bytes(b''.join(packets))
    print(json.dumps({'version':nvc.__version__,'bytes':[len(p) for p in packets],'ms':times,'config':str(encoder.GetEncodeReconfigureParams())}))
