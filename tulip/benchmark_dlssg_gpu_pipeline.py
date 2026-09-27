"""Matched interpolation -> VSR -> NVENC, recording locally without a publisher."""
from pathlib import Path
import argparse,ctypes,gc,json,os,statistics,subprocess,sys,time,urllib.request
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tulip/evidence/dlssg-gpu-20260923';OUT.mkdir(exist_ok=True)
parser=argparse.ArgumentParser();parser.add_argument('method',choices=['rife','rife_gpu','dlssg']);args=parser.parse_args()
sys.path[:0]=[str(ROOT/'tulip/research/dlssg-python'),str(ROOT/'deploy'),str(ROOT/'tulip')]
from dlssg_gpu_bridge import GpuDlssg,load_bridge
native,_,_=load_bridge()
if args.method=='dlssg':runtime=native.initialize_bridge(0)
import numpy as np,cv2,torch,PyNvVideoCodec as nvc
from xvideo.serving.vsr_output import VsrUpscaler
from xvideo.serving.gpu_video_output import CudaRgba
torch.set_num_threads(2);cv2.setNumThreads(1)
def health():
    h=json.load(urllib.request.urlopen('http://127.0.0.1:7873/health',timeout=5));g=h['upscale']['gpu_video']
    return {'at':time.time(),'sent':g['sent'],'underruns':g['underruns'],'error':g['error']}
def memory():
    return subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip()
assert int(memory().splitlines()[0].split(',')[2])>2800
inputs=np.load(ROOT/'tulip/evidence/dlssg-rife-20260923/inputs.npy',mmap_mode='r')[:64]
report={'method':args.method,'torch':torch.__version__,'before':health(),'memoryBefore':memory(),'records':[]}
stream=torch.cuda.Stream(device=0,priority=0)
with torch.cuda.device(0),torch.cuda.stream(stream):
    if args.method=='dlssg':model=GpuDlssg();report['runtime']=runtime
    else:
        os.environ['JOYAI_RIFE_FLUX_ROOT']=str(ROOT.parent/'FluxRT')
        from xvideo.serving.rife_output import RifeOutput
        model=RifeOutput('cuda:0')
    vsr=VsrUpscaler('cuda:0')
    vsr.process_gpu(inputs[0],(1440,2592)).ready.synchronize()
    ctx=ctypes.c_void_p();ctypes.WinDLL('nvcuda.dll').cuCtxGetCurrent(ctypes.byref(ctx))
    enc=nvc.CreateEncoder(2592,1440,'ABGR',False,gpu_id=0,cudacontext=ctx.value,cudastream=stream.cuda_stream,
        codec='h264',preset='P4',tuning_info='ultra_low_latency',rc='cbr',bitrate=15000000,
        maxbitrate=15000000,vbvbufsize=20000000,fps=24,gop=24,idrperiod=24,bf=0,lookahead=0,
        colorspace='bt709',repeatspspps=1,max_res=[2592,1440],qmin=0)
    rgba=torch.empty((1440,2592,4),dtype=torch.uint8,device='cuda:0');rgba[:,:,3]=255
    wrapper=CudaRgba(rgba,stream)
    def interpolate(batch,start):
        if args.method.startswith('rife'):
            if start==0:model.reset()
            result=model.process(list(batch),[{'seq':start+i} for i in range(len(batch))],{},raw_output=True,gpu_output=args.method=='rife_gpu')[0]
            if args.method=='rife_gpu':stream.synchronize()
            return result
        result=[]
        for i,f in enumerate(batch):
            frames,detail=model.process(cv2.cvtColor(f,cv2.COLOR_RGB2RGBA),reset=start+i==0)
            expected=1 if start+i==0 else 2
            assert len(frames)==expected,(start+i,detail)
            result.extend(frames)
        return result
    def encode(pixels):
        rgba[:,:,:3].copy_(pixels.wait());stream.synchronize()
        result=enc.Encode(wrapper);stream.synchronize()
        return [p['data'] for p in result]
    for _ in range(2):
        generated=interpolate(inputs[:8],0)
        for f in generated:
            pixels=vsr.process_gpu(f,(1440,2592));encode(pixels)
        del generated,f,pixels;gc.collect();time.sleep(.5)
    report['memoryAfterWarmup']=memory()
    # Warmup wrote no file. A new encoder starts a valid output stream/keyframe.
    del enc;gc.collect()
    enc=nvc.CreateEncoder(2592,1440,'ABGR',False,gpu_id=0,cudacontext=ctx.value,cudastream=stream.cuda_stream,
        codec='h264',preset='P4',tuning_info='ultra_low_latency',rc='cbr',bitrate=15000000,
        maxbitrate=15000000,vbvbufsize=20000000,fps=24,gop=24,idrperiod=24,bf=0,lookahead=0,
        colorspace='bt709',repeatspspps=1,max_res=[2592,1440],qmin=0)
    original_cpu=torch.Tensor.cpu;original_host=native.DLSSGCudaSurface.to_host_rgb
    def forbidden(*a,**k):raise AssertionError('Raw-frame CPU download in GPU pipeline')
    if args.method!='rife':torch.Tensor.cpu=forbidden;native.DLSSGCudaSurface.to_host_rgb=forbidden
    file=OUT/(args.method+'-pipeline.h264')
    try:
        with file.open('wb') as dest:
            for repeat in range(2):
                for start in range(0,64,8):
                    t=time.perf_counter();generated=interpolate(inputs[start:start+8],start)
                    t1=time.perf_counter();sr_ms=0;enc_ms=0;packets=[]
                    for f in generated:
                        a=time.perf_counter();pixels=vsr.process_gpu(f,(1440,2592));pixels.ready.synchronize()
                        b=time.perf_counter();packets.extend(encode(pixels));c=time.perf_counter()
                        sr_ms+=(b-a)*1000;enc_ms+=(c-b)*1000
                    elapsed=(time.perf_counter()-t)*1000
                    row={'repeat':repeat,'start':start,'inputs':8,'outputs':len(generated),'interpolationMs':(t1-t)*1000,'vsrMs':sr_ms,'encodeMs':enc_ms,'totalMs':elapsed,'packets':len(packets),'bytes':sum(map(len,packets))}
                    report['records'].append(row)
                    for p in packets:dest.write(p)
                    del generated,f,pixels,packets;gc.collect();time.sleep(.6)
            tail=enc.EndEncode()
            report['flushPackets']=len(tail)
            for p in tail:dest.write(p['data'])
    finally:
        torch.Tensor.cpu=original_cpu;native.DLSSGCudaSurface.to_host_rgb=original_host
    report['after']=health();report['memoryAfter']=memory()
    steady=[r for r in report['records'] if r['start']>0]
    report['medianEightInputMs']=statistics.median(r['totalMs'] for r in steady)
    report['p95EightInputMs']=float(np.percentile([r['totalMs'] for r in steady],95))
    report['medianInterpolationMs']=statistics.median(r['interpolationMs'] for r in steady)
    report['medianVsrMs']=statistics.median(r['vsrMs'] for r in steady)
    report['medianEncodeMs']=statistics.median(r['encodeMs'] for r in steady)
    report['rawDownloadsInTimedPath']=0 if args.method!='rife' else 'existing RIFE CPU output'
    report['validOutputFrames']=sum(r['outputs'] for r in report['records'])
    report['encodedPackets']=sum(r['packets'] for r in report['records'])+report['flushPackets']
    (OUT/(args.method+'-pipeline.json')).write_text(json.dumps(report,indent=2))
    assert report['validOutputFrames']==report['encodedPackets'],(report['validOutputFrames'],report['encodedPackets'])
    vsr.close()
    if args.method=='dlssg':model.close()
    del enc,model,vsr;gc.collect()
ffmpeg=ROOT/'tulip/research/visual-enhancer-v11/app/bin/ffmpeg/bin/ffmpeg.exe'
subprocess.run([str(ffmpeg),'-hide_banner','-loglevel','error','-y','-r','24','-i',str(file),'-c:v','copy','-bsf:v','h264_metadata=video_full_range_flag=1:colour_primaries=1:transfer_characteristics=13:matrix_coefficients=1',str(file.with_suffix('.mp4'))],check=True)
print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
