"""Isolated recorded-frame comparison; never edits or connects a live publisher."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'tulip/research/visual-enhancer-v11/app'
OUT=ROOT/'tulip/evidence/dlssg-rife-20260923'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode',choices=['prepare','rife','dlssg','probe','summarize'])
parser.add_argument('--multiplier',type=int,choices=[2,3,4],default=2)
parser.add_argument('--limit',type=int,default=65)
args=parser.parse_args()
OUT.mkdir(exist_ok=True)
import numpy as np
import cv2
import av
cv2.setNumThreads(1)

def save(name,data):
    (OUT/(name+'.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')

def memory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    return [dict(zip(['gpu','usedMiB','freeMiB'],map(int,row.split(',')))) for row in raw.strip().splitlines()]

def live():
    d=json.load(urllib.request.urlopen('http://127.0.0.1:7873/health',timeout=5))
    g=d.get('upscale',{}).get('gpu_video',{})
    return {'at':time.time(),'runtime':d.get('runtime_loaded'),'sent':g.get('sent'),'underruns':g.get('underruns'),'error':g.get('error'),'geometry':g.get('geometry')}

def video(name,frames,rate=24):
    with av.open(str(OUT/(name+'.mp4')),'w') as c:
        enc=c.add_stream('libx264',rate=rate);enc.width=frames.shape[2];enc.height=frames.shape[1]
        enc.pix_fmt='yuv420p';enc.options={'crf':'16','preset':'fast','threads':'2'}
        for i,pixels in enumerate(frames):
            f=av.VideoFrame.from_ndarray(np.asarray(pixels),format='rgb24');f.pts=i
            for packet in enc.encode(f):c.mux(packet)
        for packet in enc.encode():c.mux(packet)

if args.mode=='prepare':
    source=ROOT/'tulip/evidence/quality-tuning-20260917-141403/deployment-153311/compiled-sage/1248x720/native.mp4'
    frames=[]
    with av.open(str(source)) as c:
        for i,f in enumerate(c.decode(video=0)):
            if i<160:continue
            a=f.to_ndarray(format='rgb24')
            frames.append(cv2.resize(a[9:711],(1152,648),interpolation=cv2.INTER_AREA))
            if len(frames)==129:break
    assert len(frames)==129
    reference=np.stack(frames);np.save(OUT/'reference.npy',reference)
    np.save(OUT/'inputs.npy',reference[::2])
    save('plan',{'source':str(source.relative_to(ROOT)),'sourceSha256':hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),
        'sourceFrames':[160,288],'sourceFps':16,'crop':[0,9,1248,702],'size':[1152,648],
        'inputCount':65,'referenceCount':129,'inputStride':2,'nominalInputFps':12,'nominalOutputFps':24,
        'note':'Archived native JoyAI frames, aspect-preserving crop and downsample. Every other recorded frame is withheld as a reference. Playback is retimed from source 16 fps to 24 fps; input corresponds to 8 fps in original media time. Same input bytes for both engines. Live GPU workload competes with timings.',
        'release':'v11.0','releaseSha256':'47a79f18665981ceab231bd9a5e2a3f79270c1dd22a156eb1bd41f893bbb60f7'})
    video('reference',reference)
    print('Prepared 65 matched input frames and 64 withheld reference frames.',flush=True)
    sys.exit()

if args.mode=='probe':
    sys.path.insert(0,str(APP))
    from src.frame_interpolation.native import DirectDLSSGSession,initialize_bridge
    from src.core.jobs import JobController
    assert memory()[0]['freeMiB']>2800,'Insufficient output GPU memory'
    data={'multiplier':args.multiplier,'runtime':initialize_bridge(0),'frames':[],'liveBefore':live()}
    inputs=np.load(OUT/'inputs.npy',mmap_mode='r')[:17]
    try:
        with DirectDLSSGSession(1152,648,args.multiplier-1,JobController(),0,timeout=30) as model:
            for i,frame in enumerate(inputs):
                generated,detail=model.process_frame(cv2.cvtColor(frame,cv2.COLOR_RGB2RGBA),force_reset=i==0,output_rgb=True,color_range=1)
                data['frames'].append({'input':i,'generated':len(generated),'detail':detail})
                for surface in generated:surface.close()
                time.sleep(.15)
            data['diagnostics']=model.diagnostics()
    except Exception as exc:data['error']=repr(exc)
    data['liveAfter']=live()
    save('ratio-probe-'+str(args.multiplier)+'x',data)
    print(json.dumps({'multiplier':args.multiplier,'generatedCounts':[f['generated'] for f in data['frames']],
        'disabled':[f['input'] for f in data['frames'] if f['detail']['interpolation_disabled']], 'error':data.get('error')}),flush=True)
    sys.exit()

if args.mode in ('rife','dlssg'):
    inputs=np.load(OUT/'inputs.npy',mmap_mode='r')[:args.limit]
    name=args.mode+'-'+str(args.multiplier)+'x'
    if (OUT/(name+'.json')).exists():raise RuntimeError('Result exists; preserve it')
    before=memory()
    assert before[0]['freeMiB']>2800,'Insufficient free memory on physical GPU 0'
    report={'method':args.mode,'multiplier':args.multiplier,'physicalGpu':0,'memoryBefore':before,'liveBefore':live(),'records':[]}
    output=np.lib.format.open_memmap(OUT/(name+'.npy'),mode='w+',dtype='uint8',shape=((len(inputs)-1)*args.multiplier+1,648,1152,3))
    output[0]=inputs[0];index=1
    started=time.perf_counter()
    if args.mode=='rife':
        assert args.multiplier==2
        os.environ['JOYAI_RIFE_FLUX_ROOT']=str(ROOT.parent/'FluxRT')
        sys.path.insert(0,str(ROOT/'deploy'))
        import torch
        from xvideo.serving.rife_output import RifeOutput
        torch.set_num_threads(2)
        stream=torch.cuda.Stream(device=0,priority=0)
        with torch.cuda.device(0),torch.cuda.stream(stream):
            model=RifeOutput('cuda:0')
            for _ in range(2):
                model.reset();model.process(list(inputs[:8]),[{'seq':i} for i in range(8)],{},raw_output=True)
                stream.synchronize();time.sleep(.4)
            report['setupSeconds']=time.perf_counter()-started
            report['memoryAfterWarmup']=memory()
            model.reset();index=0
            for start in range(0,len(inputs),8):
                batch=list(inputs[start:start+8]);profile={}
                tick=time.perf_counter()
                pixels,meta=model.process(batch,[{'seq':start+i} for i in range(len(batch))],profile,raw_output=True)
                ms=(time.perf_counter()-tick)*1000
                output[index:index+len(pixels)]=pixels;index+=len(pixels)
                report['records'].append({'start':start,'inputs':len(batch),'outputs':len(pixels),'hostOutputMs':ms,'profile':profile})
                time.sleep(.6)
            report['torchPeakAllocatedMiB']=torch.cuda.max_memory_allocated()/2**20
            report['torchPeakReservedMiB']=torch.cuda.max_memory_reserved()/2**20
    else:
        sys.path.insert(0,str(APP))
        from src.frame_interpolation.native import DirectDLSSGSession,initialize_bridge
        from src.core.jobs import JobController
        report['runtime']=initialize_bridge(0)
        controller=JobController()
        with DirectDLSSGSession(1152,648,args.multiplier-1,controller,0,timeout=30) as model:
            rgba=[cv2.cvtColor(a,cv2.COLOR_RGB2RGBA) for a in inputs]
            for i in range(min(8,len(rgba))):
                generated,detail=model.process_frame(rgba[i],force_reset=i==0,output_rgb=True,color_range=1)
                for surface in generated:surface.close()
                time.sleep(.12)
            report['setupSeconds']=time.perf_counter()-started
            report['memoryAfterWarmup']=memory()
            for i,frame in enumerate(rgba):
                tick=time.perf_counter()
                generated,detail=model.process_frame(frame,force_reset=i==0,output_rgb=True,color_range=1)
                api_ms=(time.perf_counter()-tick)*1000
                arrays=[]
                try:
                    for surface in generated:arrays.append(surface.to_host_rgb()[...,:3].copy())
                finally:
                    for surface in generated:surface.close()
                total_ms=(time.perf_counter()-tick)*1000
                expected=0 if i==0 else args.multiplier-1
                if len(arrays)!=expected:raise RuntimeError(f'Frame {i}: expected {expected} generated, got {len(arrays)}; {detail}')
                if i:
                    for a in arrays:output[index]=a;index+=1
                    output[index]=inputs[i];index+=1
                report['records'].append({'input':i,'generated':len(arrays),'gpuSurfaceReadyMs':api_ms,'hostOutputMs':total_ms,'native':detail})
                time.sleep(.12)
            report['diagnostics']=model.diagnostics()
    assert index==len(output),(index,len(output))
    output.flush();report['memoryAfter']=memory();report['liveAfter']=live()
    report['completed']=True;report['outputCount']=len(output)
    rows=[r for r in report['records'][1:] if r.get('inputs',8)==8]
    report['hostOutputMedianMs']=statistics.median(r['hostOutputMs'] for r in rows)
    report['hostOutputP95Ms']=float(np.percentile([r['hostOutputMs'] for r in rows],95))
    report['inputFramesPerProcessingSecond']=sum(r.get('inputs',1) for r in rows)/(sum(r['hostOutputMs'] for r in rows)/1000)
    save(name,report)
    video(name,output,12*args.multiplier)
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','diagnostics']}),flush=True)
    sys.exit()

if args.mode=='summarize':
    from PIL import Image,ImageDraw
    truth=np.load(OUT/'reference.npy',mmap_mode='r')
    arrays={m:np.load(OUT/(m+'.npy'),mmap_mode='r') for m in ['rife-2x','dlssg-2x']}
    def score(a,b):
        af=a.astype(np.float32);bf=b.astype(np.float32)
        mse=float(np.mean((af-bf)**2));psnr=10*np.log10(255**2/max(mse,1e-9))
        x=cv2.cvtColor(af,cv2.COLOR_RGB2GRAY);y=cv2.cvtColor(bf,cv2.COLOR_RGB2GRAY)
        blur=lambda z:cv2.GaussianBlur(z,(11,11),1.5)
        mx,my=blur(x),blur(y);vx,vy=blur(x*x)-mx*mx,blur(y*y)-my*my
        xy=blur(x*y)-mx*my
        s=((2*mx*my+6.5025)*(2*xy+58.5225))/((mx*mx+my*my+6.5025)*(vx+vy+58.5225))
        return {'psnr':float(psnr),'ssim':float(s[5:-5,5:-5].mean()),'mae':float(np.mean(np.abs(af-bf)))}
    result={}
    for name,frames in arrays.items():
        assert np.array_equal(frames[::2],truth[::2]),'Original frames changed'
        rows=[{'frame':i,**score(frames[i],truth[i])} for i in range(1,len(truth),2)]
        result[name]={'meanPsnr':statistics.mean(x['psnr'] for x in rows),'meanSsim':statistics.mean(x['ssim'] for x in rows),'meanMae':statistics.mean(x['mae'] for x in rows),'frames':rows}
    save('quality',result)
    frames=[]
    for i in range(len(truth)):
        row=np.concatenate([truth[i],arrays['rife-2x'][i],arrays['dlssg-2x'][i]],axis=1)
        row=cv2.resize(row,(1728,324),interpolation=cv2.INTER_AREA)
        banner=np.zeros((36,1728,3),dtype='uint8')
        for x,label in zip([12,588,1164],['WITHHELD REFERENCE','RIFE 2x','DLSSG 2x']):cv2.putText(banner,label,(x,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
        frames.append(np.concatenate([banner,row],axis=0))
    video('comparison',np.stack(frames))
    worst=sorted(range(1,len(truth),2),key=lambda i:next(x['mae'] for x in result['dlssg-2x']['frames'] if x['frame']==i),reverse=True)[:3]
    for i in sorted(set([33,65,97]+worst)):
        Image.fromarray(frames[i]).save(OUT/f'comparison-frame-{i}.png')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='frames'} for k,v in result.items()}),flush=True)
