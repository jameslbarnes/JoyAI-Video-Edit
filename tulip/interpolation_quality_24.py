"""Reproducible held-out 12->24 fps evaluation. Never changes live settings.

CPU readbacks in this OFFLINE quality tool are intentional and not timed as
production performance. See the run manifest for sources and metric caveats.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time
import av, cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'deploy'), str(ROOT / 'tulip')]
cv2.setNumThreads(1)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def encode(path, frames):
    with av.open(str(path), 'w') as c:
        s = c.add_stream('libx264', rate=24)
        s.width, s.height = frames[0].shape[1], frames[0].shape[0]
        s.pix_fmt = 'yuv420p'; s.options = {'crf':'16', 'preset':'fast', 'threads':'2'}
        for i, x in enumerate(frames):
            f = av.VideoFrame.from_ndarray(np.asarray(x), format='rgb24'); f.pts = i
            for packet in s.encode(f): c.mux(packet)
        for packet in s.encode(): c.mux(packet)

def prepare(out):
    assert not (out / 'plan.json').exists(), 'Refusing to replace reference manifest'
    source = out / 'sintel-trailer.ogv'
    starts = {'snow':120, 'dialogue':276, 'desert':816}
    refs = {n:[] for n in starts}; pts = {n:[] for n in starts}
    with av.open(str(source)) as c:
        for i, f in enumerate(c.decode(video=0)):
            for name, start in starts.items():
                if start <= i < start + 73:
                    # Preserve composition, including letterboxing; never enlarge.
                    refs[name].append(cv2.resize(f.to_ndarray(format='rgb24'), (1152,648), interpolation=cv2.INTER_AREA))
                    pts[name].append(float(f.pts * f.time_base))
    cases = []
    for name in starts:
        assert len(refs[name]) == 73 and np.allclose(np.diff(pts[name]), 1/24, atol=1e-8)
        path = out / (name + '-reference.npy'); np.save(path, np.stack(refs[name]))
        cases.append({'name':name, 'reference':str(path), 'sha256':sha(path),
                      'source':'Sintel trailer, Blender Foundation / Durian Open Movie Team',
                      'decodedStartIndex':starts[name], 'sourcePtsSeconds':pts[name],
                      'description':'True consecutive 24 fps timestamps, downsampled 1920x1080 to 1152x648; dialogue includes a cut.'})
    # Analytic, exact-time procedural motion: no interpolation used in reference.
    y, x = np.mgrid[:648,:1152].astype(np.float32); x=(x-576)/324; y=(y-324)/324
    frames = []
    for i in range(73):
        t=i/24; u=x+.23*np.sin(y*2+t*.9); v=y+.18*np.cos(x*2.3-t*.7)
        r=np.sqrt((u-.25*np.sin(t))**2+(v-.2*np.cos(t))**2)
        z=np.sin(19*r-2.5*t)+.45*np.sin(9*u+7*v+t)+.2*np.cos(35*u-20*v-1.3*t)
        rgb=np.stack([.5+.48*np.sin(z+q+t*.13) for q in [0,2.1,4.2]],axis=-1)
        frames.append(np.round(rgb*255).clip(0,255).astype(np.uint8))
    path=out/'shader-reference.npy';np.save(path,np.stack(frames))
    cases.append({'name':'shader','reference':str(path),'sha256':sha(path),'description':'Analytic deformation/color shader sampled at exact i/24 seconds.'})
    old=ROOT/'tulip/evidence/dlssg-gpu-20260923/quality/occlusion-reference.npy'
    path=out/'occlusion-reference.npy';np.save(path,np.load(old,mmap_mode='r')[:73])
    cases.append({'name':'occlusion','reference':str(path),'sha256':sha(path),'description':'Existing exact-time 24 fps synthetic: rotating texture, thin lines, independent occluders; first 73 frames.'})
    save(out/'plan.json',{'inputFps':12,'outputFps':24,'framesPerCase':73,'inputStride':2,'cases':cases,
        'sourceUrl':'https://upload.wikimedia.org/wikipedia/commons/0/06/Sintel_trailer-1080p.ogv?download=1',
        'sourceSha256':sha(source),'attribution':'Sintel (c) Blender Foundation | www.sintel.org; CC BY 3.0 https://creativecommons.org/licenses/by/3.0/. Excerpts resized, alternate frames replaced for research comparison.',
        'metricSource':'https://github.com/conalld/psnr-div/tree/601bbff9681509ce5671d50782afae6f0ea03855',
        'metricNotes':'Official source imported by filename because package imports a missing visualization module. Its divergence uses cross derivatives (du/dy + dv/dx). Report unchanged official score AND axis-corrected sensitivity score (du/dx + dv/dy), threshold .01, Farneback CPU. Not a perceptual oracle.',
        'scope':'24 fps references are film and synthetic, not a 24 fps JoyAI capture. Offline copies are excluded from production speed claims.'})
    print('Prepared five genuine 24 fps cases / 180 withheld midpoints.',flush=True)

def run(out, method):
    import torch
    torch.set_num_threads(2)
    os.environ['JOYAI_RIFE_FLUX_ROOT']=str(ROOT.parent/'FluxRT')
    from xvideo.serving.interpolation_output import DlssgOutput
    from xvideo.serving.rife_output import RifeOutput
    model=(DlssgOutput if method=='dlssg' else RifeOutput)('cuda:0')
    plan=json.loads((out/'plan.json').read_text())
    def inference(truth):
        model.reset(); result=[]; profiles=[]
        for start in range(0,len(truth[::2]),8):
            batch=list(truth[::2][start:start+8]);profile={}
            frames,metas=model.process(batch,[{'seq':start+i} for i in range(len(batch))],profile,raw_output=True,gpu_output=True)
            result.extend(f.wait().cpu().numpy().copy() for f in frames)
            profiles.append(profile);time.sleep(.15)
        result=np.stack(result)
        assert result.shape==truth.shape
        assert np.array_equal(result[::2],truth[::2]),'Native originals must remain byte-identical'
        return result,profiles
    try:
        for ci,case in enumerate(plan['cases']):
            dest=out/(case['name']+'-'+method+'.npy');assert not dest.exists()
            truth=np.load(case['reference'],mmap_mode='r')
            frames,profiles=inference(truth);np.save(dest,frames)
            report={'completed':True,'frames':len(frames),'originalsExact':True,'profiles':profiles,'sha256':sha(dest)}
            if ci==0:
                repeated,_=inference(truth)
                report['repeat']={'byteExact':bool(np.array_equal(frames,repeated)),
                    'maxAbsPixelDifference':int(np.abs(frames.astype(np.int16)-repeated.astype(np.int16)).max()),
                    'meanAbsPixelDifference':float(np.abs(frames.astype(np.float32)-repeated).mean())}
            save(out/(case['name']+'-'+method+'.json'),report)
            print(json.dumps({'case':case['name'],'method':method,'frames':len(frames),'repeat':report.get('repeat'),'cadenceHolds':sum(p.get('dlssg',{}).get('cadence_holds',0) for p in profiles)}),flush=True)
    finally:
        if hasattr(model,'close'):model.close()

def module_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def score(out):
    metric=module_file('official_div',out/'psnr-div-source/psnr_div/metric.py')
    flowlib=module_file('official_flow',out/'psnr-div-source/psnr_div/flow.py')
    # Metric sanity: exact reference has zero error; wrong moving content must differ.
    summary={}
    for case in json.loads((out/'plan.json').read_text())['cases']:
        name=case['name'];truth=np.load(case['reference'],mmap_mode='r')
        variants={m:np.load(out/(name+'-'+m+'.npy'),mmap_mode='r') for m in ['rife','dlssg']}
        held=np.array(truth);held[1::2]=truth[:-1:2];variants['hold']=held
        blend=np.array(truth);blend[1::2]=((truth[:-1:2].astype(np.uint16)+truth[2::2])//2).astype(np.uint8);variants['blend']=blend
        scores={}
        for method,frames in variants.items():
            rows=[]
            for i in range(1,len(truth),2):
                gt=truth[i].astype(np.float64);dis=frames[i].astype(np.float64)
                mse=float(np.mean((gt-dis)**2));psnr=10*np.log10(255**2/max(mse,1e-12))
                gy=metric.convert_rgb_to_y_bt709(truth[i]);dy=metric.convert_rgb_to_y_bt709(frames[i])
                flow=flowlib.compute_farneback_flow(frames[i],frames[i+1])
                with np.errstate(divide='ignore',invalid='ignore'):
                    div=float(metric.calculate_psnr_div(gy,dy,flow))
                divergence=np.abs(np.gradient(flow[:,:,0],axis=1)+np.gradient(flow[:,:,1],axis=0))
                peak=float(divergence.max());mask=divergence>peak*.01 if peak>0 else np.zeros_like(divergence,dtype=bool)
                e=(gy.astype(np.float64)-dy.astype(np.float64))**2
                corrected=10*np.log10(255**2/max(float(e[mask].mean()),1e-12)) if mask.any() else None
                x=gy.astype(np.float64);y=dy.astype(np.float64);blur=lambda a:cv2.GaussianBlur(a,(11,11),1.5)
                mx,my=blur(x),blur(y);vx,vy=blur(x*x)-mx*mx,blur(y*y)-my*my;cov=blur(x*y)-mx*my
                ssim=((2*mx*my+6.5025)*(2*cov+58.5225))/((mx*mx+my*my+6.5025)*(vx+vy+58.5225))
                rows.append({'frame':i,'psnr':float(psnr),'mse':mse,'ssim':float(ssim[5:-5,5:-5].mean()),
                    'psnrDivOfficial':div if np.isfinite(div) else None,'psnrDivAxisCorrected':float(corrected) if corrected is not None else None})
            scores[method]={'midpoints':len(rows),'pooledPsnr':float(10*np.log10(255**2/max(np.mean([r['mse'] for r in rows]),1e-12))),
                **{k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in ['psnr','ssim','psnrDivOfficial','psnrDivAxisCorrected']},'frames':rows}
        summary[name]=scores
        save(out/'summary.json',summary)
        print(json.dumps({'case':name,**{m:{k:round(v,4) if isinstance(v,float) else v for k,v in s.items() if k!='frames'} for m,s in scores.items()}}),flush=True)
        display=[]
        for i in range(len(truth)):
            row=np.concatenate([cv2.resize(f[i],(576,324),interpolation=cv2.INTER_AREA) for f in [truth,variants['rife'],variants['dlssg']]],axis=1)
            banner=np.zeros((48,1728,3),dtype=np.uint8)
            for x,label in zip([12,588,1164],[name.upper()+' REFERENCE','RIFE','DLSSG']):cv2.putText(banner,label,(x,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(banner,'Sintel (c) Blender Foundation | CC BY 3.0 | resized/interpolated comparison' if name in ['snow','dialogue','desert'] else 'Analytic 24 fps reference | alternate frames withheld', (12,41),cv2.FONT_HERSHEY_SIMPLEX,.4,(200,200,200),1,cv2.LINE_AA)
            display.append(np.concatenate([banner,row]))
        encode(out/(name+'-comparison.mp4'),display)
        worst=max(range(1,len(truth),2),key=lambda i:scores['dlssg']['frames'][i//2]['mse'])
        cv2.imwrite(str(out/(name+'-worst-dlssg.png')),cv2.cvtColor(display[worst],cv2.COLOR_RGB2BGR))

def present(out,method):
    """Same VSR Ultra and NVENC settings for both methods; isolated local files."""
    import ctypes,gc,subprocess,torch,PyNvVideoCodec as nvc
    from xvideo.serving.vsr_output import VsrUpscaler
    from xvideo.serving.gpu_video_output import CudaRgba
    torch.set_num_threads(2);stream=torch.cuda.Stream(device=0)
    dest=out/(method+'-vsr1440.h264');assert not dest.exists()
    with torch.cuda.device(0),torch.cuda.stream(stream):
        vsr=VsrUpscaler('cuda:0');rgba=torch.empty((1440,2592,4),dtype=torch.uint8,device='cuda:0');rgba[:,:,3]=255
        ctx=ctypes.c_void_p();ctypes.WinDLL('nvcuda.dll').cuCtxGetCurrent(ctypes.byref(ctx))
        enc=nvc.CreateEncoder(2592,1440,'ABGR',False,gpu_id=0,cudacontext=ctx.value,cudastream=stream.cuda_stream,
            codec='h264',preset='P4',tuning_info='ultra_low_latency',rc='cbr',bitrate=15000000,maxbitrate=15000000,
            vbvbufsize=20000000,fps=24,gop=24,idrperiod=24,bf=0,lookahead=0,colorspace='bt709',repeatspspps=1,max_res=[2592,1440],qmin=0)
        wrapper=CudaRgba(rgba,stream);count=0;packets=0
        try:
            with dest.open('wb') as f:
                for case in json.loads((out/'plan.json').read_text())['cases']:
                    frames=np.load(out/(case['name']+'-'+method+'.npy'),mmap_mode='r')
                    for frame in frames:
                        pixels=vsr.process_gpu(frame,(1440,2592));rgba[:,:,:3].copy_(pixels.wait());stream.synchronize()
                        for p in enc.Encode(wrapper):f.write(p['data']);packets+=1
                        stream.synchronize();count+=1;time.sleep(.02)
                    print(json.dumps({'present':method,'case':case['name'],'frames':count}),flush=True)
                for p in enc.EndEncode():f.write(p['data']);packets+=1
        finally:vsr.close()
        assert count==packets,(count,packets)
        del enc,vsr;gc.collect()
    ff=ROOT.parent/'easel/external/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe'
    subprocess.run([str(ff),'-hide_banner','-loglevel','error','-r','24','-i',str(dest),'-c:v','copy','-bsf:v','h264_metadata=video_full_range_flag=1:colour_primaries=1:transfer_characteristics=13:matrix_coefficients=1',str(dest.with_suffix('.mp4'))],check=True)
    save(out/(method+'-presentation.json'),{'frames':count,'packets':packets,'vsr':'Ultra','geometry':[2592,1440],'fps':24,'bitrate':15000000,'sha256':sha(dest.with_suffix('.mp4'))})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','rife','dlssg','score','present']);p.add_argument('--method',choices=['rife','dlssg']);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    if a.mode=='prepare':prepare(a.out)
    elif a.mode=='score':score(a.out)
    elif a.mode=='present':
        assert a.method;present(a.out,a.method)
    else:run(a.out,a.mode)
