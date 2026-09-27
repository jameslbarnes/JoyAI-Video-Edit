"""Isolated 4K VSR/NVENC and receiver trial using the captured RIFE live show.

Never changes the live publisher, native preset, interpolation or GPU settings.
"""
from pathlib import Path
import argparse,ctypes,gc,json,os,subprocess,sys,threading,time,urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'deploy'),str(ROOT/'tulip')]
from interpolation_quality_24 import sha,save
from interpolation_receiver_eval import summarize
SOURCE=ROOT/'tulip/evidence/live-show-ab-20260924/show-rife.npy'
FF=ROOT.parent/'easel/external/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe'
CASES={'1440p-15m':(2592,1440,15000000),'4k-15m':(3840,2160,15000000),'4k-30m':(3840,2160,30000000)}

def health():
    return json.load(urllib.request.urlopen('http://127.0.0.1:7873/health',timeout=10))

def memory():
    r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
    return [[int(v.strip()) for v in row.split(',')] for row in r.stdout.strip().splitlines()]

def render(out,label):
    import torch,PyNvVideoCodec as nvc
    from xvideo.serving.vsr_output import VsrUpscaler
    from xvideo.serving.gpu_video_output import CudaRgba,H264_COLOR_METADATA
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0','This test runs only on physical output GPU 0'
    width,height,bitrate=CASES[label];out.mkdir(parents=True,exist_ok=True)
    dest=out/(label+'.h264');assert not dest.exists(), 'Refusing to overwrite trial'
    before=memory();assert before[0][2]>3500,'Not enough free memory for an isolated output trial'
    report={'case':label,'geometry':[width,height],'fps':24,'bitrate':bitrate,'source':str(SOURCE),'sourceSha256':sha(SOURCE),
        'beforeHealth':health(),'memoryBefore':before,'completed':False,'records':[],
        'scope':'Recorded RIFE outputs uploaded before timed GPU processing. Timed VSR/NVENC has no raw pixel download; concurrent live workload remains active. Excludes JoyAI/RIFE and production NDI/preview branches.'}
    torch.set_num_threads(2);frames=np.load(SOURCE,mmap_mode='r')
    stop=threading.Event();samples=[]
    def monitor():
        while not stop.is_set():
            try:samples.append({'at':time.time(),'gpus':memory()})
            except Exception as e:samples.append({'error':str(e)})
            stop.wait(1)
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    vsr=enc=None;original_cpu=torch.Tensor.cpu
    try:
        with torch.cuda.device(0),torch.inference_mode():
            stream=torch.cuda.Stream(device=0,priority=0)
            with torch.cuda.stream(stream):
                vsr=VsrUpscaler('cuda:0')
                # Diagnostic work has normal priority; the live VSR path retains its high-priority stream.
                vsr.stream=torch.cuda.Stream(device=0,priority=0)
                warm=torch.from_numpy(np.array(frames[0],copy=True)).to('cuda:0')
                for _ in range(2):
                    x=vsr.process_gpu(warm,(height,width));x.ready.synchronize()
                    assert tuple(x.shape)==(height,width,3) and float(x.tensor.float().std())>1,'Black/invalid VSR output'
                del x,warm
                report['memoryAfterVsrWarmup']=memory()
                rgba=torch.empty((height,width,4),device='cuda:0',dtype=torch.uint8);rgba[:,:,3]=255
                wrapper=CudaRgba(rgba,stream);ctx=ctypes.c_void_p()
                assert ctypes.WinDLL('nvcuda.dll').cuCtxGetCurrent(ctypes.byref(ctx))==0 and ctx.value
                enc=nvc.CreateEncoder(width,height,'ABGR',False,gpu_id=0,cudacontext=ctx.value,cudastream=stream.cuda_stream,
                    codec='h264',preset='P4',tuning_info='ultra_low_latency',rc='cbr',bitrate=bitrate,maxbitrate=bitrate,
                    vbvbufsize=20000000,fps=24,gop=24,idrperiod=24,bf=0,lookahead=0,colorspace='bt709',repeatspspps=1,max_res=[width,height],qmin=0)
                assert enc.GetNumBFrames()==0
                def forbidden(*a,**kw):raise AssertionError('CPU pixel readback in timed GPU output path')
                torch.Tensor.cpu=forbidden;packets=0
                with dest.open('wb') as file:
                    for start in range(0,len(frames),16):
                        # Fixed native-sized input upload is intentionally outside the timed stage.
                        batch=torch.from_numpy(np.array(frames[start:start+16],copy=True)).to('cuda:0');stream.synchronize()
                        for i,frame in enumerate(batch):
                            t=time.perf_counter();pixels=vsr.process_gpu(frame,(height,width));pixels.ready.synchronize();t1=time.perf_counter()
                            rgba[:,:,:3].copy_(pixels.wait());stream.synchronize()
                            encoded=enc.Encode(wrapper);stream.synchronize();t2=time.perf_counter()
                            report['records'].append({'frame':start+i,'vsrMs':(t1-t)*1000,'encodeMs':(t2-t1)*1000,'totalMs':(t2-t)*1000})
                            for packet in encoded:file.write(packet['data']);packets+=1
                            if height==2160 and start+i in (48,192,336):
                                # Diagnostic reference only, strictly after timing/encoding.
                                torch.Tensor.cpu=original_cpu
                                try:np.save(out/(label+f'-unencoded-{start+i}.npy'),pixels.wait().cpu().numpy())
                                finally:torch.Tensor.cpu=forbidden
                            del pixels,encoded
                            time.sleep(.025)
                        del batch
                        if start%96==0:print(json.dumps({'case':label,'processed':min(start+16,len(frames))}),flush=True)
                    for packet in enc.EndEncode():file.write(packet['data']);packets+=1
                assert packets==len(frames),(packets,len(frames))
                report.update(completed=True,frames=len(frames),packets=packets,rawDownloadsInTimedPath=0)
                steady=report['records'][24:]
                report['timing']={k:{'median':float(np.median([r[k] for r in steady])),'p95':float(np.percentile([r[k] for r in steady],95)),'max':float(max(r[k] for r in steady))} for k in ['vsrMs','encodeMs','totalMs']}
    except Exception as exc:
        report['error']=repr(exc);raise
    finally:
        torch.Tensor.cpu=original_cpu
        if vsr is not None:vsr.close()
        enc=None;vsr=None;gc.collect()
        stop.set();worker.join(timeout=6)
        report['memorySamples']=samples;report['afterHealth']=health()
        save(out/(label+'-render.json'),report)
    subprocess.run([str(FF),'-hide_banner','-loglevel','error','-r','24','-i',str(dest),'-c:v','copy','-bsf:v',H264_COLOR_METADATA,str(dest.with_suffix('.mp4'))],check=True)
    print(json.dumps({'case':label,'completed':True,'frames':report['frames'],'timing':report['timing'],'minSampledFreeMiB':min(s['gpus'][0][2] for s in samples if 'gpus' in s)}),flush=True)

def receive(out,seconds):
    eth=ROOT.parent/'etherea-ai';edge=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    results={};save(out/'receiver-before-health.json',health())
    for label in CASES:
        publisher=None
        with (out/(label+'-publisher.log')).open('x') as log:
            try:
                publisher=subprocess.Popen([str(FF),'-hide_banner','-loglevel','warning','-re','-stream_loop','-1','-i',str(out/(label+'.mp4')),'-an','-c:v','copy','-f','flv','rtmp://127.0.0.1:1935/joyai-4k-eval'],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                time.sleep(2);assert publisher.poll() is None,'Isolated publisher failed'
                dest=out/(label+'-receiver.jsonl')
                with (out/(label+'-receiver.log')).open('x') as receiver_log:
                    subprocess.run(['node','scripts/streaming/whep-stats-probe.js','--whep-url','http://127.0.0.1:8889/joyai-4k-eval/whep','--seconds',str(seconds),'--no-turn','--frame-timing','--browser-executable',edge,'--out',str(dest)],cwd=eth,stdout=receiver_log,stderr=subprocess.STDOUT,check=True,timeout=seconds+45,creationflags=subprocess.CREATE_NO_WINDOW)
                results[label]=summarize(dest);save(out/'receiver-summary.json',results)
                print(json.dumps({'case':label,**results[label]}),flush=True)
            finally:
                if publisher:
                    publisher.terminate()
                    try:publisher.wait(timeout=10)
                    except subprocess.TimeoutExpired:publisher.kill();publisher.wait(timeout=5)
    save(out/'receiver-after-health.json',health())

def validate(out):
    import av
    results={}
    for name,(width,height,_) in CASES.items():
        rows=[];times=[]
        with av.open(str(out/(name+'.mp4'))) as c:
            s=c.streams.video[0]
            assert (s.width,s.height)==(width,height) and float(s.average_rate)==24
            for i,frame in enumerate(c.decode(video=0)):
                times.append(float(frame.pts*frame.time_base))
                if height==2160 and i in (48,192,336):
                    actual=frame.to_ndarray(format='rgb24');ref=np.load(out/(name+f'-unencoded-{i}.npy'))
                    error=actual.astype(np.float32)-ref.astype(np.float32);mse=float((error**2).mean())
                    rows.append({'frame':i,'psnrVsUnencodedVsr':float(10*np.log10(255**2/max(mse,1e-12))),'maeVsUnencodedVsr':float(np.abs(error).mean())})
        assert len(times)==383 and np.allclose(np.diff(times),1/24,atol=1e-8)
        results[name]={'frames':len(times),'geometry':[width,height],'fps':24,'regularPts':True,
                       'fileBytes':(out/(name+'.mp4')).stat().st_size,'samples':rows}
        if rows:results[name]['meanSamplePsnr']=float(np.mean([r['psnrVsUnencodedVsr'] for r in rows]))
    results['vsrReferenceFramesIdenticalBetweenBitrates']=all(np.array_equal(np.load(out/f'4k-15m-unencoded-{i}.npy'),np.load(out/f'4k-30m-unencoded-{i}.npy')) for i in (48,192,336))
    save(out/'video-validation.json',results)
    print('All three variants have 383 frames and regular 24 fps timestamps.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['render','receive','validate']);p.add_argument('--case',choices=list(CASES));p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=int,default=65);a=p.parse_args()
    if a.mode=='render':assert a.case;render(a.out.resolve(),a.case)
    elif a.mode=='receive':receive(a.out.resolve(),a.seconds)
    else:validate(a.out.resolve())
