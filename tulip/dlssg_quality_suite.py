"""Broader held-out-frame quality checks; CPU copies here are diagnostic only."""
from pathlib import Path
import argparse,gc,json,os,statistics,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tulip/evidence/dlssg-gpu-20260923/quality';OUT.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','rife','dlssg','summary']);a=p.parse_args()
sys.path[:0]=[str(ROOT/'tulip/research/dlssg-python'),str(ROOT/'deploy'),str(ROOT/'tulip')]
if a.mode=='dlssg':
    from dlssg_gpu_bridge import GpuDlssg,load_bridge
    native,_,_=load_bridge();native.initialize_bridge(0)
import av,cv2,numpy as np
cv2.setNumThreads(1)

def encode(path,frames,rate=24):
    with av.open(str(path),'w') as c:
        s=c.add_stream('libx264',rate=rate);s.width=frames.shape[2];s.height=frames.shape[1];s.pix_fmt='yuv420p'
        s.options={'crf':'16','preset':'fast','threads':'2'}
        for i,x in enumerate(frames):
            f=av.VideoFrame.from_ndarray(np.asarray(x),format='rgb24');f.pts=i
            for packet in s.encode(f):c.mux(packet)
        for packet in s.encode():c.mux(packet)

if a.mode=='prepare':
    cases=[]
    cases.append({'name':'organic','reference':str(ROOT/'tulip/evidence/dlssg-rife-20260923/reference.npy'),
        'provenance':'Archived 1248x720 JoyAI, center-cropped/downsampled to 648p, source 16 fps retimed to 24 fps.'})
    for name,path,start in [
        ('porcelain',ROOT/'tulip/evidence/strength-fine-20260911/porcelain-blend0.1/output.mp4',64),
        ('shader',ROOT.parent/'sana-streaming/tulip/evidence/shader-ab/input.mp4',0)]:
        frames=[]
        with av.open(str(path)) as c:
            s=c.streams.video[0];source_geometry=[s.width,s.height];source_fps=str(s.average_rate)
            for i,f in enumerate(c.decode(video=0)):
                if i<start:continue
                x=f.to_ndarray(format='rgb24');h,w=x.shape[:2]
                if w/h>16/9:
                    cw=int(h*16/9);left=(w-cw)//2;x=x[:,left:left+cw]
                else:
                    ch=int(w*9/16);top=(h-ch)//2;x=x[top:top+ch]
                frames.append(cv2.resize(x,(1152,648),interpolation=cv2.INTER_CUBIC))
                if len(frames)==97:break
        assert len(frames)==97
        dest=OUT/(name+'-reference.npy');np.save(dest,np.stack(frames))
        cases.append({'name':name,'reference':str(dest),'source':str(path),'sourceGeometry':source_geometry,'sourceFps':source_fps,
            'provenance':'Lower-resolution archived content enlarged to 648p; tests motion, not recovery of 648p spatial detail. Retimed to 24 fps.'})
    h,w=648,1152;y,x=np.mgrid[:h,:w].astype('float32')
    bg=np.stack([35+35*np.sin(x/31),40+30*np.cos(y/23),75+40*np.sin((x+y)/47)],axis=-1).clip(0,255).astype('uint8')
    texture=np.stack([160+70*np.sin(x/4),120+90*np.cos(y/6),100+80*np.sin((x+y)/7)],axis=-1).clip(0,255).astype('uint8')
    frames=[]
    for i in range(97):
        im=cv2.warpAffine(bg,np.float32([[1,0,-i*2],[0,1,i*.5]]),(w,h),borderMode=cv2.BORDER_WRAP)
        m=cv2.getRotationMatrix2D((576,324),i*1.2,1);m[:,2]+=[80*np.sin(i/21),30*np.sin(i/13)]
        moving=cv2.warpAffine(texture,m,(w,h),borderMode=cv2.BORDER_REFLECT)
        mask=np.zeros((h,w),dtype='uint8');cv2.ellipse(mask,(int(400+i*4),324),(240,145),i*1.2,0,360,255,-1,cv2.LINE_AA)
        alpha=mask.astype('float32')[:,:,None]/255;im=(im*(1-alpha)+moving*alpha).astype('uint8')
        for k in range(7):
            xx=int((950-i*7+k*32)%(w+180)-90)
            cv2.line(im,(xx,100),(xx+100,555),(230,220,205),3,cv2.LINE_AA)
        cv2.circle(im,(int(800+200*np.sin(i/15)),int(220+110*np.cos(i/17))),55,(10,15,35),-1,cv2.LINE_AA)
        frames.append(im)
    dest=OUT/'occlusion-reference.npy';np.save(dest,np.stack(frames))
    cases.append({'name':'occlusion','reference':str(dest),'provenance':'Synthetic 648p/24 fps ground truth with rotation, scrolling texture, thin lines and independent occluders.'})
    (OUT/'plan.json').write_text(json.dumps({'cases':cases,'inputStride':2,'nominalInputFps':12,'nominalOutputFps':24},indent=2))
    print('Prepared four motion/content cases.',flush=True);sys.exit()

plan=json.loads((OUT/'plan.json').read_text())
if a.mode in ('rife','dlssg'):
    import torch
    torch.set_num_threads(2)
    if a.mode=='rife':
        os.environ['JOYAI_RIFE_FLUX_ROOT']=str(ROOT.parent/'FluxRT')
        from xvideo.serving.rife_output import RifeOutput
        model=RifeOutput('cuda:0')
    for case in plan['cases']:
        truth=np.load(case['reference'],mmap_mode='r');inputs=truth[::2]
        dest=OUT/(case['name']+'-'+a.mode+'.npy')
        result=np.lib.format.open_memmap(dest,mode='w+',dtype='uint8',shape=truth.shape)
        rows=[];index=0
        if a.mode=='rife':
            model.reset();model.process(list(inputs[:8]),[{'seq':i} for i in range(8)],{},raw_output=True);model.reset()
            for start in range(0,len(inputs),8):
                batch=list(inputs[start:start+8]);frames,_=model.process(batch,[{'seq':start+i} for i in range(len(batch))],{},raw_output=True)
                result[index:index+len(frames)]=frames;index+=len(frames);time.sleep(.5)
        else:
            model=GpuDlssg()
            try:
                for i,f in enumerate(inputs):
                    frames,detail=model.process(cv2.cvtColor(f,cv2.COLOR_RGB2RGBA),reset=i==0)
                    rows.append({'input':i,**detail})
                    # Copy for offline quality analysis only, outside performance measurements.
                    if i and len(frames)==1:
                        result[index]=inputs[i-1];index+=1
                        rows[-1]['missingInterpolation']=True
                    for frame in frames:
                        result[index]=frame.wait().cpu().numpy();index+=1
                    del frames,frame;gc.collect();time.sleep(.1)
            finally:model.close()
        assert index==len(truth),(index,len(truth))
        assert np.array_equal(result[::2],truth[::2]),'Originals changed'
        result.flush();(OUT/(case['name']+'-'+a.mode+'.json')).write_text(json.dumps({'completed':True,'frames':index,'details':rows},indent=2))
        print(json.dumps({'case':case['name'],'method':a.mode,'frames':index,'missingInterpolations':sum(bool(r.get('missingInterpolation')) for r in rows)}),flush=True)
    sys.exit()

def score(a,b):
    x=a.astype('float32');y=b.astype('float32');mse=float(np.mean((x-y)**2))
    xx=cv2.cvtColor(x,cv2.COLOR_RGB2GRAY);yy=cv2.cvtColor(y,cv2.COLOR_RGB2GRAY)
    blur=lambda z:cv2.GaussianBlur(z,(11,11),1.5)
    mx,my=blur(xx),blur(yy);vx,vy=blur(xx*xx)-mx*mx,blur(yy*yy)-my*my;xy=blur(xx*yy)-mx*my
    s=((2*mx*my+6.5025)*(2*xy+58.5225))/((mx*mx+my*my+6.5025)*(vx+vy+58.5225))
    return {'psnr':float(10*np.log10(255**2/max(mse,1e-9))),'ssim':float(s[5:-5,5:-5].mean()),'mae':float(np.abs(x-y).mean())}

all_results={}
for case in plan['cases']:
    name=case['name'];truth=np.load(case['reference'],mmap_mode='r')
    variants={m:np.load(OUT/(name+'-'+m+'.npy'),mmap_mode='r') for m in ['rife','dlssg']}
    summary={};errors={}
    for method,frames in variants.items():
        rows=[];deltas=[];prior=None
        for i in range(1,len(truth),2):
            rows.append({'frame':i,**score(frames[i],truth[i])})
            error=frames[i].astype('float32')-truth[i].astype('float32')
            if prior is not None:deltas.append(float(np.abs(error-prior).mean()))
            prior=error
        summary[method]={'meanPsnr':statistics.mean(r['psnr'] for r in rows),'meanSsim':statistics.mean(r['ssim'] for r in rows),'meanMae':statistics.mean(r['mae'] for r in rows),'temporalErrorChangeMae':statistics.mean(deltas),'frames':rows}
    display=[]
    for i in range(len(truth)):
        row=np.concatenate([truth[i],variants['rife'][i],variants['dlssg'][i]],axis=1)
        row=cv2.resize(row,(1728,324),interpolation=cv2.INTER_AREA);banner=np.zeros((36,1728,3),dtype='uint8')
        for x,label in zip([12,588,1164],[name.upper()+' REFERENCE','RIFE 2x','DLSSG 2x']):cv2.putText(banner,label,(x,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
        display.append(np.concatenate([banner,row]))
    encode(OUT/(name+'-comparison.mp4'),np.stack(display))
    largest=sorted(range(1,len(truth),2),key=lambda i:summary['dlssg']['frames'][i//2]['mae']-summary['rife']['frames'][i//2]['mae'])
    for i in sorted(set([largest[0],largest[-1],len(truth)//2|1])):cv2.imwrite(str(OUT/(name+f'-frame-{i}.png')),cv2.cvtColor(display[i],cv2.COLOR_RGB2BGR))
    all_results[name]=summary
    print(json.dumps({'case':name,**{m:{k:v for k,v in s.items() if k!='frames'} for m,s in summary.items()}}),flush=True)
(OUT/'summary.json').write_text(json.dumps(all_results,indent=2))
