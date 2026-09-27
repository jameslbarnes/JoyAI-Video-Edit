"""Blind 12->24 fps A/B from an actual lossless native live-show capture."""
from pathlib import Path
import argparse,hashlib,json,os,random,shutil,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'deploy'),str(ROOT/'tulip')]
from interpolation_quality_24 import present,save,sha

def prepare(out,capture):
    manifest=json.loads((capture/'manifest.json').read_text())
    assert manifest['complete'] and not manifest['error']
    assert not (out/'plan.json').exists()
    segments={};metas={}
    for c in manifest['chunks']:
        frames=np.load(capture/c['file']);assert len(frames)==len(c['metadata'])
        segments.setdefault(c['segment'],[]).append(frames)
        metas.setdefault(c['segment'],[]).extend(c['metadata'])
    key=max(segments,key=lambda k:sum(len(a) for a in segments[k]))
    inputs=np.concatenate(segments[key]);metadata=metas[key]
    assert len(inputs)>=97,'Need at least eight seconds without a scene reset'
    assert inputs.shape[1:]==(648,1152,3) and inputs.dtype==np.uint8
    seq=[m['seq'] for m in metadata]
    assert all(b==a+1 for a,b in zip(seq,seq[1:])), 'Discontinuous native sequence'
    times=np.array([m['t_capture_ms'] for m in metadata]);assert np.all(np.diff(times)>0)
    np.save(out/'native-inputs.npy',inputs)
    save(out/'plan.json',{'cases':[{'name':'show'}],'capture':str(capture),'captureId':manifest['id'],
        'selectedSegment':key,'discardedOtherSegmentFrames':manifest['written']-len(inputs),
        'inputFrames':len(inputs),'outputFrames':2*len(inputs)-1,'inputFps':12,'outputFps':24,
        'sourceCaptureFps':float(1000*(len(times)-1)/(times[-1]-times[0])),
        'sourceFrameStepsMs':{'min':float(np.diff(times).min()),'median':float(np.median(np.diff(times))),'max':float(np.diff(times).max())},
        'inputsSha256':sha(out/'native-inputs.npy'),'metadata':metadata,
        'scope':'Actual native RGB from live JoyAI before interpolation/upscaling. Both engines receive identical frames; no held-out 24 fps ground truth exists for this clip.'})
    print(json.dumps({'prepared':len(inputs),'sourceCaptureFps':1000*(len(times)-1)/(times[-1]-times[0])}),flush=True)

def infer(out,method):
    import torch
    from xvideo.serving.interpolation_output import DlssgOutput
    from xvideo.serving.rife_output import RifeOutput
    os.environ['JOYAI_RIFE_FLUX_ROOT']=str(ROOT.parent/'FluxRT');torch.set_num_threads(2)
    model=(DlssgOutput if method=='dlssg' else RifeOutput)('cuda:0')
    inputs=np.load(out/'native-inputs.npy',mmap_mode='r')
    dest=out/('show-'+method+'.npy');assert not dest.exists()
    result=np.lib.format.open_memmap(dest,mode='w+',dtype=np.uint8,shape=(len(inputs)*2-1,648,1152,3))
    index=0;profiles=[]
    try:
        for start in range(0,len(inputs),8):
            batch=list(inputs[start:start+8]);p={}
            frames,metas=model.process(batch,[{'seq':start+i} for i in range(len(batch))],p,raw_output=True,gpu_output=True)
            for f in frames:
                result[index]=f.wait().cpu().numpy();index+=1
            profiles.append(p);time.sleep(.15)
        assert index==len(result)
        assert np.array_equal(result[::2],inputs),'Native originals changed'
        result.flush()
        save(out/(method+'-interpolation.json'),{'frames':index,'originalsExact':True,'profiles':profiles,'sha256':sha(dest)})
        print(json.dumps({'method':method,'frames':index,'originalsExact':True,'cadenceHolds':sum(p.get('dlssg',{}).get('cadence_holds',0) for p in profiles)}),flush=True)
    finally:
        if hasattr(model,'close'):model.close()

def report(out):
    plan=json.loads((out/'plan.json').read_text());order=['rife','dlssg'];random.SystemRandom().shuffle(order)
    mapping=dict(zip(['A','B'],order));assert not (out/'reveal.json').exists()
    for label,method in mapping.items():
        source=out/(method+'-vsr1440.mp4');dest=out/(label+'.mp4')
        assert not dest.exists();shutil.copyfile(source,dest)
        assert sha(source)==sha(dest)
    save(out/'reveal.json',mapping)
    # Mapping stays collapsed until the viewer explicitly reveals it.
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Live show A/B</title>
<style>body{font:18px system-ui;background:#0d1219;color:#e9f1ff;max-width:1400px;margin:auto;padding:24px}p{color:#b6c3d6;line-height:1.5}video{width:100%;max-height:76vh;background:#000}button,a{font:inherit}button{background:#263c57;color:white;border:1px solid #6488af;border-radius:8px;padding:12px 28px;cursor:pointer;margin:6px}button[aria-pressed=true]{background:#1965ad}a{color:#91c6ff}details{margin:20px 0}small{color:#b6c3d6}</style>
<h1>The live show: A / B</h1><p>The same native JoyAI frames, 12→24 fps, identical VSR Ultra and 1440p / 15 Mbps NVENC output. Choose the motion you prefer before revealing the engines.</p>
<video id="player" src="A.mp4" controls loop playsinline preload="metadata"></video>
<div><button id="a" aria-pressed="true" onclick="choose('A')">A</button><button id="b" aria-pressed="false" onclick="choose('B')">B</button><button onclick="document.querySelector('video').requestFullscreen()">Full screen</button></div>
<p id="status">Viewing A. Switching preserves the clip position.</p>
<p><a href="A.mp4">Open A</a> · <a href="B.mp4">Open B</a> · <a href="REPORT.md">Methodology</a></p>
<details><summary>Reveal the engines</summary><p>REVEAL</p></details>
<small>This is a visual preference test of actual live imagery. There are no real 24 fps reference frames for the generated midpoints, so this clip cannot establish PSNR accuracy. A/B differences are the interpolation engine; the rest of the pipeline is held fixed.</small>
<script>
let version=0;function choose(label){const v=document.getElementById('player');const t=v.currentTime;const play=!v.paused;const n=++version;v.pause();v.src=label+'.mp4';v.onloadedmetadata=()=>{if(n!==version)return;v.currentTime=Math.min(t,Math.max(0,v.duration-.05));if(play)v.play().catch(()=>{});};v.load();document.getElementById('a').setAttribute('aria-pressed',label==='A');document.getElementById('b').setAttribute('aria-pressed',label==='B');document.getElementById('status').textContent='Viewing '+label+'. Switching preserves the clip position.';}
</script>'''.replace('REVEAL',f"A: {mapping['A'].upper()} · B: {mapping['B'].upper()}")
    (out/'index.html').write_text(page,encoding='utf-8')
    (out/'REPORT.md').write_text(f'''# Actual live-show interpolation comparison

Captured {plan['inputFrames']} consecutive native 1152x648 RGB frames from the
running JoyAI show, before interpolation and upscaling. Input bytes are saved
losslessly, with per-frame sequence IDs and capture timestamps. Both engines
receive exactly these bytes and produce {plan['outputFrames']} frames at 24 fps.
The nominal live input rate is 12 fps; measured capture rate in this sample was
{plan['sourceCaptureFps']:.3f} fps. No scene reset occurred within the chosen segment.

Both production adapters preserve every native original exactly. Rendering uses
the same VSR Ultra and 2592x1440 H.264 NVENC P4 / 15 Mbps / no B-frame settings.
Both are evaluated offline; diagnostic CPU downloads do not change or benchmark
the live GPU-only presentation path. No PSNR claim is possible without real
24 fps ground truth for this generated scene.

A/B assignment was randomized once, recorded in reveal.json, and copied without
re-encoding. The viewer uses a single video element to avoid dual decoding and
offers an explicit engine reveal. For comparison, watch the whole clip, then
switch; do not interpret the source-switch pause as interpolation jitter.

The renderer was restarted once to load the optional capture hook. Its existing
648p/12 fps/DLSSG/VSR configuration was retained. The lifecycle helper requested
the saved prompt, but the live show supplied newer prompts during startup and
capture. Its ownership check did not overwrite the newer state or restore the
old strength across that prompt change. Before/capture/after snapshots retain
the exact creative-state history; this is a capture of the progressing show.
The capture command is disarmed after completion; DLSSG remains live.

The tap uses a four-chunk disk queue and a maximum frame count. It copies native
CPU arrays without a GPU readback. A full queue marks capture invalid without
blocking the renderer. It preserves scene boundaries and refuses existing IDs.

Evidence: capture-manifest.json, plan.json, native-inputs.npy, per-engine
interpolation JSON, per-engine presentation JSON, settings-preserved.json,
before/after health and debug snapshots. The reveal is optional in index.html.
''',encoding='utf-8')
    print(json.dumps({'ready':str(out/'index.html'),'frames':plan['outputFrames'],'durationSeconds':plan['outputFrames']/24}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','rife','dlssg','present','report']);p.add_argument('--out',type=Path,required=True);p.add_argument('--capture',type=Path);p.add_argument('--method',choices=['rife','dlssg']);a=p.parse_args();out=a.out.resolve()
    if a.mode=='prepare':prepare(out,a.capture.resolve())
    elif a.mode=='present':present(out,a.method)
    elif a.mode=='report':report(out)
    else:infer(out,a.mode)
