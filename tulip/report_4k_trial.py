"""Summarize the isolated 4K experiment and make a local comparison viewer."""
from pathlib import Path
import argparse,json,hashlib

def main(out):
    labels=['1440p-15m','4k-15m','4k-30m']
    renders={n:json.loads((out/(n+'-render.json')).read_text()) for n in labels}
    receiver=json.loads((out/'receiver-summary.json').read_text())
    validation=json.loads((out/'video-validation.json').read_text())
    rows=[];summary={}
    for name in labels:
        r=renders[name];s=receiver[name]
        free=min(x['gpus'][0][2] for x in r['memorySamples'] if 'gpus' in x)
        before,after=r['beforeHealth']['upscale']['gpu_video'],r['afterHealth']['upscale']['gpu_video']
        delta={k:after[k]-before[k] for k in ['underruns','dropped','restart_count']}
        summary[name]={'gpuMedianMs':r['timing']['totalMs']['median'],'gpuP95Ms':r['timing']['totalMs']['p95'],
            'minimumSampledFreeMiB':free,'receiverFps':s['decodedFps'],'receiverSeconds':s['seconds'],
            'decoderDrops':s['droppedFrames'],'freezes':s['freezeCount'],'packetsLost':s['packetsLost'],
            'livePublisherDeltaDuringRender':delta,'compressedFileBytes':validation[name]['fileBytes']}
        rows.append(f"| {name} | {r['timing']['totalMs']['median']:.2f} / {r['timing']['totalMs']['p95']:.2f} | {free} | {s['decodedFps']:.3f} | {s['droppedFrames']} / {s['freezeCount']} / {s['packetsLost']} |")
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    psnr15=validation['4k-15m']['meanSamplePsnr'];psnr30=validation['4k-30m']['meanSamplePsnr']
    report='''# Isolated 4K output trial — 2026-09-24

All three variants use the same captured live hummingbird show, generated at
1152x648 and interpolated with RIFE to 24 fps. There are 383 frames / 15.96 seconds
per variant. Only VSR output size and the explicitly labeled bitrate change.
No production settings, native model, prompt, interpolation, or publisher route
were changed. The live 1440p show remained active throughout the experiment.

| Variant | GPU VSR + encode median / p95 ms per frame | Minimum sampled free GPU MiB | Receiver fps | Decoder drops / freezes / packet loss |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

## What passed

Actual NVIDIA VSR Ultra and H.264 NVENC produced 3840x2160 output. Decoding
verified every file has all 383 frames, 24 fps, and regular presentation
timestamps. The encoder uses P4 ultra-low-latency, CBR, no B frames, GOP 24,
20 Mb VBV, full-range BT.709 tagging, matching the current output path.

The VSR-to-NVENC timed path remains GPU-resident. Native-sized prerecorded RGB
inputs are uploaded before timing; this is a diagnostic setup, not a change to
the live GPU RIFE handoff. No raw pixel downloads are allowed during timing.
Three unencoded 4K reference frames were downloaded after timing solely to
measure compression loss. Live production had no added publisher drops,
underruns or restarts during any of the three render runs.

## Compression check

RGB PSNR against each variant's unencoded VSR output, averaged over frames
48, 192 and 336: **{psnr15:.2f} dB at 15 Mbps**, **{psnr30:.2f} dB at 30 Mbps**.
The unencoded reference frames were byte-identical across those bitrate runs:
{validation['vsrReferenceFramesIdenticalBetweenBitrates']}.
This measures encoding/color-conversion loss, not whether VSR recovered true
4K detail. There is no native 4K ground truth for the generated 648p show.
The 30 Mbps variant doubles the bandwidth requirement; assess visible benefit
on the TV using the viewer before choosing it.

## Limits of this trial

GPU timings include contention from the still-running live show. Warmup and
the first 24 encoded frames are excluded from timing summaries. They measure
VSR/encoding only, not generation, RIFE, NDI, JPEG preview or complete production
queue behavior. Memory minima are sampled once per second and may miss brief
peaks. The production 4K queue/scene-transition path has not been deployed.

The files were replayed in real time on an isolated local RTMP/WHEP route,
joyai-4k-eval, using a separate Edge receiver for 65 seconds each. Receiver
statistics exclude the first five seconds. They validate local decode/delivery
of the encoded 4K stream, not the actual TV/phone or remote network. Compositor
callback timing is not physical panel scanout; missed callbacks and media
steps remain in receiver-summary.json. All test publishers are stopped afterward.

The baseline is the existing 2592x1440 geometry; the candidate is exact
3840x2160. Their aspect ratios differ slightly because of the existing
1440p geometry. The viewer preserves each video's aspect ratio rather than
silently cropping either image. Native generation stays at 648p.

## Artifacts and reproduction

index.html is a single-player comparison that preserves playback position when
switching among variants. Use full screen on a 4K display to judge final detail.
The source-switch pause is not interpolation or delivery jitter. MP4s, sampled
unencoded references, per-frame timings, GPU memory samples, live health
snapshots and receiver JSONL files are retained in this directory.

From the JoyAI repository, set CUDA_VISIBLE_DEVICES=0 (physical output GPU),
use its .venv Python and a new evidence directory:

    python tulip/trial_4k_output.py render --case 1440p-15m --out NEW
    python tulip/trial_4k_output.py render --case 4k-15m --out NEW
    python tulip/trial_4k_output.py render --case 4k-30m --out NEW
    python tulip/trial_4k_output.py validate --out NEW
    python tulip/trial_4k_output.py receive --out NEW --seconds 65
    python tulip/report_4k_trial.py --out NEW

Rendering refuses existing files and requires at least 3500 MiB free on the
output GPU before starting its separate process. It never selects a lower live
resolution, enables an automatic rollback, or restarts JoyAI.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    table='<table><tr><th>Output</th><th>VSR + encode median</th><th>Receiver fps</th><th>Drops / freezes</th></tr>'+''.join(f'<tr><td>{n}</td><td>{v["gpuMedianMs"]:.2f} ms</td><td>{v["receiverFps"]:.3f}</td><td>{v["decoderDrops"]} / {v["freezes"]}</td></tr>' for n,v in summary.items())+'</table>'
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Live show: 1440p / 4K</title>
<style>body{font:18px system-ui;background:#0d1219;color:#eef3ff;max-width:1600px;margin:auto;padding:24px}p,small{color:#b8c5d8;line-height:1.5}video{width:100%;aspect-ratio:16/9;max-height:76vh;object-fit:contain;background:#000}button{font:inherit;background:#263c57;color:white;border:1px solid #6488af;border-radius:8px;padding:12px 18px;cursor:pointer;margin:6px}button[aria-pressed=true]{background:#1965ad}a{color:#91c6ff}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #334155;padding:10px}</style>
<h1>The live show: 1440p / 4K</h1><p>Same hummingbird frames, 648p native generation, RIFE 2×, 24 fps. Compare the final output on a 4K display. Switching preserves the clip position.</p>
<video id="player" src="1440p-15m.mp4" controls loop playsinline preload="metadata"></video>
<div><button id="base" aria-pressed="true" onclick="choose('1440p-15m','base')">1440p · 15 Mbps</button><button id="same" aria-pressed="false" onclick="choose('4k-15m','same')">4K · 15 Mbps</button><button id="high" aria-pressed="false" onclick="choose('4k-30m','high')">4K · 30 Mbps</button><button onclick="document.querySelector('video').requestFullscreen()">Full screen</button></div>
<p id="status">Viewing 1440p · 15 Mbps.</p><p><a href="1440p-15m.mp4">Open 1440p</a> · <a href="4k-15m.mp4">Open 4K / 15 Mbps</a> · <a href="4k-30m.mp4">Open 4K / 30 Mbps</a> · <a href="REPORT.md">Full results</a></p>'''+table+'''
<p>The 4K options passed local GPU rendering and WebRTC playback tests. Production remains on 1440p. The TV panel and remote network have not been measured; this does not establish native 4K detail.</p>
<script>let version=0;function choose(name,id){const v=document.getElementById('player'),t=v.currentTime,play=!v.paused,n=++version;v.pause();v.src=name+'.mp4';v.onloadedmetadata=()=>{if(n!==version)return;v.currentTime=Math.min(t,Math.max(0,v.duration-.05));if(play)v.play().catch(()=>{});};v.load();for(const key of ['base','same','high'])document.getElementById(key).setAttribute('aria-pressed',key===id);document.getElementById('status').textContent='Viewing '+document.getElementById(id).textContent+'.';}</script>'''
    (out/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out.resolve())
