"""Render a portable local evidence report from completed benchmark JSON."""
import argparse,html,json
from pathlib import Path

def main(out):
    summary=json.loads((out/'summary.json').read_text())
    receiver=json.loads((out/'receiver-summary.json').read_text())
    rows=[];cards=[]
    for name,methods in summary.items():
        r,d=methods['rife'],methods['dlssg']
        rows.append(f"| {name} | {r['pooledPsnr']:.2f} / {d['pooledPsnr']:.2f} | {r['ssim']:.4f} / {d['ssim']:.4f} | {r['psnrDivOfficial']:.2f} / {d['psnrDivOfficial']:.2f} | {r['psnrDivAxisCorrected']:.2f} / {d['psnrDivAxisCorrected']:.2f} |")
        cards.append(f'<article><h2>{name.title()}</h2><video controls loop preload="metadata" src="{name}-comparison.mp4"></video><p>Reference / RIFE / DLSSG. <a href="{name}-worst-dlssg.png">Highest DLSSG error frame</a></p></article>')
    rec=[]
    for name,s in receiver.items():
        rec.append(f"| {name} | {s['decodedFps']:.3f} | {s['droppedFrames']} | {s['freezeCount']} | {s['expectedDisplayGapsMs']['p99']:.1f} / {s['expectedDisplayGapsMs']['max']:.1f} | {s['mediaJumpsOver1_5Frames']} |")
    text='''# Interpolation quality and pacing — 2026-09-24

Five genuine 24 fps references, 73 frames each. Both production GPU adapters
receive the same 37 even frames per clip at a nominal 12 fps. Only the 36
withheld midpoint frames are scored: 180 predictions per engine. Original
frames remain byte-identical. Neither engine substituted held frames.

## Quality (RIFE / DLSSG; higher is better)

| Case | Pooled RGB PSNR dB | Mean luma SSIM | Published PSNR-DIV | Axis-corrected DIV sensitivity |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

The shader result favors RIFE strongly; the synthetic occlusion result favors
DLSSG strongly. The film cases are closer and metric rankings can disagree.
These clips do not establish a universal winner or substitute for judging
the current JoyAI imagery. Hold and blend baselines are in summary.json.

PSNR is pooled over all withheld RGB pixel errors per clip. SSIM uses BT.709
luma, an 11x11 Gaussian window and a five-pixel border crop. PSNR-DIV uses
Farneback flow from each predicted midpoint to the following original frame.
Only finite midpoint scores are averaged; raw per-frame scores are retained.

Published metric implementation: conalld/psnr-div commit
601bbff9681509ce5671d50782afae6f0ea03855, preserved under psnr-div-source.
The official code uses du/dy + dv/dx, whereas vector-field divergence is
du/dx + dv/dy. An analytic expanding-flow check confirmed the discrepancy.
Both the unmodified official metric and an explicitly labeled axis-corrected
sensitivity result are reported. Both use its normalized 0.01 mask threshold.
Neither is treated as a definitive perceptual score. Source/license remain
separate from production: CC BY-NC 4.0, Daly, Ramsook and Kokaram, QoMEX 2025.

## Repeatability

The snow clip was repeated after resetting each engine. RIFE was byte-exact.
DLSSG differed by a mean absolute 0.00484/255 per RGB channel over the entire
clip (maximum 10/255). Native originals were exact in both runs. The test
procedure is reproducible; DLSSG output was not strictly bit-deterministic.

## Receiver pacing

| Route | Decoded fps | Decoder drops | Freezes | Display gap p99 / max ms | Media steps >62.5 ms |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rec)+'''

RIFE and DLSSG comparison files use identical VSR Ultra and GPU NVENC settings:
2592x1440, 24 fps, H.264 P4 ultra-low-latency, 15 Mbps CBR, GOP 24, no B frames.
Each contains 365 verified decoded frames. A separate RTMP/WHEP route loops
each file in real time; it never replaces the live publisher. This controls
delivery/receiver settings and content sequence, but does not measure engine
compute throughput. The live-DLSSG sample separately includes the real producer.
Other live GPU workload continues throughout, so this is not an isolated lab.

Measurements use a separate headless Edge receiver on this PC and discard its
first five seconds. requestVideoFrameCallback records media time, expected
display time, presentation time and presented-frame counters. Display refresh
quantization can produce 33/50 ms intervals for 24 fps; that alone is not a
streaming defect. Skipped callbacks and media jumps are reported, not silently
equated with decoder drops. This does not measure the actual TV panel.

## Source validity and artifacts

Sintel trailer (c) Blender Foundation / Durian Open Movie Team | www.sintel.org.
CC BY 3.0: https://creativecommons.org/licenses/by/3.0/ . Three excerpts are
downsampled from 1920x1080 to 1152x648, retaining composition/letterboxing;
alternate frames are replaced in comparison variants. Individual source PTS
are asserted to advance by exactly 1/24 second. The source has gaps elsewhere;
those gaps are not included in the selected excerpts. No source frame rate
is relabeled. Shader and occlusion references use exact-time analytic motion.

These are film and synthetic references, not native 24 fps JoyAI output.
Quality metrics apply before VSR/encoding at 648p. The 1440p files allow final
presentation inspection, but are not a measurement of recovered 4K detail.

plan.json has source URLs, frame timestamps and hashes. environment.json
records code/checkpoint hashes, versions, GPUs and live settings. Per-engine
JSON includes profiles, repeat checks and missing-midpoint counts. Raw .npy
files preserve references and predictions; comparison MP4s and worst-error
stills make the scores reviewable. Diagnostic CPU copies are intentional and
are not used to claim production GPU throughput.

DLSSG remains selected live. No resolution, prompt, strength, VSR setting or
automatic fallback policy was changed by this evaluation.

## Reproduce

Run from the JoyAI repository with its existing .venv and CUDA_VISIBLE_DEVICES=0
(physical output GPU). Use a new evidence directory; preserve/copy source and
the pinned metric source into it. Then:

    python tulip/interpolation_quality_24.py prepare --out NEW_DIR
    python tulip/interpolation_quality_24.py rife --out NEW_DIR
    python tulip/interpolation_quality_24.py dlssg --out NEW_DIR
    python tulip/interpolation_quality_24.py score --out NEW_DIR
    python tulip/interpolation_quality_24.py present --method rife --out NEW_DIR
    python tulip/interpolation_quality_24.py present --method dlssg --out NEW_DIR
    python tulip/interpolation_receiver_eval.py --out NEW_DIR --seconds 65
    python tulip/interpolation_quality_report.py --out NEW_DIR

Inference/reference/presentation commands refuse to replace their artifacts.
Receiver logs also refuse overwrites. Scoring/report files can be regenerated.
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    table='<table><tr><th>Case</th><th>RIFE PSNR</th><th>DLSSG PSNR</th><th>RIFE DIV</th><th>DLSSG DIV</th></tr>'+''.join(f"<tr><td>{n}</td><td>{s['rife']['pooledPsnr']:.2f}</td><td>{s['dlssg']['pooledPsnr']:.2f}</td><td>{s['rife']['psnrDivOfficial']:.2f}</td><td>{s['dlssg']['psnrDivOfficial']:.2f}</td></tr>" for n,s in summary.items())+'</table>'
    page='''<!doctype html><meta charset="utf-8"><title>RIFE / DLSSG comparison</title>
<style>body{font:17px system-ui;background:#10151c;color:#eef2fa;max-width:1400px;margin:auto;padding:32px}h1{font-size:32px}p{line-height:1.5;color:#b8c5d8}a{color:#7ebaff}video{width:100%;background:#000}article{margin:30px 0;padding:20px;background:#1a2430;border-radius:12px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #394554;text-align:left}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}</style>
<h1>RIFE / DLSSG: measured comparison</h1><p>Five verified 24 fps references. 180 withheld frames per engine. Higher scores are better. No universal quality winner; see the shader and occlusion cases.</p>
<p><a href="REPORT.md">Full methodology and receiver results</a> · <a href="summary.json">Raw quality scores</a> · <a href="receiver-summary.json">Receiver pacing</a></p>'''+table+''.join(cards)+'''
<h2>Full 1440p presentation files</h2><p>Same VSR Ultra / 24 fps / 15 Mbps NVENC settings. These separate players are for inspection, not synchronized timing measurements. Clip order: snow, dialogue, desert, shader, occlusion.</p><div class="pair"><div><h3>RIFE</h3><video controls loop preload="metadata" src="rife-vsr1440.mp4"></video></div><div><h3>DLSSG</h3><video controls loop preload="metadata" src="dlssg-vsr1440.mp4"></video></div></div>
<p>Sintel © Blender Foundation / Durian Open Movie Team · <a href="https://www.sintel.org">sintel.org</a> · <a href="https://creativecommons.org/licenses/by/3.0/">CC BY 3.0</a>. Resized excerpts and interpolation variants. Shader/occlusion: analytic references.</p>'''
    (out/'index.html').write_text(page,encoding='utf-8')
    print(out/'REPORT.md')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out.resolve())
