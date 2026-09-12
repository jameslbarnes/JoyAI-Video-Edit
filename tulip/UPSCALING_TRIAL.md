# Tulip GPU-0 output upscaling trial — September 12, 2026

**Result:** compact pixel-space SR is the promising option on this footage. The
existing Flux latent upscaler can be reused only after encoding finished JoyAI RGB
frames into Flux's VAE space. That round-trip was slower and did not look sharper.
This is NOT native JoyAI latent SR, and is not evidence that a future upscaler
trained for JoyAI's own VAE would be inferior.

The live app remains unchanged at 648x360 with its existing RIFE. These scripts
are isolated trials, not an installed primary-provider upscaling integration.
They never write prompts, shader strength, scene/cache state, mode, model service
parameters or app routing. GPU 1 kept generating JoyAI while Flux remained paused
on GPU 0. Only SR weights (and the tiny VAE for the latent experiment) ran on GPU 0.

## Artifacts and versions

All paths below are relative to the JoyAI repo. Existing `.venv` was reused without
installing/upgrading packages. Native Windows, two RTX 5090s. At initial check GPU
0 had 9,929 MiB free; it also retained Flux weights and existing shader/display
work. GPU 0 is not dedicated or idle. JoyAI text encoding also uses it on prompts.

- Python 3.12.10; torch 2.9.1+cu128; diffusers 0.36.0; numpy 2.2.6;
  OpenCV 4.13.0; Pillow 12.2.0; websockets 16.0; safetensors 0.8.0.
- Official [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) checkout:
  `tulip/research/Real-ESRGAN`, commit
  `a4abfb2979a7bbff3f69f58f58ae324608821e27` (BSD-3-Clause, LICENSE retained).
- `tulip/models/realesrgan/realesr-general-x4v3.pth`, SHA256
  `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292`.
- `tulip/models/realesrgan/realesr-general-wdn-x4v3.pth`, SHA256
  `1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca`.
- Only these two small release checkpoints were downloaded (~10 MB total).
  They implement the official denoising-weight interpolation; .5 was the live
  trial setting, with 0 (weak denoising) also compared offline.
- No BasicSR installation. The loader reads the pinned official architecture and
  removes only its BasicSR registry import/decorator in memory. Source on disk is
  untouched. Weight loads use `weights_only=True`, hash checks and strict shapes.
- Pixel route: FP16/channels-last SRVGG, native 4x network -> antialiased bicubic
  GPU reduction to 2x (1296x720). This last resize differs from the official CLI's
  CPU Lanczos final resize. Batch size 1, cudnn autotune, no torch.compile.
- Flux experiment reads `../FluxRT` at
  `bf4c169edfc813424d14d8427c4bc37c7530f89b`: BF16 tiny TAEF2 encode -> one Flow
  Upscaler step -> tiny TAEF2 decode. Input pads to 16px multiples then output
  crops to 1296x720. Seed resets to 52 per frame (avoids random noise changes).
  No Flux DiT/text encoder is loaded by the trial. Existing scheduler config is
  read from the local FLUX.2-klein-4B snapshot
  `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
- Read-only Flow checkpoint SHA256:
  `91eb93b40179441e569e01fe04f20fd9b951e434f88c84dd0ba523f830a81839`.
  Read-only TAEF2 checkpoint SHA256:
  `701d31c0ecbef3b609fe236e2e3a65ffb17f3a94237acdd771e8ea3194df557a`.

## Measured results

96 frames from the live JoyAI/RIFE NDI output, saved as a six-second 16fps test
clip of the operator's rotating orange/teal cube. This is sampled presentation
output, NOT a claim of 96 independent native model frames. No new prompt was
authored. Final pixel/latent comparison both read the same saved input MP4.

| Isolated 2x stage | Mean processing | p95 | GPU-stage mean | Warm stage throughput |
|---|---:|---:|---:|---:|
| Pixel SR .5, FP16 | 6.85 ms | 9.19 ms | 5.13 ms | 146 frames/s |
| Flux latent round-trip, BF16 | 24.30 ms | 29.41 ms | 22.31 ms | 41 frames/s |
| Pixel SR weak denoise, FP16 | 6.76 ms | 8.66 ms | 5.12 ms | 148 frames/s |

Processing includes CPU RGB input conversion/upload, model, final resize/crop and
synchronized GPU->CPU byte output. Input video decode and final file encoding are
outside this timing. GPU events cover SR only for pixel and encoder+SR+decoder for
the latent route. This is postprocessing throughput, NOT JoyAI generation FPS.

Model construction/load (excluding Python/import/CUDA context startup): pixel
.10s, latent 3.79s. First inference/autotuning: pixel .86s, latent 11.52s. Excluded
from warm rates. Pixel first independent run was 1.19s. Warm PyTorch peak
allocated/reserved: pixel 78/150 MiB, latent 722/882 MiB. These are NOT total
process GPU memory. Scripts cap PyTorch allocation at 12% of GPU 0 capacity.

Thirty-second actual live NDI trial, corrected targeted sockets (`?video=0`):

- 640 processed frames, 21.32 frames/s; two explicitly dropped stale input slots.
- One pending input frame maximum. No model state/cache feedback from SR outputs.
- Pixel processing p50/p95 6.77/13.79ms.
- Input frame available to Python -> submitted raw output: p50/p95
  **17.26/25.46ms**. Includes queueing, SR, packing and socket submission; excludes
  earlier source/NDI decode and subsequent bridge encoding, network and display.
  This is NOT capture-to-phone latency or confirmed delivery latency.
- Queue p50/p95 .84/6.95ms; socket submission 2.01/3.64ms.
- Independent receiver verified 353 returned 1296x720 frames, 312 differing sparse
  image samples; representative received PNGs saved. Output receipt count is not
  native generation count, and the bridge may repeat frames.
- Live-stage PyTorch peak allocated/reserved 175/198 MiB (includes warmup).

The earlier equivalent trial processed 21.27fps. A plain Lanczos resize baseline
through the same receive/publish path processed 20.43fps (4 stale-slot drops),
with 9.63/13.48ms arrival-to-submission p50/p95. The sidecar NDI path, not isolated
SR compute, limits this prototype to about 20–21fps. The first trial sockets also
received unnecessary shared video; targeted sockets did not materially improve
the rate. Do not enable this transport by default and advertise 32fps.

JoyAI stayed at 16.12 native displayed frames/s in both the baseline and initial
SR live test. In the SR test there was one additional capture tick drop, no output
drops, and epoch 10 remained unchanged. Its sampled source-to-canvas latency
averaged 989ms (baseline 979ms), maximum 1057ms in both. This small non-simultaneous
sample does NOT prove zero contention. Native Easel shader GPU frame-time
percentiles, remote iPhone presentation and SR contention during GPU-0 text
encoding were not measured. No claim of improved shader rendering performance.

## Visual findings

Inspected full-resolution comparison PNGs, five time points across the clip and
received live NDI output. Pixel SR .5 cleans jagged/soft edges and surface noise,
but smooths grain/fine texture and can produce an overly polished look. Weak
denoise preserves more texture but is much closer to the original softness.
Flux latent round-trip remains softer with small shading changes; it did not
provide a visible detail advantage here. Cube motion and palette remain
recognizable in all sampled outputs. Existing warped cube geometry comes from
the source; upscaling does not fix it. This short single-scene inspection is not
proof of shimmer-free temporal restoration across different shaders/scenes.

- `tulip/evidence/upscale-comparison-20260912/comparison.mp4` — three-way moving
  comparison (display-scaled columns; inspect full-resolution PNG for fine detail).
- `.../motion-contact-sheet.png`, `.../pixel-vs-latent.png`.
- `tulip/evidence/upscale-pixel-same-input-20260912/sr-2x.mp4` — pixel .5, full 720p.
- `tulip/evidence/upscale-flux-latent-20260912/sr-2x.mp4` — latent, full 720p.
- Each trial directory includes `report.json`, input and comparison frames.
- `upscale-live-targeted-20260912` holds the final live report/received frames;
  `upscale-live-resize-baseline-20260912` holds the transport baseline.

## Reproduce / shutdown

From `C:\Users\Admin\src\joyai-video-edit`, use the existing isolated environment.
For a new installation of the research assets only (do not overwrite an existing
research checkout):

```powershell
git clone --depth 1 https://github.com/xinntao/Real-ESRGAN.git tulip/research/Real-ESRGAN
git -C tulip/research/Real-ESRGAN fetch --depth 1 origin a4abfb2979a7bbff3f69f58f58ae324608821e27
git -C tulip/research/Real-ESRGAN switch --detach a4abfb2979a7bbff3f69f58f58ae324608821e27
New-Item -ItemType Directory -Force tulip/models/realesrgan
# Download both official files; the Python loader verifies the hashes above:
Invoke-WebRequest https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -OutFile tulip/models/realesrgan/realesr-general-x4v3.pth
Invoke-WebRequest https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth -OutFile tulip/models/realesrgan/realesr-general-wdn-x4v3.pth
```

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
.venv/Scripts/python.exe tulip/upscale_trial.py --out tulip/evidence/NEW --seconds 6 --fps 16
# Same input, Flux latent comparison:
.venv/Scripts/python.exe tulip/upscale_trial.py --kind flux-latent --input tulip/evidence/NEW/input.mp4 --out tulip/evidence/NEW-LATENT --seconds 6 --fps 16
# Live isolated trial (JoyAI must be active, Flux paused):
.venv/Scripts/python.exe tulip/upscale_live_trial.py --out tulip/evidence/NEW-LIVE --seconds 30
# Transport-only baseline:
.venv/Scripts/python.exe tulip/upscale_live_trial.py --kind resize --out tulip/evidence/NEW-BASELINE --seconds 30
.venv/Scripts/python.exe tulip/test_output_upscalers.py
```

The live trial uses only loopback bridge 7779 and read-only status ports 8793 and
7865; it opens no new listening ports. It publishes a uniquely named
`ETHEREA Upscale Trial PID` sender, not the production sender. It stops after the
bounded trial (5–60 seconds selectable). Ctrl-C also closes receivers and sends
an acknowledged stop for its exact sender. After the tests, a real bridge status
query confirmed no trial senders remained. Process exit releases SR weights.

Four tests passed: GPU shape/determinism for both SR paths, denoise validation,
and command-ack handling (initial WebSocket hello must not count as success).

No production rollback is needed: the app was never switched to this output.
Keep the evidence and scripts; no other caches/environments/recordings were
deleted. For production, the next step is a bounded GPU-0 SR stage directly on
JoyAI output (preferably before final JPEG encode), avoiding the extra NDI
receive/publish round-trip. Recheck RIFE ordering, 32fps delivery, scene resets,
Photo/Video switching and shader frame times before enabling it in the app.
