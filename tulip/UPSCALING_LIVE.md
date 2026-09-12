# JoyAI live 720p output on Tulip

September 12, 2026. This implements the winning pixel-space experiment from
[UPSCALING_TRIAL.md](UPSCALING_TRIAL.md), not Flux latent conversion.

## Pipeline and controls

JoyAI (physical GPU 1, native 648x360 at 16 input FPS, existing FP8 DiT/BF16 VAE)
-> raw CPU RGB -> existing native-resolution FP16 RIFE 2x (GPU 1)
-> FP16 Real-ESRGAN compact SR (physical GPU 0) -> final JPEG quality 95
-> existing JoyAI WebSocket -> Etherea 1296x720 canvas -> production NDI sender
-> existing preview publisher / WebRTC. There is no experimental NDI sidecar hop.

The SR model is the pinned general-x4v3/weak-denoise blend at .5, native 4x
followed by antialiased GPU bicubic reduction to 2x. Exact source revision,
checkpoint hashes, package versions and download commands are in the trial doc.
No new packages, drivers or models were needed for this integration.

The capture canvas remains 648x360. SR/interpolated pixels never enter JoyAI's
KV/VAE history. One GPU SR frame at a time; one presentation chunk of at most
16 frames. Four final JPEG worker threads, OpenCV inner threads fixed at one
in the JoyAI process only. All encoding tasks finish before the chunk returns.
The existing output pump backpressures generation; no additional rolling queue.
The browser allows at most 24 outstanding native input frames with SR (16
without), accounting for the extra output-stage chunk. Its presentation buffer
remains 375ms initially, with a one-second maximum output queue. This is bounded
pipeline credit, not a promise of low capture-to-display latency.

The desktop tile has independent **SR 2x / SR off** and **RIFE 2x / RIFE off**
buttons. SR toggles preserve prompt, shader adherence, session epoch and RIFE.
SR status/timings appear in `getStats().upscale`, `/health.upscale`, and each
WebSocket `output_frame.profile.upscale`. The protocol adds `set_upscale` with
boolean `enabled`, `upscale_changed` and `upscale_error`; native `width`/`height`
remain input dimensions, with separate `output_width`/`output_height` metadata.
Failed SR inference latches SR off, reports the error and returns native output;
retry requires a service restart. Unsupported source shapes are rejected.

Etherea's JoyAI NDI cap is 1296px; other tiles retain their 960px cap. Actual JPEG
dimensions drive the output canvas, so an ACK cannot crop an older in-flight frame.
The SDK's raw NDI receiver detects resolution changes *before* writing mismatched
bytes. The preview supervisor then re-probes/reopens its fixed-resolution encoder.
The local webstream launcher passes `--expect-size WIDTHxHEIGHT` from its probe.
Thus SR resolution toggles can briefly reconnect WebRTC, but cannot reinterpret
720p pixels using a 360p rawvideo stride. The iPhone needs no new provider API to
receive 720p; a phone-side SR toggle UI is not implemented here.

Flux weights stay resident on GPU 0 but Flux generation is paused in Video mode.
SR weights stay resident but do not run when JoyAI is inactive in Photo mode.
GPU 0 also renders shaders/display and runs JoyAI prompt encoding: it is not
dedicated. SR is output restoration, not native 720p model inference.

## Launch / rollback

From the JoyAI repository, with the existing isolated environment/assets:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tulip/run-service.ps1 -Gpu 1 -TextGpu 0 -Port 7873 -RifeFluxRoot C:/Users/Admin/src/FluxRT -UpscaleGpu 0
```

The launcher maps physical GPUs `1,0` to logical `cuda:0,cuda:1`. It rejects SR
on the generation GPU and checks at least 1 GiB free on the SR GPU. It does not
stop competing workloads. Service access stays at `127.0.0.1:7873` with the
existing local-origin filter. No new listeners or external services.

Local Etherea `.env` adds `JOYAI_UPSCALE_GPU=0`; its runtime supervisor forwards
`-UpscaleGpu`. This machine configuration is not committed. The existing backend
must be restarted normally before a changed environment affects a future launch;
the currently running JoyAI service was launched explicitly with the flag above.
Runtime logs: `tulip/upscale-service-v2-20260912.log` and `.err.log`.

For immediate native-output bypass, use the desktop tile's **SR off**. To disable
the feature at launch, omit `-UpscaleGpu` (default -1), and remove the machine
environment override or set `JOYAI_UPSCALE_GPU=-1`. A model service restart resets
temporal state; SR toggling does not. Restore only this feature's dedicated
commits if reverting code, preserving unrelated work. No environments, caches,
recordings or other services were deleted.

For a JoyAI-only reload while preserving the current tile/prompt/strength:

```powershell
# From etherea-ai; use a NEW snapshot path:
node scripts/smoke-tests/joyai-transition-live.cjs prepare logs/NEW-RELOAD.json
# Stop only the verified JoyAI listener process, then launch the command above.
# After /health.runtime_loaded is true:
node scripts/smoke-tests/joyai-transition-live.cjs resume logs/NEW-RELOAD.json
```

## Verification commands

```powershell
# JoyAI repo: tests use only small output models, not another JoyAI session.
$env:CUDA_VISIBLE_DEVICES='0'
$env:JOYAI_RIFE_FLUX_ROOT='C:/Users/Admin/src/FluxRT'
.venv/Scripts/python.exe tulip/test_upscale_pipeline.py
.venv/Scripts/python.exe tulip/test_rife_output.py 2
.venv/Scripts/python.exe tulip/test_stream_controls.py

# Etherea repo: test the real current tile; scene uses its existing prompt.
node scripts/smoke-tests/joyai-transition-live.cjs sr-off logs/NEW-OFF.jsonl
node scripts/smoke-tests/joyai-transition-live.cjs sr-on logs/NEW-ON.jsonl
node scripts/smoke-tests/joyai-transition-live.cjs rife-off logs/NEW-RIFE-OFF.jsonl
node scripts/smoke-tests/joyai-transition-live.cjs rife-on logs/NEW-RIFE-ON.jsonl
node scripts/smoke-tests/joyai-transition-live.cjs scene logs/NEW-SCENE.jsonl
node scripts/smoke-tests/joyai-upscale-summary.cjs logs/NEW-ON.jsonl
node scripts/streaming/whep-stats-probe.js --whep-url http://127.0.0.1:8889/hermes-screen/whep --seconds 15 --no-turn --browser-executable "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --out logs/NEW-WHEP.jsonl

# SDK repo: authenticated actual Photo/Video verbs, finishes in Video.
.venv/Scripts/python.exe scripts/primary-mode-smoke.py logs/NEW-SWITCH.json
node --test scripts/ndi-geometry.test.mjs scripts/primary-feed.test.mjs
```

Do not run standalone GPU tests during a measured live interval. Observer scripts
save timestamps/PNGs; the scene test also saves a WebM. These are local canvas/NDI/
WebRTC checks, not measured iPhone capture-to-screen latency. They do not move
the mouse, navigate the desktop, change creative prompts or start Flux and JoyAI
generation concurrently. Operator shader changes during trials are not suppressed.

## Measured results and evidence

Final post-switch 35-second observation (first four samples excluded, 30.4s
measured), `etherea-ai/logs/upscale-verified-steady-20260912.jsonl`:

| Metric | Result |
|---|---:|
| Native model frames displayed | 16.12/s |
| RIFE presentation frames displayed | 32.27/s |
| Frames submitted to production NDI | 31.74/s |
| Output size | 1296x720 |
| Local canvas callback gap p50 / p95 / max | 31 / 42 / 50ms |
| Browser source-sample -> canvas latency p50 / p95 | 1093 / 1128ms |
| SR cost per 16-presentation-frame chunk, sampled p50 / p95 | 176.6 / 186.7ms |
| Final JPEG cost per chunk, sampled p50 / p95 | 48.2 / 55.7ms |
| SR + final JPEG per chunk, sampled p50 / p95 | 226.6 / 237.3ms |
| Capture tick drops / output queue drops in measured window | 0 / 0 |

These are actual paced model/presentation counts, not a claim of 32 native
generation FPS or maximum DiT throughput. Stage wall time includes synchronized
SR D2H output and JPEG encoding; it excludes the earlier native RIFE pass and
later WebSocket/NDI/WebRTC delivery. Timings are sampled once per second from
chunk profiles. The NDI sender can drop submissions under its existing 8 MB
socket backpressure threshold; presentation FPS is not proof of delivery FPS.

The original no-SR baseline was 16.11 native / 32.26 presentation FPS, source-to-
canvas p50/p95 929/936ms. Native bypass in the new raw-frame pipeline was 16.11 /
32.26 FPS and 1024/1026ms. Another warm SR interval was 1268/1307ms. Treat the
added local latency as roughly 0.1-0.4s on these samples, not one guaranteed
number: scenes and operator strength changed across trials. No phone-display
latency or native Easel shader GPU frame-time percentiles were measured.

Final real WHEP receive (`upscale-verified-whep-20260912.jsonl`) was 1296x720,
30 decoded FPS in warm intervals, 0 packet loss, 0 decoder drops, 0 reported
freeze events in 15 samples. Maximum RTP jitter 16ms; maximum observed video
callback gap 116.7ms. This is a local receive-only peer on the actual published
feed, not a claim about remote Wi-Fi/iPhone screen behavior. The chandelier's
separate 64x64 rendition is unchanged.

The real SDK repeat `easel-agent-sdk/logs/upscale-primary-switch-v2-20260912.json`
passed: Photo 8.812s, Video 5.781s, 41 new distinct Flux outputs during the Photo
observation with no JoyAI session; 170 new JoyAI presentation outputs during Video
with Flux pause acknowledged and its output counter unchanged. New tile startup
retained 720p. These switch times exclude final phone/WebRTC presentation handover.

SR off/on and RIFE off/on preserved epoch 17. The same-prompt scene test moved
17 -> 18, returned generated frames, and used 14 old-scene frames plus 44 live
shader-bridge frames. Maximum callback gap was 685ms during that reset/preparation
test: continuity is still not seamless. See `upscale-scene-20260912.jsonl.webm`
and its PNG for actual 720p scene-transition video.

Inspected the real canvas PNGs before/after switching: butterfly meadow and
subsequent operator/Cue insect imagery with strong magenta/purple shader blending.
Output is full-frame, colored and nonblank; SR produces clean edges but can
smooth texture and does not repair warped geometry. The scene WebM contains
actual moving generated/interpolated output, not a mock or copied source clip.
This inspection does not establish universal temporal shimmer-free restoration.

Rejected/intermediate results are preserved, not counted as final passes:

- Initial serial JPEG integration: ~14.07 native FPS; final encoding ~159ms/chunk.
  Bounded four-worker encoding reduced this to ~48ms; output-stage input credit
  then restored ~16 native FPS. Initial v2 observation overlapped short GPU tests.
- First SDK switch test failed its five-second Flux distinct-output assertion.
  The repeat passed within five seconds; the test now preserves failure samples
  and permits bounded extra warmup observation, without removing the assertion.
- First return to Video instantiated a cached `scene-rife-1` browser module,
  leaving a cropped 360p canvas despite 720p service output. The import is now
  versioned `upscale-1`; the running manager/client was migrated and a real
  repeat switch plus WHEP dimension check passed. The intermediate 360p WHEP
  file `upscale-final-whep-20260912.jsonl` is NOT the final 720p pass.

Final checks: 45 browser tests across JoyAI/lifecycle/video-mode/scene suites;
12 runtime-supervisor tests; 9 session-control tests; two raw SR pipeline tests
(including real GPU RIFE/SR, reset, bounds, deterministic native originals and
simulated SR failure); legacy real-GPU RIFE byte-preservation smoke; three Node
NDI-geometry/primary-feed tests; Python compilation and Git whitespace checks.

SR model construction/load in the resident service: .178s; explicit one-frame
warmup .070s (warm driver/kernel caches), separate from the roughly 2-3 minute
JoyAI checkpoint reload. Both weights remain resident across Photo/Video switches.
No new peak-process-VRAM measurement was made for this integrated run; isolated
SR allocation measurements are in the trial report and must not be substituted
for total live-process VRAM.
