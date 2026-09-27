# GPU-only interpolation comparison, September 23, 2026

Subsequent deployment: [GPU-resident RIFE live acceptance](RIFE_GPU_LIVE_20260923.md).
The optional GPU RIFE handoff is now enabled at 648p; DLSSG remains an offline
quality candidate. Statements below describe the earlier offline experiment.

DLSSG can beat RIFE on some motion. The broader comparison found a substantial
DLSSG advantage on controlled occlusion and moving thin lines, while RIFE stayed
closer to the organic JoyAI reference. Neither is a universal quality winner.
Removing the intermediate GPU-to-CPU-to-GPU transfer benefits both engines.

These were offline recorded-frame experiments. Live remains 1152x648 at 12
native fps, RIFE 2x, VSR Ultra, and 2592x1440 NVENC output at 24 fps. No live
engine, settings, prompt, publisher, or installed Python packages were changed.
The new optional adapter paths are implemented but not enabled by live callers.

## Output-stage performance

Same 64 recorded native inputs, two passes, eight-input chunks, VSR Ultra to
2592x1440, H.264 NVENC P4 ULL at 15 Mbps / 24 fps with no B-frames. Warmup is
excluded. Trials were paced under concurrent live GPU load, so these are
indicative measurements rather than isolated hardware limits.

| Path | Median per eight inputs | p95 per eight inputs |
|---|---:|---:|
| Existing RIFE CPU-output path | 317.8 ms | 394.9 ms |
| GPU-only DLSSG -> VSR -> NVENC | 172.1 ms | 262.5 ms |
| GPU-only RIFE -> VSR -> NVENC | 150.5 ms | 250.5 ms |

GPU-only refers to the interpolation/output chain after initial recorded-input
upload. Raw frame downloads were prohibited during the timed GPU paths; encoded
bytes leave the GPU. The timings include interpolation, VSR, and NVENC, excluding
JoyAI generation, CPU preview/NDI, disk writes, and trial pacing. They do not
establish an equivalent improvement in end-to-end live latency or native fps.
Each final run encoded and independently decoded 254 frames at the correct
geometry and 24 fps. The harness flushes NVENC at completion.

DLSSG showed lower device-wide memory usage than RIFE in these runs. Concurrent
live allocations and different initialization boundaries prevent an exact
per-engine VRAM saving claim.

## Quality

Every other recorded frame was withheld as a midpoint reference. Both methods
received identical inputs and preserved originals exactly. Metrics below are
means across withheld frames; higher PSNR/SSIM and lower RGB MAE are better.

| Content | RIFE PSNR | DLSSG PSNR | RIFE SSIM | DLSSG SSIM | RIFE MAE | DLSSG MAE |
|---|---:|---:|---:|---:|---:|---:|
| Organic JoyAI | 29.673 | 28.716 | 0.92946 | 0.92560 | 4.262 | 4.589 |
| Porcelain JoyAI | 27.442 | 27.201 | 0.91947 | 0.92083 | 5.286 | 5.335 |
| Procedural shader | 37.663 | 36.479 | 0.99294 | 0.99324 | 1.246 | 1.330 |
| Controlled occlusion/thin lines | 25.838 | 30.659 | 0.96282 | 0.98147 | 3.257 | 1.702 |

Representative and high-error stills were inspected. DLSSG preserved moving
lines that RIFE visibly broke in the controlled case, but it also had failures
on some frames. Both showed warping/double contours in difficult organic
regions. Porcelain and shader outcomes were close and metric-dependent.
These metrics and selected stills do not settle perceptual flicker or preference
on a television; side-by-side videos are saved for that review.

The organic source was 1248x720/16 fps, center-cropped and reduced to 1152x648
(129 references, 64 withheld). Porcelain and shader sources were lower resolution
and enlarged to 648p (97 references, 48 withheld each); they test content motion,
not newly captured 648p detail. Recorded clips are retimed to 24 fps, but input
spacing corresponds to 8 fps in source time. The synthetic occlusion case is
true 1152x648/24 fps with 12 fps inputs and 48 withheld references. Four samples,
including one synthetic sample, are insufficient for a universal ranking.

Newness alone cannot establish superiority for this integration. Standard
[NVIDIA DLSSG integration](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_G.md)
uses depth, motion vectors, HUD-less color, and camera/frame constants. This
community video wrapper estimates motion from images with NVIDIA optical flow;
it does not receive our shader's geometry or motion buffers. Its result must be
judged on our actual content and inputs.

## Implementation and verification

- `deploy/xvideo/serving/vsr_output.py`: accepts CUDA RGB uint8 tensors and
  `GpuFrame` objects, waits for producer readiness, and retains the existing VSR
  synchronization and owned-output behavior. NumPy input remains supported.
- `deploy/xvideo/serving/rife_output.py`: optional `gpu_output=True` with raw
  output, preserving GPU frame storage and readiness events. Default behavior
  remains unchanged.
- `dlssg_gpu_bridge.py`: persistent DLSSG session and refcounted CUDA/DLPack
  surface handoff to Torch/VSR without generated-frame readback.
- `verify_dlssg_gpu.py`: exact native preservation, exact VSR tensor/NumPy
  agreement, no raw downloads, retained DLSSG/VSR buffers, invalid-input checks.
- `verify_rife_gpu.py`: exact pixels/metadata for 2x, 4x, and 16-frame scene
  morphs; retained original/generated storage; raw downloads prohibited.
  Deterministic cuDNN is enabled only in this transport verification to avoid
  autotuning variability observed even between repeated CPU-output runs.
- `benchmark_dlssg_gpu_pipeline.py`: performance and decoded-output validation.
- `dlssg_quality_suite.py`: reproducible source preparation, inference, metrics,
  comparison videos, and selected comparison frames.

Hardware: physical output GPU 0, RTX 5090. Torch 2.9.1+cu128 / Python 3.12.10.
DLSSG uses Visual Enhancer v11.0, bridge 1.3.0, ABI 3, runtime 310.9, with NVIDIA
optical flow and CUDA interoperability. PyAV 18.1.0 is installed only in
`research/dlssg-python`; live PyAV 13.1.0 was left intact. The older PyAV failed
the bridge's FFmpeg CUDA primary-context initialization.

During a 249.848-second observation, live delivery measured 11.995 native fps,
23.991 browser canvas fps, and 23.999 publisher fps. There were two browser
output drops, zero GPU publisher drops/underruns/gaps, no client errors, and
median preview latency of 1791 ms. This was not an independent WHEP receiver
test. Final read-only checks found healthy workers, no current publisher error,
and zero allocator OOMs on either device; lifetime counters are separate from
the observation-window deltas.

## Evidence and next decision

All evidence is under [evidence/dlssg-gpu-20260923](evidence/dlssg-gpu-20260923).
See `pipeline-summary.json`, both interop verification JSONs, and
`quality/summary.json`. Quality videos show reference / RIFE / DLSSG:

- [Occlusion and thin lines](evidence/dlssg-gpu-20260923/quality/occlusion-comparison.mp4)
- [Organic JoyAI](evidence/dlssg-gpu-20260923/quality/organic-comparison.mp4)
- [Porcelain JoyAI](evidence/dlssg-gpu-20260923/quality/porcelain-comparison.mp4)
- [Procedural shader](evidence/dlssg-gpu-20260923/quality/shader-comparison.mp4)

The immediate performance candidate is the GPU-only RIFE path, which preserves
the current interpolation model and was fastest here. DLSSG remains a serious
quality candidate. A live integration must also preserve CPU preview/NDI,
startup/reset behavior, scene transitions, and buffer ownership; the offline
output-stage timings do not qualify those live behaviors. No live replacement
or 3x/native-resolution change was made in this experiment.
