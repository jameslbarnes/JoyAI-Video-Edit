# Recorded 648p DLSSG / RIFE comparison

Follow-up: [GPU-only paths and four-content quality comparison](DLSSG_GPU_COMPARISON_20260923.md)
supersede this experiment's single-clip recommendation. Removing the raw CPU
round trip benefits both engines, and DLSSG wins the controlled occlusion case.
The original measurements below remain a record of the CPU-output experiment.

The user approved an offline comparison. The live renderer remains 1152x648 at
12 native fps, RIFE 2x, VSR Ultra, 2592x1440 output at 24 fps. No runtime settings,
prompt, strength, model code, or publisher were changed by this experiment.

## Reproducibility

- `compare_dlssg_rife.py prepare|rife|dlssg|probe|summarize` is the isolated harness.
- Evidence and videos: `evidence/dlssg-rife-20260923`.
- Visual Enhancer v11.0, published September 21, extracted under
  `research/visual-enhancer-v11/app`, using its bundled Python and dependencies.
- Archive SHA-256 verified against the GitHub release digest:
  `47a79f18665981ceab231bd9a5e2a3f79270c1dd22a156eb1bd41f893bbb60f7`.
- DLSSG bridge 1.3.0 / ABI 3 / runtime 310.9; CUDA interoperability and NVIDIA
  optical flow active on physical GPU 0, the output RTX 5090.
- RIFE uses the same FluxRT weights and `RifeOutput` adapter as live delivery,
  FP16, native eight-frame batches, CPU raw output, on physical GPU 0.

Source is archived native JoyAI output at 1248x720/16 fps. Frames 160–288 were
center-cropped to 1248x702 and downsampled to 1152x648. Every other frame was
withheld: both methods received the same 65 input frames, with 64 recorded
midpoints available as references. Both preserved original input frames exactly.
Videos are retimed to 24 fps for viewing; the input spacing corresponds to 8 fps
in the original recording. This is one content sample, not a live native 648p
capture or a general quality benchmark.

## Results

| Measurement | RIFE 2x | DLSSG 2x |
|---|---:|---:|
| Mean held-out PSNR | 29.673 dB | 28.717 dB |
| Mean held-out luminance SSIM | 0.92946 | 0.92560 |
| Mean absolute RGB error | 4.262 | 4.589 |
| Processing with CPU output | 19.251 ms/input, batch mean | 62.043 ms/input, median |

RIFE's seven full steady batches had median 156.455 ms per eight inputs.
Its final partial one-input batch took 689 ms and is excluded from steady
throughput; the raw result JSON includes it in its original aggregate, so use
`summary.json` for the corrected full-batch comparison.

DLSSG returned a GPU surface in median 12.460 ms per input; native stage timings
were median 11.069 ms total, including 8.375 ms NGX and 0.804 ms optical flow.
Downloading the generated RGB surface brought the measured host-output path to
62.043 ms. Thus its core GPU path merits further work, but the current CPU frame
boundary removes its potential speed benefit. GPU-ready DLSSG timing must not
be presented as an apples-to-apples speedup over RIFE's CPU-output timing.

RIFE's process reported peak PyTorch allocation 1339 MiB / reservation 2096 MiB.
Device-wide usage increased about 2108 MiB after RIFE warmup versus 942 MiB after
DLSSG warmup. These device-wide deltas include the concurrently running show;
they are indicative, not isolated per-engine VRAM measurements.

Inspected representative and high-error comparison frames. Both interpolate the
coral/anemone motion plausibly, with local warping/double contours in difficult
regions. RIFE has the numerical advantage on this sample. No claim is made
that it wins on all shader styles or that static frame inspection fully measures
temporal artifacts. `comparison.mp4` shows reference / RIFE / DLSSG side by side.

## 3x probe

The runtime advertised `multi_frame_count_max=5`. An initial strict 3x run
rejected the first post-reset interval because it returned zero generated
frames with `interpolation_disabled=true`. Its partial `dlssg-3x.npy` is not a
valid completed comparison and has no success report.

A fresh-process 17-frame probe clarified startup behavior: generated counts
were `[0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]`; only input 1 reported disabled.
3x therefore works after the initial interval. Any live integration must handle
that startup gap explicitly. 720p at 8 native fps to 24 fps remains a possible
future experiment; it was not enabled or qualified here.

## Live observation and decision

During the main comparison, 185.868 seconds of observation measured 24.028
published fps, 24.001 browser canvas fps, median preview latency 1790 ms, zero
publisher underruns, gaps, dropped frames, or client errors. These measurements
are not an independent WHEP receiver test. Experiments were paced while the
show continued; competing GPU activity makes timing results indicative.

Keep RIFE live. A subsequent DLSSG experiment should connect GPU surfaces
directly to VSR/NVENC and repeat quality checks across more content before a
live replacement. Shader-derived custom masks are a separate neural-rendering
experiment: they control spatial application of enhancement, not transfer of
shader details, and animated masks need additional integration/alignment work.
