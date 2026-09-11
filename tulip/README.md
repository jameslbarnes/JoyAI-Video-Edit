# JoyAI trial on Tulip (2026-09-10)

## Status

Shader slider: direct FluxRT blend units (zero neutral; no .005 cap). See
`SHADER_STRENGTH.md` for the new scale, migration and high-strength caveats.

Latest continuous preset: `presets/etherea-continuous-v2.json`. The former temporal
position limit of 128 caused long-run visual corruption; the corrected limit 8
passed 2,049 fixed-prompt frames. Hot prompts now preserve temporal state. See
`CACHE_DIAGNOSTICS.md` for ablation evidence, live limitations and launch commands.
The original v1 preset and short approved video below are historical baselines.

**Primary workstream:** the user approved `evidence/shader-graphs-only/output.mp4`.
The exact preset is frozen in `presets/etherea-coral-v1.json`. A real persistent
WebSocket service and initial Etherea provider integration now pass paced input,
prompt-reset and reconnect tests. See the sibling Etherea repository's
`docs/JOYAI_STREAMING.md` for current status, live measurements, launch commands
and remaining production/NDI/SDK verification. The paragraphs below describe the
earlier offline installation milestone, not the current integration status.

Real persistent V2V generation and frame inspection succeeded: 33 shader frames
and 97 structured-motion frames. See `evidence/shader-valid` and
`evidence/structured-valid`. This is not yet an Etherea provider integration or a
paced live-input benchmark. FluxRT and existing application environments have not
been modified. Optimization and LongLive/OPSD-V comparisons are tracked separately.
The idle SANA service on port 7871 was gracefully unloaded to free GPU 0.
All selected model artifacts finished downloading and passed checksum verification.
The first load was deliberately interrupted before inference to tighten the VRAM
guard: WDDM's CUDA free-memory value exceeded `nvidia-smi` physical free memory.
The corrected harness takes the smaller value. This was not a model-quality test.

Verified: Windows 11 build 26200, two RTX 5090 32 GB GPUs, no installed WSL,
existing VS 2022 Build Tools 14.44.35207. Python 3.12.10. GPU 0 has approximately
22 GiB free after unloading SANA; GPU 1 hosts FluxRT. C: had 374.49 GiB free before
this trial. No recordings or pre-existing caches were deleted during this trial.

## Sources and isolation

- Source: https://github.com/jd-opensource/JoyAI-Video-Edit
- Source commit: `69fdaea89e2fd7708435790d73eb30989e722a5f`
- Follow the checked-out `DEPLOYMENT.md`, not older cached search summaries.
- Installation root: `C:\Users\Admin\src\joyai-video-edit`
- Dedicated Python: `.venv\Scripts\python.exe`
- Repo-local CUDA compiler and development libraries: `.toolchain\Library`
- Required JoyAI revision: `39491dda89fd2b535777fd8644a6631ea89b857c`
- Required MiMo revision: `4bfb270765825d2fa059011deb4c96fdd579be6f`
- Model files and checksums: `models-manifest.json` (incrementally written; a
  partial manifest does not mean the download is complete).
- Model locations: `tulip\models\JoyAI-Video-Edit` and
  `tulip\models\MiMo-VL-7B-RL-2508`.
- The older duplicate 0804 DiT checkpoint and optional face/body detectors are
  deliberately omitted. No compatible JoyAI/MiMo model cache was found in the
  standard existing Hugging Face hub cache.

No driver, WSL, system CUDA installation, global PATH change, pagefile change, or
package installation into the Etherea/FluxRT/Scope/SDK environments was made.
The conda executable is used only to create a separate prefix, not alter base.

## Reproduce the environment

PowerShell, from the installation root, after applying the recorded compatibility
changes in this checkout:

```powershell
uv venv --python C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe .venv
uv pip install --python .venv/Scripts/python.exe -r tulip/requirements-windows.txt
C:\Users\Admin\miniconda3\Scripts\conda.exe create --yes --prefix C:\Users\Admin\src\joyai-video-edit\.toolchain --override-channels -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8 libcublas-dev libcusparse-dev libcusolver-dev
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tulip/build-native.ps1
.venv/Scripts/python.exe tulip/probe-native.py
```

The toolchain has nvcc 12.8.93. `build-native.ps1` imports the existing MSVC
environment into that process only, sets CUDA library/include paths for conda's
layout, and builds sm_120a kernels. The default light extension excludes CUTLASS
FP8 kernels. The required GeForce `JOYOMNI_FP8_FAST_ACCUM=1` path instead uses the
upstream Triton quantizer and GEMM. Do not turn fast accumulation off with this
light extension: the alternative FP8 C++ operators are not installed.

Verified native numerical probe: RMSNorm, norm modulation, identity-RoPE Q/K
normalization, and actual FP8 linear execution on GPU 0. FP8 linear relative RMSE
against BF16 was 0.03694 on the small seeded probe, not a model-quality score.
Results: `native-probe.json` (expanded weighted-rotation probe also passed;
its FP8 relative RMSE was 0.03648). Installed versions: `packages.json`.

SageAttention 2.2.0 subsequently built successfully on native Windows. Eager and
CUDA-graph tests passed, including changing all inputs between graph replays:
relative RMSE 0.03635 / 0.03635 / 0.03712 against FP32 SDPA. See `sage-probe.json`.
These are kernel correctness probes, not full-model generation or speed results.

For SageAttention (build and CUDA-graph verification are a separate gate):

```powershell
git clone https://github.com/thu-ml/SageAttention.git deploy/tmp/SageAttention
git -C deploy/tmp/SageAttention checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
git -C deploy/tmp/SageAttention apply ../../sageattention-cudagraph-stream.patch
# Apply the Windows changes recorded in this trial: MSVC flags and narrow
# torch/types.h includes in CUDA sources/headers (retain torch/extension.h in pybind C++).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tulip/build-native.ps1 -Sage
.venv/Scripts/python.exe tulip/probe-sage.py
```

Do not repeat `git apply` on an already patched Sage checkout. The current-stream
patch is required for correct CUDA-graph replay. FlashAttention 4 is omitted per
the upstream RTX 5090 instructions.

## Download only the selected artifacts

```powershell
.venv/Scripts/python.exe -u tulip/download-models.py *> tulip/download-models.log
```

Downloads use pinned revisions, bounded 16 MiB requests, resumable range files,
and SHA256 verification against upstream LFS metadata. Keep `.partial` and
`.ranges.json` together to resume. Approximately 51 GB of model artifacts are
required. The final log must say `All required artifacts verified`.

## Bounded shader trial

First ensure GPU 0 is available. Do not stop FluxRT or other services implicitly.
The launcher itself neither starts nor stops existing services. It chooses a GPU
with `CUDA_VISIBLE_DEVICES` and uses only that device. The Python harness limits
its allocator to the smaller of CUDA-reported and physical free VRAM minus
0.75 GiB and reports an OOM if it cannot
fit. It does not change system GPU-memory/paging policy.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tulip/run-trial.ps1 -Gpu 0 -Frames 33 -Width 648 -Height 360 -OutputDir tulip/evidence/shader-new *> tulip/trial.log
```

Input is the existing real Easel shader recording, read-only:
`C:\Users\Admin\src\sana-streaming\tulip\evidence\shader-ab\input.mp4`.
The source is 640x352 at 16 FPS, 97 frames. The first test takes 33 frames.
Seed 42, two denoising steps, FP8 DiT image/text linears, BF16 activations/VAE.
The prompt requests a detailed bioluminescent alien coral transformation.

This calls the upstream persistent streaming session API, not repeated offline
inference. The first chunk takes one input frame; subsequent chunks take eight
frames. At most two chunks are in flight. The initial correctness trial disables
Torch compilation and DiT CUDA graphs; it must not be presented as the optimized
24-FPS configuration advertised upstream. VAE warmup still runs and is included
in the separately labeled load/warmup timing. A later `-Optimized` run enables
compilation/graphs, subject to successful backend validation and enough VRAM.

JoyAI requires dimensions divisible by **24**: 648x360 is valid; the initially
attempted 640x352 is not. The harness validates this before model loading and
refuses to overwrite an output directory containing metrics.

Evidence defaults to `tulip\evidence\shader-valid`: `output.mp4`, representative PNGs and
`metrics.json`. MP4 playback follows the source FPS, with no interpolation or
repeated-frame speedup. JPEG delivery quality is 95. The harness records per-chunk
CUDA-event stage timings when supplied by upstream, time to first generated JPEG,
PyTorch allocated/reserved peaks, and submission-to-JPEG intervals. This is an
unpaced bounded API test, **not capture-to-display latency or a live demo**.
Successful generation must still be visually reviewed; a file's existence is
insufficient. Current evidence status explicitly distinguishes these stages.

Structured-motion comparison and evidence inspection:

```powershell
.venv/Scripts/python.exe tulip/make-control.py
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tulip/run-trial.ps1 -Frames 97 -InputVideo C:\Users\Admin\src\joyai-video-edit\tulip\evidence\structured-input\input.mp4 -OutputDir tulip/evidence/structured-new
.venv/Scripts/python.exe tulip/review-video.py --input tulip/evidence/structured-input/input.mp4 --output tulip/evidence/structured-new/output.mp4
.venv/Scripts/python.exe tulip/summarize-trial.py tulip/evidence/structured-new/metrics.json
```

Optimized trial (separate evidence directory):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tulip/run-trial.ps1 -Frames 97 -Optimized -OutputDir tulip/evidence/shader-optimized-new *> tulip/trial-optimized.log
```

The optimized launcher imports existing MSVC Build Tools into its child process.
`TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=0` avoids Windows path-length failures in long
generated Triton kernel names. The first optimized attempt was stopped after
these failures; its directory is explicitly marked `ABORTED.md`. Do not use its
incomplete metrics as a speed result. Compilation and graph warm-up are separate
from warm generation throughput.

The short-name attempt compiled both landscape decode shapes successfully but
was stopped after more than ten minutes of startup before all remaining warm-up
shapes completed. Full VAE compilation is not fully verified. The successful
practical partial optimization is **DiT graphs with eager VAE**:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tulip/run-trial.ps1 -Frames 97 -GraphsOnly -OutputDir tulip/evidence/shader-graphs-new *> tulip/trial-graphs.log
```

Measured `evidence/shader-graphs-only`: 97 actual frames; warm delivered throughput
16.43 FPS (88 frames over 5.36 seconds after two initial chunks); load including
VAE warm-up 157.76 seconds; first JPEG 4.86 seconds after starting submission.
Warm local submission-to-JPEG p50/p95: 970.8/1159.7 ms; PyTorch peak
allocated/reserved: 19.44/21.02 GiB. The graph-stage 25.15 FPS includes scheduler
arithmetic and clean-KV storage/commit, **not pure DiT-only throughput**.
Graph capture succeeded in 1.5 seconds and `graph_path=1` was verified on warm chunks.
The original uncompiled structured run delivered 16.58 warm FPS; these different
inputs are not a controlled end-to-end graph speedup comparison.

Visual findings: the shader becomes textured branching pink/cyan/orange coral,
with persistent forms, later warping and increasingly painterly/soft details.
The structured input keeps recognizable moving cube/sphere objects, adds coral
scenery and later occludes them rather than reliably replacing their materials.
Every output frame decoded; no exactly identical consecutive frames. Sequential
frames and chunk boundaries were inspected, not a desktop playback session.
See per-run `comparison-*.jpg`, `decode-audit.json` and `benchmark-summary.json`.

## Compatibility changes and rollback

Changes are confined to this new checkout:

- `deploy/joyomni_ops/setup.py`: Windows C++ flags.
- `deploy/joyomni_ops/{include/joyomni_ops.h,csrc/fused_qknorm_rope_3d_kernel.cu}`:
  narrow Torch headers to avoid NVCC/MSVC's ambiguous `std` error.
- `deploy/joyomni_ops/csrc/pybind.cpp`: Windows module export for the linker.
- `deploy/joyomni_ops/joyomni_ops/__init__.py`: load Windows `.pyd` extension.
- `deploy/xvideo/serving/serve_joyomni_streaming.py`: asyncio on Windows instead
  of unavailable uvloop.
- SageAttention sub-checkout: upstream stream fix, MSVC flags, narrow headers.
- `tulip/`: isolated build, download, probe and trial scripts, logs and manifests.

The foreground trial exits and releases its GPU memory on completion/error. Use
Ctrl+C in its owning terminal to cancel; do not kill unrelated Python processes.
No JoyAI background service has been launched by these commands. To restore idle
SANA, use its existing `C:\Users\Admin\src\sana-streaming\tulip` launch
documentation after the JoyAI process has exited. No Etherea/SDK provider changes
need rollback. Keep the isolated directory for reproducibility; removal is not
performed automatically and should target only this verified installation root.
