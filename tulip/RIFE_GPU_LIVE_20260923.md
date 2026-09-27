# GPU-resident RIFE handoff at 648p

User-approved integration of the verified GPU-only output path. Native generation
stays 1152x648 / 12 fps, RIFE stays 2x, VSR stays Ultra, and NVENC stays
2592x1440 / 24 fps / 15 Mbps. DLSSG is not selected by this change.

## Behavior

The streaming and prefetched presentation paths request GPU RIFE output when
`rife_gpu` is enabled in `output-presentation.json`, VSR is active, and both
engines use the same GPU. RIFE output events and owned storage pass directly to
VSR. The existing CPU NDI/JPEG branch downloads its final resized pixels after
VSR; it does not cause an intermediate RIFE frame download/re-upload.

Configuration changes are applied at bounded chunk boundaries. Already-prepared
GPU chunks remain valid after a manual toggle. A consumer that needs CPU input
(for example an incompatible enhancement order) explicitly downloads that bounded
chunk; health diagnostics report `rife_gpu_handoff.input_download_bytes`.
Compatible GPU handoff reports `gpu_input: true` and zero input download bytes.

## Manual revert

From this repository, run:

```powershell
.venv/Scripts/python.exe tulip/set_rife_gpu.py off
```

This restores CPU RIFE output after the bounded in-flight chunks drain. It does
not restart JoyAI, replace the scene, alter cadence, or restore old source files.
Use `on` to re-enable it. No automatic rollback was added.

The complete pre-change source/settings snapshot and checksums are in
`evidence/rife-gpu-live-20260923-170650/before` and `before-manifest.json`.
Existing immutable 648p favorites remain untouched.

## Verification

- 27 presentation, output, pacing, configuration, and prompt-control regression
  tests passed.
- `verify_rife_gpu_presentation.py` exercised the production presentation path
  on the output GPU with local publisher collectors: 15 frames, zero intermediate
  raw downloads, exact GPU/NDI/JPEG agreement with the CPU-input path, 15 expected
  NDI/preview downloads, safe manual toggle, and consumer-change fallback.
- Prior `verify_rife_gpu.py` covered exact pixels/metadata for 2x, 4x, scene morph,
  and retained frame ownership.
- Baseline live observation: 62.948 seconds, 11.994 native fps, 23.988 browser
  canvas fps, 23.909 publisher fps, median preview latency 1790 ms, zero new
  publisher drops/underruns/gaps or client errors.

## Live acceptance

GPU RIFE handoff is now live. The deployment used the existing JoyAI lifecycle
and preserved the current prompt and strength exactly. Only JoyAI restarted;
no dependency packages or other application services were changed.

- Manual on/off/on: same live session and NVENC publisher throughout, no
  underruns, and GPU handoff enabled at completion.
- Prompt changes: 41.399-second window, 24.010 canvas fps, no new drops,
  underruns, delivery gaps, or client errors. Original prompt restored.
- Same-content scene transition: epochs 1 -> 2, 861 observed frames, including
  29 fade-out frames and 38 live-shader bridge frames; no client errors. Publisher
  stayed on the same process with no restart.
- Steady window after transition: 73.295 seconds, 12.006 native fps, 23.999 canvas
  fps, 23.931 publisher-counter fps, no drops/underruns/gaps or client errors.
- Independent WHEP receiver: 149.261 seconds, **24.005 decoded fps at 2592x1440**,
  zero freezes, dropped frames, or lost packets, including the scene change.
- Median capture-to-browser-preview latency: 1713 ms, versus the pre-deployment
  1790 ms. This is an indicative 77 ms improvement, not a TV latency measurement
  or an isolated causal benchmark; generation and fixed playback buffering remain.
- Both GPUs: zero OOMs. Model GPU had one allocation retry after scene replacement;
  output GPU had zero. No current upscaler, NDI, or publisher error.
- The publisher's lifetime record includes one 84.2 ms delivery gap before the
  transition/steady test windows (42.97 ms pipe write with 12 frames queued).
  There were no lifetime publisher drops or underruns and no receiver freezes.

The initial independent receiver attempt used an unreachable localhost STUN
endpoint and failed ICE. Repeating with the previously working STUN setting
connected successfully, without a renderer change. Both probes have exited.

Evidence: `evidence/rife-gpu-live-20260923-170650/final-summary.json`, with raw
observations, receiver stats, source/settings snapshots, and the recorded scene
transition alongside it. The earlier offline 318 -> 151 ms output-stage benchmark
excludes CPU preview/NDI and JoyAI generation; it must not be presented as the
measured full live latency improvement.
