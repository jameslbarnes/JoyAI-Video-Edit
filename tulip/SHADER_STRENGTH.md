# JoyAI shader influence — September 11, 2026

## Direct Flux-unit slider (supersedes the mapping below)

The runtime now uses `blend = value`, `referenceGain = 1`, neutral/default **0**.
10% means a 0.10 source/noise latent blend, using the same initialization equation
as FluxRT. The API accepts 0..1; desktop uses Flux's 0.01..0.99 range with .01
steps, and iOS source uses Flux's 0..0.30 dial range. There is no overlay, midpoint
remapping, or .005 cap. Output metadata identifies `mode: direct_latent_blend`.
The .125-per-chunk smoothing and cache-preserving updates remain unchanged.

This is matching units/operation, NOT matching model quality. Existing ablations
show that .10 and above can approach raw-shader copying. Users can now explore
that range explicitly. No new claim of Flux-equivalent output is made.
Deploying this Python change requires a model reload and clears temporal history;
do not reinterpret an old .85 target as .85 blend during migration. Start at zero
or explicitly select a small direct blend after reconnecting. Zero restores the
baseline parameters but does not erase already-generated history.

The following sections record the **superseded normalized-dial experiments**.

The bounded shader-influence control is now implemented in the runtime, Etherea,
and SDK. The iOS source is updated but still requires a signed Mac/Xcode build.
This is an experimental artistic control, not calibrated Flux-equivalent adherence.

## Current control

`model.primary.strength.set {"value": 0.0..1.0}` uses a neutral midpoint of **0.5**.
Below neutral, reference latent gain decreases toward .75; above neutral, a
quadratic source/noise initialization blend increases toward .005. These are
model inputs, not a source/output crossfade. 100% on the dial does NOT mean a
100% source blend. The original inference path is unchanged at neutral.

The model advances by at most .125 per chunk, with one latest-wins target.
A midpoint-to-endpoint change takes four 8-frame chunks (~2s at 16fps), plus
pipeline/delivery delay. It neither resets KV/VAE history nor recaptures graphs.
Create scene and reconnect still reset history. Returning to neutral restores
the original parameters, not the earlier scene's pixels. Runtime restart resets
the dial to neutral; session resets retain its target.

Loopback service: GET/POST `http://127.0.0.1:7873/strength`; POST body `{"value":.5}`.
Etherea proxy: GET/POST `/api/joyai/strength`. WebSocket: `set_strength` ->
`strength_queued` / `strength_error`. Output profiles report actual applied
`shader_strength` including revision, target, value, settled, blend and gain.
An HTTP acknowledgement is not proof of generated output with that setting.

## Fine sweep and selected range

`tulip/evidence/strength-fine-20260911`: 23 arms, 6,935 generated frames, two
captured shaders and two prompts (coral and cobalt/ivory porcelain). All 23 MP4s
fully decoded; zero exactly repeated adjacent frame pairs. This is a diagnostic,
not a perceptual quality score. Model load was 104.53s, separate from inference.

Porcelain .001/.003/.005 blends retained clear expressive ceramic faces;
.01 already distorted the face. Coral tolerated more, so the production cap
uses the stricter .005 result. Gain .75 remained coherent but changed scale and
composition; monotonic visual adherence across arbitrary prompts is unproven.
The larger .03 experimental ramp is NOT the deployed dial range.

`porcelain-comparison.mp4` shows neutral / .003 / .005 left to right. Each arm
uses persistent temporal state; they are separate sessions, not one slider sweep.
Seed alone does not ensure pixel-identical VAE sampling across sessions.

Reproduce the fine sweep only while the live model is paused:

```powershell
cd C:\Users\Admin\src\joyai-video-edit
.venv/Scripts/python.exe tulip/test_shader_dial.py
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tulip/run-cache-ab.ps1 -Gpu 1 -OutputDir tulip/evidence/NEW-FINE-RUN -InputVideo C:/Users/Admin/src/sana-streaming/tulip/evidence/shader-ab/input.mp4 -Jobs tulip/strength-fine-jobs.json
```

Use a new output directory. Jobs record the exact local input paths and prompts.
No new dependencies or model artifacts were installed for this control.

For the live SDK commands, delivery fixes, evidence and measured limitations,
see sibling `easel-agent-sdk/docs/JOYAI_SHADER_STRENGTH.md`.

## Measured outcome

Corrected sweep: seven arms x 513 frames (3,591 generated frames total). Every
MP4 fully decoded; no adjacent frames were exactly identical, and all inspected
decoded tensors were finite. Representative early/late and ramp-boundary frames
were visually inspected. Input is a 97-frame repeating shader, not a camera feed.

| Setting | Visual finding | Offline generated FPS including session initialization |
|---|---|---:|
| Neutral | Detailed branching coral; baseline retained | 31.75 |
| Blend .10 | Almost raw shader; coral editing largely disappears | 33.82 |
| Blend .25 | Almost raw shader, not a useful intermediate | 33.74 |
| Blend .50 | Almost raw shader, not a useful intermediate | 33.51 |
| Reference gain .75 | Recognizable coral, different branch scale/composition | 33.22 |
| Reference gain .50 | Recognizable finer coral clusters, altered composition | 32.73 |
| Blend ramp 0 -> .25 -> .5 -> 0 | Coral -> shader-like -> coral without resetting temporal history | 32.50 |

Initial images from the blend arms looked nearly frozen. Full decoding found
zero exact repeated adjacent frames; comparison against the input instead showed
near-copying of the source (mean absolute RGB distance 3.89–5.23/255, versus 95.50
for neutral). Pixel distance supports that observation, but is not an adherence
or quality score. We do not conclude that reference gain is monotonic from this
single prompt/input example. Neither lever is a calibrated user-facing dial yet.

Cold runtime load was 108.18 seconds, separate from the table. Largest sampled
PyTorch allocated memory was 19,299.54 MiB; this is chunk-sampled, not a process
peak. All arms used RTX 5090 physical GPU 1. GPU 0 retained other workloads.
These offline figures do not claim live application or iOS latency/throughput.

Decision from this coarse sweep: reject the .10–.50 initialization range. The
subsequent finer sweep above established the much smaller deployed range.

## Controls under test

- `blend`: linearly interpolate the starting noise toward the normalized source
  video latent, before the existing two-step denoising schedule. Analogous to
  the local FluxRT image-latent initialization patch, but not assumed equivalent
  for this distilled model. Source reference tokens still remain present.
- `reference_gain`: multiply reference video latents before their token
  projection. This is NOT attention scaling or trained classifier-free guidance.
  Lower gain is not guaranteed to produce less adherence or better quality.
- Neutral `(blend=0, reference_gain=1)` is an exact no-op in the helper.

Both operations are out-of-place, before copying into CUDA graph input buffers.
No graph recapture, token-count change, or KV/VAE reset is introduced. Generated
history can still retain the effect of previous strength values: returning the
control to neutral is not a fresh-session reset.

## Reproduction

Use the existing isolated environment; install no new packages. Requires at
least 21 GiB free **after** the harness safety margin. Do not run concurrently
with the live model on the same GPU. This sweep uses a user-approved JoyAI pause.

```powershell
cd C:\Users\Admin\src\joyai-video-edit
.venv/Scripts/python.exe tulip/test_strength_controls.py
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tulip/run-cache-ab.ps1 -Gpu 1 -OutputDir tulip/evidence/strength-ab-v2-20260911 -InputVideo C:/Users/Admin/src/sana-streaming/tulip/evidence/shader-ab/input.mp4 -Frames 513 -Arms position8,blend10,blend25,blend50,ref75,ref50,strength_ramp
```

Output directory must not already exist. Each arm uses the same looping
97-frame shader input, coral prompt, seed 42, 648x360, two steps, FP8 DiT/BF16
VAE, temporal-position cap 8 and normal persistent KV. Graphs enabled; no
interpolation, periodic cache resets, or rolling sink. Output playback is 16 FPS.
The 513-frame ramp changes blend 0 -> .25 -> .5 -> 0 every 16 chunks without
restarting its session. Five CPU tests cover boundaries, validation, exact
neutral behavior, no input mutation, and the ramp schedule.

`report.json` records chunk profiles and the actual strength used, and fails if
a strength arm bypasses the instrumented denoise path (including the first
two eager startup chunks before graph replay can reuse history). MP4s and representative
PNGs are saved per arm. Offline elapsed times include per-session initialization;
they are not capture-to-display latency or phone performance. Asynchronous VAE
sampling means a seed alone does not establish exact pixel parity across arms.

Restore the production app through the SDK after the test process exits:
```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:/Users/Admin/src/easel-agent-sdk/primary-model.ps1 -Action select -Provider joyai -Live
```

Rollback: set the live dial to .5 to remove both conditioning modifications
without interrupting output. Explicit Create scene clears any accumulated
history. For full feature rollback, revert only the new shader-control hunks
and module after stopping JoyAI; preserve the unrelated cache/prompt fixes.
The offline harness's temporary session-method patches disappear at process exit.

The first attempt (`strength-ab-20260911`) intentionally failed validation after
baseline and blend10 because the graph-only hook skipped the first eager chunks.
Its modified arm is not valid evidence for a uniformly applied strength setting.
The corrected harness intercepts noise creation inside the single locked DiT
worker and covers both eager startup and graph replay; the temporary patch is
restored immediately after each denoise call.
