# Presentation output resolution

## October 20, 2026: official SDK 1.3 Streaming Ultra trial

The rollback baseline is NVIDIA's `nvidia-vfx` 0.1.0.1 / SDK 1.2.0,
standard Ultra mode 4, with the restored serial GPU preview path. The operator
explicitly approved pausing generation for GPU tests and restarting JoyAI to try
Streaming Ultra (23), strength 1.0. That selection is now in the machine-local
startup file and the live health endpoint confirms SDK 1.3 / mode 23 / strength
1.0. The operator described the projector image as "Sharper and better" and kept
Streaming Ultra selected. Future pauses or restarts still require approval.

Spout requires JoyAI and Easel in the same Windows desktop session. Check the
actual model PID's SessionId before starting generation. On this host the backend
API currently runs in session 0, so its runtime start launched an unusable Spout
session; that load was stopped before resuming. The approved retry uses a hidden
interactive launcher in Easel's session 1. Its script, environment snapshot and
launch record are in the trial evidence directory below. Preserve native stderr
with OS-level file handles: PowerShell 5.1 `*>>` plus `ErrorActionPreference=Stop`
terminated the first interactive attempt on normal server logging. Do not launch
duplicate models or repeat the session-0 start for this Spout show.

The official Windows CPython 3.12 wheel 0.2.0.0 / SDK 1.3.0 is installed separately
under `tulip/research/vfx-0.2.0.0`. The original `vfx-probe` package is retained.
Its SHA-256 is
`cebaa12411f7e22b39293205c5c4d07b8073d14ca1e0e7727ee4375207a36445`.
Do not install over the package loaded by the show or replace native DLLs in-process.

`vfx_runtime.py` selects a pinned release when the VSR adapter is constructed.
An optional `tulip/vfx-runtime.json` accepts `release`, `quality` and `strength`.
The trial uses `{"release":"0.2.0.0","quality":23,"strength":1.0}`. The corresponding
`JOYAI_VFX_RELEASE`, `JOYAI_VSR_QUALITY` and `JOYAI_VSR_STRENGTH` environment
variables override the file. There is no configuration watcher or live DLL swap.
With no overrides, the defaults remain 0.1.0.1 and mode 4. Selection fails on
missing packages, native-version mismatch, or a different already-imported VFX
release. `health.upscale.vsr` reports the selected package, native SDK, mode and
strength once this code is running. This trial does not re-enable the reverted
direct binding, parallel preview preparation, or display-pacing experiments.

The SDK 1.3 adapter sets input geometry before load and uses the indexed device's
default CUDA stream. The old custom-stream adaptation failed initialization or
returned black frames with Streaming Ultra. It also waits for RGB8 export before
DLPack import/copy; outputs remain independently owned. SDK 1.2 retains its former
path for rollback. These are correctness requirements, not asynchronous optimizations.

Private evidence and rollback files are in
`%LOCALAPPDATA%/etherea/quality/20261020-vfx-upgrade/streaming-ultra-trial/`.
The CPU suite passed 76 tests. The production Streaming Ultra adapter passed 64
alternating frames, cross-stream input, retained ownership, geometry changes and
clean teardown on the live GPU mapping. Completed upscaling averaged 4.43 ms,
p95 6.42 ms, maximum 15.02 ms in this paused-show fixture. This is not a display
cadence result. The pre-trial 120-second live baseline already failed cadence:
23.82 received fps, 151 ms maximum gap, 21 missing sender sequence numbers and
four publisher underruns; the largest receipt gap overlapped an editor UI call.
The new 120-second live sample measured 23.86 received fps, 151 ms maximum gap,
17 missing sender sequence numbers and four new local publisher underruns.
Intervals over 62.5 ms increased from 2.36% to 3.36%. Both strict gates failed;
this is an accepted image improvement, not a demonstrated cadence improvement.
Large gaps again overlapped editor UI work, with a recurring five-second pattern.

The original SDK 1.2 fixture also exposed mixed-frame output under its existing
stream ordering. A reference with completed export synchronization passed and was
used for the same-input image comparison. Do not call this proof of the cause of
the live show's jitter. The Streaming result looks crisper on edges and foliage
in that fixed native frame, and the operator preferred the live projector image.
Temporal stability remains unresolved and requires a separate measured fix.

After the operator approves a pause, `tulip/test_vsr_release_live.py` can test
each release in a separate process. It refuses active or unknown JoyAI sessions
before importing Torch and checks again between batches. Use the same actual
1152x648 native image for old mode 4, new mode 4, new HighBitrate Ultra (19), and
new Streaming Ultra (23). It checks cross-stream inputs, output ownership,
alternating frames, geometry changes and completed-work timing, and saves images.
Pixel differences are not perceptual quality scores; fixture throughput is not
projector cadence. A later approved activation must still pass the 120-second
real-show gate below. Frame-time tails and texture stability matter alongside
spatial sharpness.

The operator chose Streaming Ultra to improve soft generated content. Being
uncompressed does not imply a sharp, high-quality source. HighBitrate Ultra is
another comparison candidate, not an established winner. NVIDIA documents the mode families in its
[Python API](https://docs.nvidia.com/maxine/vfx-python/latest/api.html) and
[VSR guide](https://docs.nvidia.com/maxine/vfx/latest/Filters/VideoSuperResolution.html).

## October 20, 2026: native VFX GPU buffers

**Rolled back at the operator's request.** The active VSR wrapper and progressive
output path were restored from the pre-thread-change source snapshot. GPU preview
preparation is serial; only CPU JPEG encoding uses the original four workers.
The newer `defer_preview` call keyword remains accepted but returns the original
ordered JPEG list, with `preview_deferred: false` in the profile. Spout,
1440p Ultra quality 4, native resolution and RIFE settings remain selected.
The direct implementation and queue/protocol tests below are retained for
controlled investigation, not enabled by the live VSR entry point.

The experimental VSR implementation calls the public VFX C API through a small,
version-checked binding to the bundled SDK 1.2.0 / wheel 0.1.0.1. It wraps
persistent CUDA RGBA8 input/output storage with `NvCVImage_Init`, binds an
explicit CUDA stream before loading, and uses asynchronous `NvVFX_Run`.
Each returned RGB frame owns separate storage and a completion event. Consumers
wait on that event in their own streams. Geometry changes and teardown wait
only for the effect's stream; steady processing does not synchronize the device.

The standard Ultra model (quality 4), generation dimensions, RIFE multiplier
and output resolution remain unchanged. `health.upscale.vsr` reports the actual
implementation, SDK, quality, device, host submission time and GPU span.
`run_host_ms` and `first_publish_s` are submission measurements, not completed
GPU work or display latency. The SDK's Ultra HighBitrate mode 19 worked but was
slower in the isolated fixture; mode 23 was rejected by this installed SDK.
Neither model switch is deployed.

In the reverted experiment, local GPU output was submitted before preview
resize/readback/JPEG. Each of the
four bounded preview workers owns a CUDA stream and reusable pinned host buffer.
The initial implementation still joined those futures at chunk end and returned
input capture credit through JPEG reception, allowing optional preview work to
starve the next chunk. The negotiated `independent_preview` protocol returned
`presentation_progress` after direct submission. A separate consumer owns one
active preview batch and one replaceable pending batch; it drops only superseded
browser previews, never the paced local/WebRTC presentation frames. Cancellation
drains active GPU jobs before scene resources are reused. Older clients and
explicit segmented recording retain the original synchronous contract.

With live generation paused, `tulip/test_direct_vsr_live.py OUTPUT --device cuda:0
--compare-legacy` checks byte equality against the retained `LegacyVsrUpscaler`,
alternating frames, eight-frame asynchronous bursts, cross-stream input,
retained-output ownership, geometry changes and idempotent cleanup. It also
checks the restored serial preparation contract: JPEG workers receive CPU arrays
and the caller joins the ordered previews at each chunk boundary. It does not
claim cross-chunk preview independence. Use the CUDA ordinal appropriate to the
test process's visibility mapping, not the live service's logical numbering.

The sibling Etherea `npm run test:video` suite covers queue bounds, failure and
cancellation, independent capture credits, frame ordering, timing-tail gates and
known-motion jumps. `npm run test:video:live -- OUTPUT --layer ID --osc-port PORT
--seconds 120` is the required deployment acceptance test. It observes the actual
projector path; isolated GPU throughput or average FPS is insufficient. See
`tulip/AGENTS.md`. Steady-state defaults at 24 fps require >=23.52 fps,
p95 gaps <=64.5 ms, p99 <=85.33 ms, maximum <=104.17 ms, <=2% intervals over
62.5 ms, and no missing/reversed frames, new publisher underruns or drops.

The live 16-frame output-processing median fell from about 221 ms to 188 ms
in the measured steady intervals. These are observations with different scene
content, not a controlled model-quality benchmark. Steady sending stayed near
24 fps, but seven short native queue underruns coincided with two scene changes.
Faster processing and average FPS alone do not establish smooth presentation.
Keep evaluating source receipt, projector submission and image motion separately.

Later live measurements did reproduce severe stalls, including a minute around
14 fps with no prompt change or completed scene transition. Browser-preview
futures still gate the next chunk, and their sampled end-of-chunk wait reached
2.03 seconds. That wait includes pending GPU work and readback, not just JPEG
encoding. The recent asynchronous implementation is a regression candidate;
the earlier steady averages and isolated pixel tests do not clear it. See
Etherea's `docs/JOYAI_INPUT_WORKER.md` and the private October 20 timing evidence.

The public buffer/stream contract is documented in NVIDIA's
[VFX API reference](https://docs.nvidia.com/maxine/vfx/latest/API/Reference/VideoEffectsFunctions.html)
and [VSR filter guide](https://docs.nvidia.com/maxine/vfx/latest/Filters/VideoSuperResolution.html).

## October 19, 2026: GPU-local Spout transport

The current native profile is 1152×648 with RIFE 2× and VSR Ultra at
2592×1440/24 fps. The older trial notes below describe earlier profiles.
`local_transport: "spout"` in `output-presentation.json` selects the local
GPU publisher; retain `direct_ndi: true`, `progressive: true`, the selected
height, and the existing `gpu_video` configuration. The health field
`upscale.direct_ndi` retains its legacy name and adds `transport: "spout"`.

`spout_output.py` prepares immutable GPU frames. `tulip/spout-native/sender.cpp`
copies them into an owned CUDA/D3D11 texture ring on the same adapter, then
presents on an independent native thread using a high-resolution Windows
waitable timer. There is no raw CPU download on this local branch. A late
frame does not cause a burst of catch-up frames. Queue gaps, missed deadlines,
shared-texture access failures and actual native submissions are reported.
Transition kernels are warmed before playback to avoid first-use stalls.
Scene replacement now retires queued frames only after the first replacement
texture is ready, in the native submission transaction. Reset retains the
moving shader queue. The feeder releases the producer/status condition while
waiting for GPU copies, and reported queue capacity includes the native ring.
After a client stop, historical counters remain available for diagnostics, but
the retired native queue cannot throttle creation of the next sender. Queue
depth reports only currently owned textures.

Build against Spout2 tag `2.007.018`, commit
`6b68a5b4fdeca6130b851e5f9366467a844860e5`, and CUDA driver development headers:

```powershell
cmake -S tulip/spout-native -B tulip/spout-native/build -A x64 `
  -DSPOUT_ROOT=<Spout2-checkout> -DCUDA_ROOT=<CUDA-headers-and-lib-root>
cmake --build tulip/spout-native/build --config Release
```

The CUDA root must contain `include/cuda.h` and `lib/x64/cuda.lib`.
Easel must be built with that SDK in `external/Spout2`; its `SpoutSource`
receives the GPU texture and preserves existing layer IDs and transforms.
The SDK's `primaryModel.localOutputs.joyai` selects transport `spout` and
source `ETHEREA JoyAI Spout`; the separate `senders.joyai` NDI identity
remains for network consumers. Easel's `spout_ndi_relay` is an optional,
separately supervised network exporter, not part of local display delivery.

Run `test_output_geometry.py` for CPU configuration checks. With generation
stopped, `test_spout_native_live.py <evidence-dir>` checks actual GPU color,
ownership after input-buffer reuse, and native pacing under Python stalls;
`test_spout_publisher_live.py <evidence-dir>` exercises bursts and scene
handoffs. `test_spout_pacing_live.py <evidence-dir>` deliberately stalls the
feeder, checks concurrent producer/status access, and verifies that old motion
continues until a replacement is ready without draining the reset bridge.
It also closes with a full native queue and verifies that the same publisher
can create a fresh sender and resume, without reloading the model.
Run GPU integration checks in the interactive Windows session
used by the show. Session 0 is not a substitute for WGL/DX interop validation.
Live validation must sample the actual Easel receiver's advancing frame
counter in addition to publisher counters.

Etherea's transport worker ages post-chunk queue snapshots using its monotonic
clock and the configured output FPS. Repeated snapshots on the preview frames
of one chunk do not refresh that timestamp. This avoids treating a batch's
peak queue depth as a constant reserve. The target reserve, capture-rate
bounds and outstanding-input limits are unchanged; stale feedback reverts to
nominal capture. Worker regressions live in Etherea's
`scripts/streaming/joyai-transport-worker.test.mjs`.

Rollback first stops the optional Spout-to-NDI relay to release the shared
NDI name, then selects `local_transport: "ndi"` at a chunk boundary and restores the
local layer binding through the SDK's verified NDI route. Preserve the other
output and generation settings when editing the JSON.

### Local shader input

`input-transport.json` independently selects `transport: "spout"`, an explicit
`source`, and an indexed CUDA `device` on the shader's graphics adapter. The
worker negotiates this on a new session and sends four-byte capture-credit
tokens using its existing paced, bounded input controller. Older clients
continue with MJPEG. Setting `transport: "mjpeg"` and restarting only the
client session rolls input back without reloading model weights.

Easel exports an additional Spout texture only for the explicitly named
`EASEL_SHADER_SPOUT_ZONE`, configured in its SDK stack environment. It uses
the same flipped GPU framebuffer as NDI and retains NDI for other consumers.
It never follows whichever zone happens to be active. The receiver DLL
`joyai_spout_input` copies into owned CUDA memory, handles texture changes,
and rejects stopped/stale publishers. GPU resizing matches the existing
separable uint8 bicubic boundary behavior. Shader frames stay on GPU through
VAE preprocessing; the vision tokenizer downloads an anchor on prompt changes.
A separate short-lived GPU reader keeps scene-transition shaders moving.

This is not necessarily an entirely GPU-only path across adapters. On this
machine CUDA reports no peer access between the display and generation GPUs;
the native RGB batch transfer therefore uses system-memory staging. There is
no NDI decode, browser canvas, or JPEG encode/decode in the negotiated model
input path. `health.input_transport` reports actual captures, source frame
numbers, read latency, explicit PIL downloads and cross-GPU byte counts.

Tests: `python -m unittest tulip.test_spout_input`; with generation stopped,
run `tulip/test_spout_input_live.py <evidence-dir>` in the interactive show
session. It checks color, orientation, retained-frame ownership, source
geometry changes, stale-frame rejection, cross-adapter pixel equality and
capture from the real shader. The worker's flow-control tests are in Etherea's
`scripts/streaming/joyai-transport-worker.test.mjs`.

The native model and RIFE remain at 648x360. RealESRGAN general-x4v3 already computes 2592x1440; the final bicubic antialiased resize selects the delivered size.

Optional machine-local `tulip/output-presentation.json`:

```json
{"height":1080}
```

Supported heights: 720 (1296x720, default) and 1080 (1944x1080). Both preserve the model aspect ratio. The setting is read once per presentation chunk, under the existing output lock. Invalid/partial edits retain the last valid setting and expose `config_error` in health. Deleting the file restores 720. Resolution changes do not reset generation, prompt, temporal history, or RIFE.

Rollback: atomically replace the JSON with `{"height":720}`. The client accepts native, 720, and 1080 frames and resizes only upon decoded frame arrival. The JoyAI NDI export permits width up to 1944; other providers retain their existing caps. The existing WebRTC publisher detects NDI geometry changes and restarts/re-probes its fixed-size rawvideo pipe, briefly interrupting delivery.

Keep RIFE at 2x and neural rendering disabled for this trial. The optional NR helper currently supports only 1296x720 and bypasses other shapes. This does not raise native generation resolution or add model detail beyond the existing SR result.

Higher resolution increases JPEG, GPU/CPU transfer, canvas readback, raw NDI transport, and receiver decode costs. Validate fresh NDI ingest and WebRTC playback, not just the paced NDI sender count.

Validation: `python tulip/test_output_geometry.py`, full SR module import, JS syntax checks, actual client drawOutput native/720/1080/rollback acceptance checks. Live trial evidence and outcome are recorded separately.

## 2026-09-15 live trial: reverted to 720

Deployed through SDK JoyAI runtime lifecycle and app.reload, preserving current prompt/strength. The first deployment hit a default-argument initialization typo; corrected it, verified full module import, then started successfully. GPU text residency, RIFE2, NR disabled verified after restart.

At 1944x1080, the 114-second observer measured 30.91 canvas fps, 32.00 paced NDI fps, but only 13.25 fresh NDI ingests/s. Output drops and NDI send errors were zero, which alone did not detect the duplicated frames. Capture-to-canvas latency was approximately 1.5 seconds. A 90-second WHEP probe confirmed actual1944x1080, 2853 decoded frames, zero packet loss/dropped frames, and one237ms freeze. This decoder success did not imply smooth source motion: repeated images still encoded at32fps.

Rolled back by atomically setting output-presentation.json to height720, without restarting/reloading the model. A subsequent34-second observation measured32.29 display/fresh NDI fps and32.00 paced NDI fps, zero output discards/send errors, latency1298ms. Confirmed1296x720, RIFE2, SR enabled, no SR/config errors. Keep720 active. The delivered-frame bottleneck must be addressed before retrying higher resolution; do not simply increase buffering.

Evidence: SDK logs/output-1080-soak-20260915.json and output-720-rollback-soak-20260915.json; Etherea logs/output-1080-whep-20260915.jsonl. The followup1080 probe overlaps the rollback and is not a clean1080 sample. Final rollback WHEP verification: Etherea logs/output-720-rollback-whep-20260915.jsonl.

Final30-second rollback WebRTC check:1296x720 at32fps, zero dropped frames, freezes, or packet loss.
