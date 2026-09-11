# Prompt continuity and corruption investigation

2026-09-10. The positional-range correction passed the controlled long-sequence
test. Arbitrary scenes and hours-long sessions are not yet exhaustively validated.

## Finding and correction

The former local `max_temporal_ids=128` caused severe late-frame corruption even
with a fixed prompt and with DiT history removed. Graph-off inference reproduced
it too. This setting controls RoPE positions, not cache capacity. The integration
had increased it beyond upstream's browser value of 8. Setting it to 8 preserved
coherent coral through 2,049 frames (128s at 16fps), where 128 collapsed by frame
1,024 in the same source/prompt test. This isolates positional range as the main
cause of the reproduced failure; it is not evidence of a memory overflow.

Production now uses `presets/etherea-continuous-v2.json`: position limit 8, no
periodic reset, no rolling sink replacement, original precision/steps/resolution.
The original approved short-run `etherea-coral-v1.json` remains unmodified for
reproduction, **not** as a recommended long-stream configuration.

## Verified prompt behavior

### September 11 evening: hot graph-capture failure

The live log reported `cudaErrorStreamCaptureUnsupported` in VAE normalization
while a prompt update captured a new DiT graph, followed by
`cudaErrorStreamCaptureInvalidated`. Both GPUs had about 10 GiB free; this was
not a VRAM-allocation error. The input was also pointed at JoyAI's own sender,
leaving a subsequent session with zero input/output frames until FLUX-IN was
restored. The damaged CUDA service required a restart.

Hot prompt application now only reuses an already-captured shape. A new shape
runs eagerly with actual prompt embeddings and preserved KV/VAE state; it does
not capture alongside active VAE workers. Initialization after an explicit reset
can still capture before starting the workers. This is a conservative workaround:
new-shape prompt updates can be slower until reset, not a claim that concurrent
graph capture is now safe. Profile flags `prompt_graph_reused` and
`prompt_graph_capture_deferred` expose the choice. Eight CPU control tests cover
the no-hot-capture rule, reuse, and existing prompt/cache semantics.

Recovery logs: `tulip/freeze-recovery-20260911-2335.log` and `.err.log`.
The existing launcher remains `tulip/run-service.ps1 -Gpu 1 -TextGpu 0`.
Model loading has a high transient host-RAM requirement; when both models are
resident it may page heavily. Stage JoyAI loading before the Flux companion.

Recovery verification: SDK `logs/joyai-freeze-recovered-20260911/audit.json`
measured 16.18 generated canvas FPS over 20 seconds, capture-to-canvas p50/p95
1060/1082ms. `model-1.jpg` is a fresh independently received NDI output (other
one-shot grabs timed out). Coral prompt and direct blend .03 were restored;
all five video workers were alive after 1,841 output frames. This recovery
check did not stress-test additional hot prompt shapes; the new guard has CPU
regression coverage, not a new long-run concurrency qualification.

The patched server accepts `update_prompt {prompt, request_id}` on the existing
WebSocket. It prepares image/text conditioning on a background encoder thread,
then applies it in the DiT worker between chunks. Latest pending request wins;
failed encoding keeps the old prompt. Acknowledgement means queued, not applied:
`output_frame.profile.prompt_revision` and `/debug.session.prompt_control` identify
the actual applied revision. No session counter, VAE history or video KV reset.
Explicit start/reset/reconnect still creates new temporal history.

On Tulip, `JOYAI_GPU=1`, `JOYAI_TEXT_GPU=0` in Etherea's local `.env`. The text
encoder still uses sequential CPU offload (not a second resident 7B model); its
temporary GPU work runs on physical GPU 0. Video DiT/VAE stay on GPU 1. New text
lengths require a graph capture at a chunk boundary; this can cause a short hitch.
First observed updates: 1.4-2.0s background encoding, 0.36-0.40s graph/application.
Do not call this zero-latency prompt switching.

The 180s real-app audit at SDK `logs/joyai-hotprompt-rolling-20260910/audit.json`
recorded four applied prompt revisions in epoch 1, 16 generated canvas FPS,
sampled capture-to-canvas p50/p95 995/1000ms. These are 200ms polls, not physical
display latency. The audit's experimental SDK `prompt.set` request was accepted
by the backend but not proof of model delivery; real Cue/operator prompt changes
provided the observed revisions. The audit utility now uses optional
`--instruction` through `prompt.inject`/Cue, or read-only observation by default.

## Failed experiment: rolling first-frame anchor

`JOYAI_SINK_REFRESH_CHUNKS=32` refreshed the sink from recent generated KV every
16 seconds. It retained only two cache entries and ~19,060MiB allocated memory,
but the captured images and operator inspection still showed severe smearing
and broken structure. **Not a quality fix. Disabled in the local configuration.**
Memory bounds and advancing frames are not evidence of acceptable output.

Upstream already evicts KV. `max_temporal_ids=128` bounds positional indices,
not 128 stored chunks. The steady-state window is first chunk + previous chunk
+ current input, with two retained cache entries after eviction. Unlimited
history is neither needed nor a plausible fix for the observed corruption.

## Controlled ablation

`run-cache-ab.ps1` loads one isolated model and compares identical looping input,
fixed prompt and seed across baseline graph mode, eager mode, no DiT history,
and clamped decoded-frame feedback. Each arm has a fresh session. FP8 DiT and
BF16 VAE are unchanged. The no-history arm also loses graph eligibility; its
speed cannot be attributed to cache size alone. These are offline diagnostics,
not live-throughput measurements. CUDA/async VAE sampling may introduce small
numerical differences despite fixed seeds.

```powershell
# Release JoyAI using the SDK/Etherea lifecycle first; never run beside it.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tulip/run-cache-ab.ps1 `
  -Gpu 1 -Frames 1025 -OutputDir tulip/evidence/cache-ab-new `
  -InputVideo <existing-shader-input.mp4>
```

Outputs: per-arm MP4, representative PNGs, chunk timing, decoded finite/range
statistics, latent RMS, and `report.json`. Test-only method wrappers do not modify
the production model implementation. No model weights or packages change.

Control tests: `.venv/Scripts/python.exe tulip/test_stream_controls.py -v` (6 CPU
tests of actual methods, extracted without loading the GPU model).

First ablation (`evidence/cache-ab-20260910`) reproduced corruption at 1,025
frames in graph, eager and no-KV arms, with no prompt changes. All decoded values
were finite and already within [-1,1]; clamping feedback therefore cannot explain
or cure this run. The positional limit was 128 in all four initial arms.

Follow-up `evidence/position-ab-20260910` compares limits 8, 2, and None (relative
positions) for 2,049 frames each. Upstream `deploy/static/index.html` initializes
its optional positional limit to 8; its Python default is None. Our integration
had chosen 128. This was not a cache-capacity increase and must not be described
as one. The completed position-8 arm took 59.75s for 2,049 output frames including
initialization (~34.3fps offline, not paced app throughput). All 2,049 decoded;
zero exact adjacent repeats. Representative early, middle and final frames were
visually inspected: recognizable luminous coral branches, with some soft edges
and imperfect geometry, without the prior tiled/smeared collapse. See
`position8/comparison-3.jpg` and `position8/review-frame-2048.png` for final output.

Restore through the SDK primary selector after diagnostics. For an explicit
manual service launch (only when the supervised runtime is stopped):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tulip/run-service.ps1 `
  -Gpu 1 -TextGpu 0 -SinkRefreshChunks 0 -Port 7873
```

Loopback access remains unchanged. Logs remain in Etherea `logs/joyai-runtime.log`.
`JOYAI_SINK_REFRESH_CHUNKS=0` is persisted in Etherea `.env`. Prompt revisions,
actual temporal position and retained cache entries are visible via `/debug` and
the SDK's observed primary-model stats. No driver, model-weight or package change.
