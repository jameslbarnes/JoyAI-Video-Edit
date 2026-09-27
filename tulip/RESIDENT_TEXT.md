# Resident BF16 prompt encoder — Tulip, 2026-09-12

## Persistent launcher setting ? 2026-09-14

The launcher now reads `tulip/runtime-settings.json` on every start. This local,
gitignored file contains `{"residentText": true}` on Tulip. Normal SDK/backend
supervisor starts therefore retain full BF16 GPU residency without adding a
special command-line flag or restarting the backend. An explicitly supplied
`-ResidentText` switch (including `-ResidentText:$false`) takes precedence.
Missing settings default to offload; invalid boolean values fail with an error.
Separate-GPU and 20 GiB free-memory preflights remain enforced.

The SDK's existing `primaryModel.keepResident=false` and
`preserveVideoRuntime=false` keep Photo/Video model lifetimes exclusive.
For rollback, set `residentText` to false and restart JoyAI through the SDK.
Keep the other model unloaded until the encoder has returned to offload.

Validation: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File
 tulip/test-runtime-settings.ps1` checks launcher syntax, missing-file default,
true/false settings, and rejection of non-boolean values. Normal SDK-managed restart was verified with all 8,306,217,216 parameter
elements resident on logical cuda:1 / physical GPU0, BF16. It used the existing
supervisor command without an explicit -ResidentText flag.

Fresh live prompt measurements on September 14: before restoration, encoding
2547?2605 ms and first updated canvas 3555?3620 ms (three samples). After
restoration, encoding 450?462 ms and first updated canvas 1393?1814 ms (two
completed samples); another prompt revision superseded the first requested
sample. These are real live updates with evolving scene input, not identical
frozen-image benchmarks. A sampled GPU memory reading was 27,079 MiB on GPU0
and 23,902 MiB on GPU1; this is not a peak measurement.

Evidence: etherea-ai/logs/joyai-encoder-current-20260914.json and
joyai-encoder-restored-resident-20260914.json. The initial connection needed a
retry after cold GPU setup. A subsequent selection check saw changing live
shader values and declined verification; the following SDK selection passed.
Saved prompt and strength were restored; no shader edits were authored.
Final live playback check: 31.77 displayed fps, 31.74 browser-to-NDI accepted fps, median capture-to-canvas 1473 ms, p95 frame gap 36 ms and maximum 49 ms. Live output remains RIFE 2x -> SR 2x with NR disabled.

The historical explicit-launch instructions below describe the September 12
trial. Its statement that normal restarts always use offload is superseded by
this persistent local setting.


Opt-in tested configuration: physical GPU1 runs existing FP8 DiT / BF16 VAE,
physical GPU0 runs the **unchanged BF16 MiMo-VL encoder**, shaders, NVENC preview
and FP16 output SR. RIFE remains 2x. No new packages, weights or quantization.
`JOYOMNI_FP8_TXT` refers to the DiT text stream, not this prompt encoder.
Encoder revision remains `4bfb270765825d2fa059011deb4c96fdd579be6f`.

Flux was intentionally stopped (not deleted) with the user's authorization.
All 8,306,217,216 encoder parameter elements were verified on logical cuda:1,
physical GPU0, dtype torch.bfloat16 through the new `/health.textEncoder` field.
The switch explicitly sets `JOYOMNI_TE_OFFLOAD=0`; normal launches explicitly
set it to 1, preserving the original safe offloaded configuration.

## Measurements

Three same-current-prompt updates through the real active Etherea client's
WebSocket, then observed applied revisions reaching its output canvas:

| Mode | Encoder wall time, three trials | Client send → first updated canvas |
|---|---|---|
| Sequential CPU offload; paused Flux resident | 4310 / 4358 / 4151 ms | 5465 / 5279 / 5064 ms |
| Full BF16 residency; Flux stopped | 424 / 328 / 408 ms | 1543 / 2279 / 1787 ms |

Median encoder speedup ~10.6x. Encoder wall time includes prompt/image
preparation, lock waiting, GPU execution and completed CPU embedding handoff;
it is not isolated GPU kernel time. Canvas endpoint is the actual first new
revision presentation, in these trials a RIFE midpoint, not semantic convergence.
Cue/LLM/ASR and iPhone display latency are excluded. One real user/Cue scene
creation occurred during the resident window; this is a small live comparison,
not a controlled same-image/prompt-length A/B. Updates preserved temporal cache.

Separate steady 35-second observation, final ~30.5 seconds used for rates:

- 16.12 native model frames displayed/sec; 32.27 including 2x RIFE. These are
  paced delivery rates, **not maximum uncapped inference throughput**.
- Canvas gaps p50 31 ms, p95 36 ms, max 46 ms. Zero capture/output drops and
  zero renderer long tasks reported. No scene reset in this steady window.
- Source-sample → canvas p50 1392 ms / p95 1422 ms. Includes existing chunk
  accumulation and playback buffers; excludes upstream capture and phone.
- Sampled whole-device peak usage: GPU0 25,142 MiB; GPU1 23,277 MiB. These
  include competing workloads and are not process-only allocation peaks.
- Sampled JoyAI video-device PyTorch allocated/reserved: 19,028 / 22,592 MiB.
  Text-device PyTorch peaks and instantaneous process GPU peaks not instrumented.
- Physical RAM free after loading: 51.45 GiB. Cold loading temporarily reached
  only ~1.2 GiB free while building CPU DiT weights. Process launch → observed
  readiness was about 2.5 minutes, including VAE warmups; not isolated stages.

The 1296x720 canvas PNG was inspected: nonblank generated koala imagery with
strong purple shader texture, softness and deformation still visible. This
changes encoding latency, not model quality. No claim that cache artifacts are
fixed or isolated shader GPU timing is improved. Cue output-frame ingestion
remained healthy after restart (900/900 acknowledged at the final check).

## Launch / stop / rollback

From `C:/Users/Admin/src/joyai-video-edit`, existing isolated environment:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tulip/run-service.ps1 -Gpu 1 -TextGpu 0 -Port 7873 -RifeFluxRoot C:/Users/Admin/src/FluxRT -UpscaleGpu 0 -ResidentText
```

Launch hidden when backgrounding. The live launch used Python `subprocess.Popen`
with `creationflags=subprocess.CREATE_NO_WINDOW`, redirected stdout/stderr to
`tulip/resident-text-20260912b.log` and `.err.log`. Windows Start-Process in the
agent environment failed on duplicate Path/PATH keys before starting a process.
The launcher refuses residency on the video GPU or with less than 20 GiB free
on the text GPU; it never stops another workload itself. CUDA_VISIBLE_DEVICES
is `1,0`; service remains bound to 127.0.0.1:7873.

Before loading, stop only the verified Flux runtime through its supervisor:

```powershell
Invoke-RestMethod http://127.0.0.1:7860/api/runtimes/fluxrt/stop -Method Post -ContentType application/json -Body '{"force":true,"cleanupOrphans":false}'
```

For live JoyAI reloads, from etherea-ai use the existing snapshot helper before
stopping the runtime, then resume after `/health` is ready:

```powershell
node scripts/smoke-tests/joyai-transition-live.cjs prepare logs/NEW-reload.json
Invoke-RestMethod http://127.0.0.1:7860/api/runtimes/joyai/stop -Method Post -ContentType application/json -Body '{"force":true,"cleanupOrphans":false}'
# Launch service above; wait for health.runtime_loaded=true.
node scripts/smoke-tests/joyai-transition-live.cjs resume logs/NEW-reload.json
node scripts/smoke-tests/joyai-encoder-live.cjs logs/NEW-encoder.json
node scripts/smoke-tests/joyai-transition-live.cjs observe logs/NEW-cadence.jsonl
```

Reload resets temporal cache; helper preserves prompt and strength, refusing
to overwrite if the tile/prompt changed while stopped. Preserve existing NDI,
Easel, Cue, backend and preview services. During maintenance use SDK ops.hold
with a bounded TTL and release your own hold after verification.

Local SDK `nanos.json`: changed **only** primaryModel.keepResident true → false;
preserveVideoRuntime was already false. This prevents Photo selection from
retaining this memory-heavy JoyAI service while loading Flux. Photo/Video are
now cold switches, not instant switches. SDK config is local and not committed.
Automatic JoyAI restarts still use the normal offloaded preset: full residency
is deliberately an explicit `-ResidentText` launch, not a silent new default.

Rollback: snapshot/stop JoyAI as above, launch the same command **without**
`-ResidentText`, resume, then (after checking GPU0 capacity) launch paused Flux:

```powershell
# From etherea-ai:
.venv/Scripts/python.exe -u scripts/launch_fluxrt_companion.py --gpu 0 --text-offload --start-paused
```

Only after the encoder is offloaded again may primaryModel.keepResident be
restored to true. Never load Flux concurrently with the full resident encoder.
No environment, driver, cache, recording or model files were removed.

## Evidence / tests

Local ignored artifacts in etherea-ai/logs:
`joyai-encoder-offload-20260912.json`, `joyai-encoder-resident-20260912.json`,
`joyai-resident-cadence-20260912.jsonl` and `.png`,
`joyai-resident-reload-20260912.json`. GPU samples in this repo:
`tulip/resident-memory-20260912.jsonl`. Evidence includes private creative prompts;
do not publish raw captures without checking them.

```powershell
.venv/Scripts/python.exe -m unittest discover -s tulip -p test_encoder_status.py
.venv/Scripts/python.exe -m unittest discover -s tulip -p test_stream_controls.py
```

3 encoder-placement tests and 9 cache/prompt control tests passed. Launcher
PowerShell syntax parsed; same-GPU residency preflight refused as intended.
Real service load, BF16 placement, 3 live prompt updates, cache-preserved logs,
35-second delivery observation and Cue frame-ingestion health verified.
Photo/Video cold switching was not exercised during this experiment.
