# Progressive presentation trial

Enabled `progressive:true` in `tulip/output-presentation.json`, alongside
`height:1080,direct_ndi:true`. Native generation remains 648x360 at16fps,
RIFE2 produces32fps, direct output1944x1080, desktop preview1296x720, NRoff.

`OutputUpscale` now optionally publishes each completed SR frame immediately
and submits its preview JPEG to the existing four-worker pool while the next
frame is upscaled. The same SR model, precision, resize and JPEG quality apply.
All futures are consumed before the chunk returns (maximum32); no rolling
work queue. Raw-output/NR-enabled calls retain the original path. A partial SR
failure keeps output geometry stable through the rest of that chunk and
reports the latched error. The toggle is read at chunk boundaries.

Deployment used the SDK runtime-only reload, restored the exact saved prompt
and strength, and verified independent fresh `TULIP (Easel - FLUX-IN)` input,
GPU-resident text encoding and the masked1024-token prompt graph. Backup:
SDK `logs/restart-20260915-205758/`.

Validation:32 passing CPU tests, including ordering, overlap, partial failure,
direct publisher, prompt graph, prompt controls and strength controls.

Measured in the same runtime with35-second off/on observations:

| Metric | Batch | Progressive |
|---|---:|---:|
| Median SR+preview stage |311.6ms|231.6ms|
| Median SR work alone |187.5ms|204.8ms|
| Median capture-to-desktop |1501.5ms|1459.5ms|
| Fresh NDI rate |31.93fps|31.99fps|
| Output drops |0|0|

Progressive first-frame submission was about14ms, rather than waiting for
the complete~200ms SR batch. Overlap adds some SR contention but reduces total
stage time by~80ms. Full-stream savings are smaller; scene changes and playback
buffer state varied during observations, so the42ms median difference is not
a controlled latency guarantee. Evidence: SDK
`logs/progressive-ab-off-20260915.json`, `progressive-ab-on-20260915.json`, and
`progressive-enabled-20260915.json`.

The mobile publisher bitrate was then raised from10000 to15000kbps in
`C:/Users/Admin/bin/etherea-webstream.env`; only the identified hermes-screen
NVENC publisher was restarted, under SDK ops hold/resume. Original config and
verified process command: SDK `logs/restart-20260915-210415/`.

Post-change same-prompt scene test:32.04freshfps, all15morph frames, no output
drops, same direct socket ownership, RIFE2. Evidence:
SDK `logs/restart-20260915-210518/scene-transition-observation.json`.
WebRTC probe: Etherea `logs/progressive-15mbps-whep-20260915.jsonl`.
This workstation receiver does not measure the phone's wireless link.

Higher native resolution was not enabled: generation still takes~380-400ms
per500ms chunk; progressive delivery improves scheduling rather than freeing
the substantial GPU capacity needed for78% more native pixels.

Rollback: atomically set `progressive:false` in output-presentation.json;
no model restart needed. Restore HERMES_SCREEN_BITRATE_KBPS=10000 and restart
only the hermes-screen NVENC publisher to revert bitrate. Preserve other
configuration values. No change to attention or existing prompt/scene fixes.
