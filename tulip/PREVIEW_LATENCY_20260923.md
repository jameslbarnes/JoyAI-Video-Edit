# Preview latency measurement and 648p buffer trial

Final decision: retain the existing **500 ms preview target**. The 375 ms trial
saved about 0.12 seconds, but longer-prompt stress produced two preview underruns
and a 344 ms display gap. The setting was manually restored without restarting
JoyAI or the video publisher. The 648p GPU-only RIFE path remains live.

The prior 1.7-second metric is capture-to-local-browser-canvas delay. It is not
a TV/phone end-to-end latency measurement or new-scene startup time. GPU output,
NDI, and browser preview have separate delivery queues; their buffers must not
be added together as if they were serial stages of the same path.

## Measured baseline

`joyai-preview-latency.cjs` correlates native displayed frames with their original
capture/send timestamps and the last input in their eight-frame chunk. It uses
the server's chunk and presentation timings, and records browser arrival/display
timestamps. Interpolated frames are excluded from this decomposition because
their metadata refers to an endpoint rather than a separately captured midpoint.

35-second sample, 390 native frames after warm observation exclusion:

| Stage | Mean |
|---|---:|
| Input JPEG capture/encoding | 11.7 ms |
| Wait for remaining input frames in the chunk | 293.5 ms |
| JoyAI processing | 580.0 ms |
| RIFE submission plus VSR/NDI/preview processing | 241.7 ms |
| Remaining delivery/scheduling time | 36.8 ms |
| Browser arrival-to-display wait | 539.8 ms |
| Total | 1703.5 ms |

The output-stage measurement includes the GPU RIFE completion wait inside VSR;
RIFE's submission-only timer is not interpreted as inference completion time.
Browser decode wait was effectively zero because frames are decoded ahead.

At 12 native fps, eight inputs span 583 ms between the first and last captured
frame, giving about 292 ms average accumulation delay. The chunk size is tied
to `pipeline.vae_scale_factor_temporal` in the current streaming model, rather
than being an arbitrary transport batch size. Browser playback then spreads
the burst of output across time and maintains its motion reserve.

## Trial and retained changes

The client temporarily used a 375 ms steady preview target for 1152x648 /
12 native / 24 output with GPU RIFE. The longer test rejected it as a default;
all profiles now retain 500 ms unless explicitly overridden for a test.
New scenes retain at least 500 ms of reveal reserve even during a 375 ms test;
legacy RIFE scene morphs retain their full existing reserve. Native capture,
GPU/TV publisher buffers, interpolation, VSR, bitrate, model settings, and
resolution are unchanged.

The existing client adopted the preview methods without a model restart or
session reset. The persistent source and current client both use the restored
500 ms default. `getStats()` now reports the effective `playbackBufferMs`.

## Initial verification

- 65-second measurement including two prompt changes: 759 native frames,
  mean latency **1580.9 ms** (median 1576), versus 1703.5 ms baseline. Mean browser
  wait fell to 415.8 ms; model and output processing stayed about the same.
- Zero observed preview underruns. Maximum display-callback interval 68.6 ms;
  95th percentile 52.1 ms. This is bounded live evidence, not proof of zero jitter.
- Concurrent prompt test: 24.018 canvas fps, zero output drops, publisher drops,
  underruns, delivery gaps, or client errors; original prompt restored.
- Same-content scene test during the 375 ms trial: first server output at 1244 ms,
  first generated presentation at 2428 ms, full reveal at 2663 ms. Live shader
  motion bridges the wait. Maximum generated presentation gap 58.5 ms,
  zero publisher restarts/drops/underruns or client errors.
- 11 capture-clock, scene-transition, and shader-bridge tests passed. The
  existing capture feedback fixture was missing its interpolation multiplier;
  it now explicitly supplies 1 for its 24-native/24-output case.

Evidence is in `evidence/preview-latency-20260923`. The pre-edit client is saved
as `joyai-client-before.js` (SHA-256
`7fd26091a942d2dd011ed30b80a0f3edd8c61673966d03690ba77c9206b92416`).

From the Etherea repository, explicitly restore the prior live preview target:

```powershell
node scripts/smoke-tests/joyai-preview-buffer.cjs 500 NEW_EVIDENCE.json
```

Use `auto` instead of `500` to remove the override and use the 500 ms default. This changes
only the local preview clock; it does not restart the renderer or the video
publisher. No automatic resolution rollback was added.

## Longer-prompt result

The four-minute test exercised both prewarmed prompt capacities, including
1024 actual tokens. Over 248.491 seconds it maintained 24.001 canvas-counter
and publisher fps, with zero output drops, publisher underruns, delivery gaps,
or client errors, and restored the original prompt. Those aggregate counters
alone missed two short preview stalls.

The independent 180-second per-frame observer recorded two preview underrun
episodes and a maximum display-callback gap of 344 ms. Mean generation time
rose from 580 to 627 ms across the larger prompts; output processing remained
about 242 ms. The browser mean wait was 436 ms and total mean latency 1654 ms.
This is why the initially successful short trial was not kept.

The next useful preview change is earlier delivery of completed JPEGs: VSR
already submits GPU video frames progressively, but the WebSocket preview waits
for the whole presentation chunk's JPEG results before sending them. That work
needs an ordered, bounded preview-send path with cancellation/backpressure
handling. It was identified here, not implemented or measured as a speedup.
Larger latency reductions also require faster generation or a change to the
model's temporal chunking. Faster interpolation alone cannot remove these waits.

Final restored-500 check: 30 seconds, 332 native frames after warm exclusion,
1701.5 ms mean preview delay, zero preview underruns, maximum callback gap
65.9 ms, unchanged scene epoch. Live readback confirms the original prompt and
strength restored, GPU RIFE active with zero intermediate download bytes,
the same NVENC publisher PID, zero publisher restarts/drops/underruns, and zero
OOMs on both GPUs. The restored-500 window does not repeat the full long-prompt
stress test and must not be represented as that test.
