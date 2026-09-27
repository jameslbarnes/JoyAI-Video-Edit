# Session-age-dependent jitter — September 16

The overnight runtime reached roughly 50,000 chunks and delivered about 16.4 fps instead of 32, with repeated 300–650 ms delivery gaps. Resetting only the GPU publisher left throughput at 16.35 fps; raising the runtime from BelowNormal to Normal priority left it at 16.60 fps. Neither resolved the aged slowdown. The priority change was diagnostic only; it is not included in the runtime configuration.

## Cause and change

`JoyOmniV2VStreamingSession._denoise_chunk` and `_next_selected_chunk_ids` each called `Pipeline._get_chunk_windows`, then used only the final window. The batch helper enumerates every historical chunk, building a dictionary and selected-ID list for each. Its CPU work and temporary allocations grow linearly with session age even though the attention/cache window stays bounded.

A 15-second, 20 Hz py-spy profile of the actual aged process (PID 29612) collected 175 GIL-held samples without sampling errors. Approximately 114 samples were inside historical window construction through the denoising and cache-eviction paths. This is a share of GIL-held samples, not a percentage of wall time or GPU time. Blocking Python here delays frame preparation and publisher threads too, which can misleadingly appear as slow GPU preparation in elapsed stage timings.

Added `_get_last_chunk_window` and switched both streaming call sites to it. The original batch helper remains unchanged. The new method produces identical current-window bounds and selected IDs, including global sink behavior, window size one, first chunks, and partial final chunks. For a fixed attention window, work is independent of stream age. No model precision, attention, pixel processing, resolution, interpolation, or bitrate settings were changed.

## Reproduction and tests

CPU benchmark on this host (`streaming-window-benchmark-20260916.json`):

| Session chunks | Historical lookup median | Historical peak allocation | Direct lookup median | Direct peak allocation |
| --- | --- | --- | --- | --- |
| 100 | 0.031 ms | 26,720 B | 0.0004 ms | 424 B |
| 50,000 | 32.07 ms | 19,611,744 B | 0.0005 ms | 552 B |
| 100,000 | 76.37 ms | 39,168,352 B | 0.0005 ms | 552 B |
| 1 billion | Not run | Not run | 0.0005 ms | 552 B |

Timing and allocation measurements were separate. This isolated CPU benchmark is not a claim of the exact in-process stall duration under GPU/thread/GC contention. The live profiler establishes that the same path dominated Python activity in the degraded process.

`python -m unittest tulip.test_streaming_window_lookup tulip.test_stream_controls`: 17 tests passed. Window tests compare against the original batch implementation across 480 combinations, enforce bounded iteration at up to one billion chunks, exercise the actual streaming eviction method, and check invalid inputs. Compilation checks passed. A billion-chunk helper test is not an eight-hour end-to-end soak test.

## Evidence locations

SDK logs (relative to `C:/Users/Admin/src/easel-agent-sdk`):

- `logs/aged-jitter-20260916-092921.json`: aged health/debug/runtime snapshot.
- `logs/aged-admin-gil-20260916.txt` and `aged-admin-stacks-20260916.txt`: live aged profile and thread stacks.
- `logs/restart-20260916-093243/publisher-isolation-summary.json`: publisher-only reset control.
- `logs/restart-20260916-093850/`: Normal-priority control.
- `logs/restart-20260916-094311/`: patched runtime deployment and saved live prompt/strength/configuration.

The bounded GPU cache and available physical memory did not establish memory exhaustion as the cause. The demonstrated defect is repeated temporary allocation and CPU work proportional to stream age.

## Deployment and long-run observation

SDK deployment completed successfully with new runtime PID 5940. The saved prompt, strength, fresh configured shader input, GPU-resident text encoder, and masked prompt graph were verified. The new runtime retained the original BelowNormal process priority (16384), so the unsuccessful Normal-priority experiment was not carried forward. GPU output remains 2592 x 1440, 32 fps, H.264 at 15 Mbps, with RIFE 2x and the existing SR settings. The existing mobile viewer was reconnected from the temporary 1080p fallback to `joyai-gpu` and confirmed established.

Fourteen additional direct NDI/progressive/GPU publisher tests passed (31 tests across the two focused invocations). The post-deployment source/desktop audit is recorded in SDK `logs/restart-20260916-094702/`; overlapping WebRTC receiver evidence is in Etherea `logs/agefix-whep-20260916.jsonl`.

Completed four-minute audit: 31.9948 fps for both GPU sender and NDI, zero sender drops, median GPU send work 2.24 ms, and one 68.55 ms delivery gap during a scene transition. No recurring steady-state gaps above 62.5 ms occurred in the measured interval (first five seconds excluded; encoder startup preceded the audit). The overlapping receiver decoded 7,672 frames with zero freezes, decoder drops, or lost packets. Desktop observed no callback stalls above 100 ms. Both scene resets and the reversible prompt update completed, with no audit errors. Temporary desktop hooks were removed and the receiver probe exited. Source/receiver intervals overlap but are not identical. These short-run results confirm restored behavior, while the code/profiler/large-index tests identify and address the age-dependent mechanism.

Read-only monitoring started at 09:48 EDT for ten hours, sampling health and debug every ten seconds, with no control actions. Script: SDK `logs/monitor-session-aging-20260916.py`; data: `logs/session-aging-20260916-094819/samples.jsonl`; process PID 26472. This records current chunk count, generation profile, bounded cache, sender fps and recent delivery gaps. It is a telemetry collector, not an automatic repair service. Full eight-hour post-fix uptime validation is still pending.
