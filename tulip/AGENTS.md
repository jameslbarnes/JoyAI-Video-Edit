# Video tests and performance experiments

## Live Spout recovery

Sender creation explicitly pushes the CUDA context belonging to its RGBA tensor,
then restores the caller's context. Cached `torch.empty` allocations do not prove
a context is current after native sender teardown. CPU regression:
`python -m unittest tulip.test_spout_cuda_context` with `PYTHONPATH=deploy`.
The October 24 GPU context probe also verified recovery from an empty current
context and balanced restoration. This changes sender initialization only;
frame preparation, presentation clocks and quality settings are unchanged.

JoyAI and Easel must run in the same interactive Windows session. Before any
Spout model launch, verify both process SessionIds. A backend running in Session
0 cannot launch a working desktop Spout reader; `run-service.ps1` now refuses
that launch before loading weights. Use the configured interactive launcher.
Do not restart a loaded model to repair the radio renderer: the SDK's
`app.reload` reloads an existing app; `stack.ensure` can reopen an exited app
while adopting healthy components. Verify actual input/output frames afterward.

Changes to capture, generation, interpolation, enhancement, publishing, receiving
or presentation require cadence validation. Average FPS alone cannot establish
smooth playback. Preserve the operator's artwork, resolution, VSR quality,
interpolation settings and display refresh unless explicitly testing one of them.

The operator accepted the October 20 morning rollback as substantially smoother.
Keep the original VSR wrapper and serial GPU preview preparation as the live
baseline, with Spout, 1440p Ultra and the selected 24 Hz projector mode. Do not
re-enable the reverted VFX/GPU-preview-worker/display-pacing experiments without
a new operator request. The measurement tools remain enabled for tests. Evidence:
`%LOCALAPPDATA%/etherea/quality/20261020-preview-isolation/thread-rollback/`.

For the October 20 overnight investigation the operator explicitly authorized
necessary Easel/JoyAI restarts and reconnects, preserving scene and quality,
keeping automatic recording disabled, and rolling back regressions. This is
specific to that investigation, not standing approval for later interruptions.
Spout's `feeder` health telemetry separates CPU selection, packing and native
interop submission spans. It retains only 64 slow events in memory, with no
extra GPU synchronization or per-frame file writes. A long CPU span includes
driver/scheduling waits and is not proof of long GPU execution. CPU tests:
`tulip.test_feeder_trace`; correlate with native queue/underrun counters and the
real receiver/projector gate before changing playback buffering.
The October 21 overnight transition-headroom candidate adds a bounded
`spout_transition_reserve` override (2..6, default remains 2) without changing
the ordinary 16-frame reserve. These controls loaded at 01:57 EDT with the
two-frame default retained. Three ABBA/BAAB comparisons (60 same-prompt resets)
measured 16 local underruns across 30 two-frame resets and zero across 30
three-frame resets. An earlier three-frame run still failed the receiver gate
on a separate display-thread stall; buffering is not its fix. After Easel's
persistence work moved off that thread, both three-frame runs passed the full
gate and both two-frame runs still starved. The local configuration adopted
three frames at 07:39 EDT, adding about 41.7 ms of transition headroom only.
Keep the ordinary reserve and the default for other installations unchanged.
`spout_profile_preparation` (default false)
adds reusable CUDA events for selection and packing; elapsed times are read
only after the existing native copy has waited for the producer. These spans
include CPU launch gaps and GPU scheduling/dependency waits, not just kernel execution.
`test_spout_reserve` covers hot reload, invalid values and rollback without
replacing the session/publisher. `test_spout_preparation_live.py OUTPUT
--generation-paused` checks the real feeder with a private source while the
show is paused; never run it beside live generation.
The fixture uses absolute producer deadlines so status/HTTP overhead does not
accumulate into an undersupply of frames. It retains producer times and GPU
spans on assertion failures as well as passes; never discard failed evidence.
The optional native `jsp_copy_trace` export splits completed handoffs into
context, event-wait enqueue, graphics-map, array lookup, copy enqueue, unmap and
stream-completion CPU spans. It has its own size-checked ABI; the original
`Stats` layout is unchanged. Matching submission tokens are retained in slow
feeder events. These markers add no CUDA waits; timings include existing waits.
Copy diagnostics also report whether the producer event was ready at submission
and a CUDA-event span after that dependency. The event span can include GPU
scheduling/resource waits; it is not pure memory-copy execution time.
`output-presentation.json` accepts `spout_copy_priority` 0 (unchanged default) or
-1 (CUDA high-priority hint) for controlled live comparisons without reloading
weights. Only the feeder changes its completed copy stream. No queue, pixels,
frame clock or visual setting changes. `test_spout_native_live.py OUTPUT
--generation-paused [--library STAGED_DLL]` now verifies idle generation, native
trace ABI, priority changes, ownership and receiver cadence through the real SDK.
The separate size-checked `jsp_present_trace` ABI retains at most 64 completed
starvation/late-send events, with empty-queue wait, send span, actual inter-send
gap and token. The native presentation thread adds no GPU synchronization or I/O;
Python copies this history only on health reads, never on each feeder queue poll.
Use these durations to interpret underrun counts without weakening acceptance.
The native fixture requires a freshly built `test_spout_receive` from the active
Easel checkout (default `easel-installation/build/Release`), not an older receiver
binary. Its intentional starvation is saved separately from steady cadence.
`test_frame_deadline.py` checks the input submission guard separately from
worker-raised TimeoutError, including caller cancellation. Python aliases
`asyncio.TimeoutError` to that same exception; inspect the deadline context's
expired state rather than claiming every input failure took the outer 15 s.
`fresh_input.py` provides a bounded two-second grace for the model's Spout input.
Only the typed no-fresh-frame result is retried; driver errors propagate, and
no stale image is substituted. The optional scene-preview reader retains its
short timeout. `test_fresh_input.py` reproduces the measured 816 ms interruption,
permanent disconnects, remaining-budget limits and fatal errors. Deployed October
21 00:33; an authorized 800 ms shader-mutex interruption exercised the retry and
recovered fresh input without replacing the model/session/publishers. Two local
underruns during that intentional outage are retained, not counted as seamless
playback. Normal 120-second gates before/after passed with zero new underruns.
Evidence: `20261020-overnight-gaps/input-grace-fault-01` and the adjacent baseline
and recovery gates. `test_spout_source_gap_live.py` is an explicitly authorized,
bounded fault fixture; never run it as an ordinary passive health check.

The later October 20 SDK 1.3 Streaming Ultra trial was explicitly approved,
including the paused-show GPU checks and required JoyAI restart. Mode 23 with
strength 1.0 is active and the operator preferred it as "Sharper and better."
The 120-second live cadence gate still failed: 23.86 received fps, 151 ms maximum
gap, 3.36% of intervals over 62.5 ms and four new local publisher underruns.
The previous Ultra baseline also failed; image acceptance is not cadence acceptance.
The operator still requires asking before future pauses or restarts. The old
release remains the no-override default and rollback. `OUTPUT_RESOLUTION.md`
documents the pinned releases, default-stream compatibility correction, evidence,
startup selector and paused-show test. Do not restore the reverted parallel
preview or display-pacing experiments as part of this trial.

Run the sibling Etherea `npm run test:video` CPU regression suite. It includes
fault-injected blocked previews across chunk boundaries, bounded queues,
cancellation/owned-buffer lifetime, capture credits, ordering, cadence tails
and controlled motion jumps/duplicates.

GPU component tests must run with live generation stopped. Never benchmark a
second inference workload beside the live show. Spout native/publisher tests
assert native timing and underruns; real Spout/NDI receivers assert fresh-frame
gap percentiles/maxima, FPS and missing frames. Color/ownership tests do not
certify cadence. Offline quality/presentation benchmarks report chunk timing;
their compute throughput is not evidence of paced display playback.

Every deployed video performance change must also pass the real show gate:
`npm run test:video:live -- OUTPUT --layer ID --osc-port PORT --seconds 120 --project-path PATH`
from the sibling Etherea repository. `--no-motion` isolates timing without
decoder/optical-flow CPU load. Use the existing show for this test; do not launch
a competing benchmark. Save the JSON/CSV artifacts and report failures honestly.
Do not loosen bounds simply to make a change pass. Measure steady scenes and
deliberate transitions separately, retaining all samples within each run.

The gate checks fresh-frame p95/p99/max gaps, late-interval frequency, frame IDs,
publisher underruns/drops and telemetry coverage. Easel timestamps measure CPU
submission, not physical scanout. Motion outliers in arbitrary art remain
diagnostic; known-motion fixtures provide assertions without rejecting intended
artistic motion. No automatic video archive is needed or authorized by testing.
The October 21 final audit found that a cold reload had restored an older saved
shader dial while all configuration hashes remained unchanged. Capture the live
`/strength` target and mode before a reload and verify them afterward. The shared
gate now checks live settled values, with `--expected-strength VALUE` and
`--expected-strength-mode MODE` for a pre-change baseline. This dial is separate
from VSR enhancement strength. The original live combined value was restored and
persisted at 08:14; earlier overnight comparisons are labeled separately from
the repeated acceptance at the restored setting.

The October 20 Reset failure escaped from optional `SpoutShaderBridge`
construction (D3D resource allocation HRESULT 0x8007000e). Scene preparation now
falls back to its existing source-frame transition if that extra receiver
cannot initialize. `test_scene_bridge_failure.py` executes the real handler
with allocation/capture failure injection and verifies scene preparation
continues without closing the healthy primary input. An old bridge that cannot
stop rejects the request without losing ownership. The shared live gate checks
frame age and full-duration coverage, including the frozen tail that the prior
interval-only cadence test missed. Test Reset as well as steady playback.

Speech/image provenance (activated October 21) uses immutable `FrameOrigin` values
captured at the denoise boundary. Queued prompts cannot relabel older frames.
`spout-native/FrameTrace.h` keeps 256 published token/public-counter pairs;
Python keeps 512 bounded token origins, registered before native submission.
The 40-byte size-checked optional `jsp_frame_trace` export preserves the old
Stats ABI. `/debug/frame-trace` is the only endpoint returning this history;
never put the ring in per-frame WebSocket profiles or ordinary health replies.
It copies metadata only and adds no pixel readback, GPU synchronization or
presentation deadline changes. Rollback DLLs report telemetry unavailable.
CPU tests: `test_prompt_timing`, `test_spout_frame_origin`, `test_stream_controls`,
and native `test_spout_frame_trace`; the Etherea shared CPU gate runs them.
The user-approved activation preserved the live prompt, influence dial/overlay,
1440p Spout output and VSR 23. The first cadence gate retained one skipped
source frame during an Easel UI stall; the second passed at 24.00 fps without
sequence gaps, publisher drops or underruns. Future restarts still require the
user's approval. See Etherea's
`scripts/AGENTS.md` narration trace instructions for audio/projector joins and
the distinction between a prompt-conditioned frame and recognizable imagery.
