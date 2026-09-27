# Cache and session-lifetime audit — 2026-09-24

The user reported that the picture seems to corrupt over time. This audit
changed no running model, creative controls, quality preset, cache policy, or
publisher. It prepared a small session-teardown correction on disk only.
The correction was initially prepared without reloading the runtime. It was
**deployed on 2026-09-25** after the user explicitly requested a restart; see
[restart verification](evidence/cache-fix-restart-20260925-104328/RESTART.md).
The 648p/RIFE/VSR Ultra/4K profile and all seven settings files were preserved.
A short local receiver check confirmed 4K at 24 fps with no decoded drops or
reported freezes; this does not prove the gradual quality issue is resolved.

## Observed live state

Evidence: `tulip/evidence/cache-audit-20260924-185144/`.

- Initial current scene age was 129.7 seconds, at chunk 193, although the
  overall runtime had been active for hours. The temporal history held two
  entries; there were two compiled prompt graphs.
- The following 89.72-second observation spanned an externally initiated scene
  change, epoch 21 to 22. History returned to one entry at initialization, then
  two, and temporal positions restarted at zero before reaching their cap of 8.
  Graph count stayed at two. No reset was issued by this audit.
- No GPU OOMs. Model allocator retries rose from 20 to 21 across the observed
  scene change; output-GPU retries stayed zero. This is allocator pressure,
  not evidence that KV values were corrupted. Model allocation sampled between
  22,295 and 23,145 MiB; snapshots alone do not prove absence of leaks.
- Generation median was 580.52 ms per eight-frame chunk. Publisher recorded
  zero new drops, underruns, restarts, or delivery gaps. These counters do not
  establish visual quality or contradict the earlier WebRTC receiver freeze.

Prompt updates deliberately retain visual history and the first-frame sink.
New scenes clear history. The decoder also carries the last generated frame
forward through a pseudo latent, as upstream does. This makes temporal drift or
old-content persistence plausible without unbounded storage or invalid values.
No matched visual experiment in this audit establishes which mechanism explains
the user's current observation.

The earlier controlled corruption experiment is documented in
`CACHE_DIAGNOSTICS.md`. Its reproduced failure came from a positional cap of 128;
the correction to 8 remains active. Rolling sink replacement failed to resolve
that earlier failure. Neither automatic resets nor sink refresh were enabled.

## Prepared teardown correction

Local `JoyOmniV2VStreamingSession.close()` joined Python workers then cleared VAE
and transformer caches, without explicitly waiting for queued CUDA work.
Worker exit only establishes that submissions ended. The final pseudo-encode
can still be pending independently of the postprocessing worker, and a prepared
next scene can promptly reuse shared runtime resources.

Added a wait for every CUDA device used by the session after joining workers and
before clearing either cache. This follows the barrier present in current
[upstream session close](https://github.com/jd-opensource/JoyAI-Video-Edit/blob/main/deploy/xvideo/serving/joyomni_streaming.py).
It runs at teardown, not every generated frame, and preserves allocator reuse.

`test_session_cache_teardown.py` models pending GPU work surviving worker exit:
two tests failed against the old method. The correction passes both-device
completion, GPU-error ordering, and CPU-only teardown checks. Together with
prompt, graph, history-window, shared-VAE, and no-rollback tests, **38 tests pass**.
These CPU tests establish ordering, not a reproduced full-model GPU corruption
or a perceptual improvement. A controlled GPU/session-replacement test remains
necessary before claiming that it fixes the user's symptom.

The patch, pre-change source, test log, and source hashes are saved with the
evidence. No automatic restart, fallback, quality rollback, or periodic cache
clearing was added. Live remains 648p generation, RIFE, VSR Ultra, 4K/24/15 Mbps.

## Remaining quality diagnosis

Distinguish within-scene degradation from old-scene persistence after edits and
from transport/presentation jitter. A matched recorded-input experiment should
compare persistent history with explicitly fresh sessions at identical prompt,
seed, precision and resolution, retaining native outputs and cache/latent range
statistics. Run it in a separately scheduled model session; a second full JoyAI
instance must not be loaded alongside the memory-constrained live renderer.
