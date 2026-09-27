# Generated-motion scene handoff - REJECTED and disabled 2026-09-25

The user rejected this implementation: "that's awful it feels broken". Cadence
checks did not establish acceptable visual quality. Replaying and blending old
motion is unsuitable for the requested continuous scene changes.

The previous browser client and shader transition were restored live at 13:46
local time, without a model/session restart or changes to creative controls or
quality settings. All four modified publisher/server source files were restored
from hash-verified pre-experiment snapshots. The cache-teardown fix remains.
The running server still contains backward-compatible replay code/history until
its next ordinary runtime restart, but the restored client never requests it.
Do not restart generation solely to unload that inactive code.

Restoration evidence and archived experiment utilities:
`tulip/evidence/restore-shader-handoff-20260925-134606/`.
The requested shader-free, continuously generated scene handoff remains unsolved.
The remainder below is a historical experiment report, not current deployment
status or user acceptance.

## Historical deployment report

The user requires continuous visible motion and direct scene-to-scene transitions,
without a shader interlude or held-frame fallback. The implementation is live in
JoyAI and the existing radio client. All seven quality/performance settings files
match the pre-deployment snapshot. Native 1152x648 at 12 fps, GPU RIFE x2,
VSR Ultra, and 3840x2160 H.264 at 24 fps / 15 Mbps remain the active profile.
The explicit test prompt was restored; the SDK operations hold was released.

## Chosen approach

Keep the loaded generator and existing output GPU/encoder. Retain references to
48 immutable generated frames (two seconds at 24 fps) in each presentation
consumer. The 4K history remains on the output GPU and costs about 1.11 GiB;
there is no new raw-frame GPU-to-CPU round trip on that branch. The separate NDI
and 720p browser consumers retain bounded history in their existing memory path.

While the prompt encodes, the current scene continues generating. When its
session is replaced, play its queued tail, then a forward replay of retained
motion with overlapping dissolves at the loop boundary. This is replay, not a
second concurrently generating scene. No temporal reversal or repeated still
frame is used. Once the new scene has at least 0.75 seconds of output queued,
dissolve between two moving images over 0.4 seconds, then release the old history.
The shader remains a model input; it is never displayed by this handoff.

This resets scene history using the existing safe session lifecycle rather than
partially clearing KV while leaving VAE/previous-frame state behind. Weights,
compiled graphs, publisher, NVENC session, transport and timestamps persist
across ordinary scene replacements. RIFE history resets without a shader morph.
No periodic cache reset, quality rollback, or second full model was added.

The browser advertises the new capability and marks replay separately from fresh
generated frames. It preserves pending motion until the reveal reserve exists,
coalesces rapid requests, waits for the current dissolve before another reset,
and closes evicted ImageBitmaps without releasing an active replay's references.
The two-second memory bound is independent of how long preparation takes.

## Validation

- 32 Python tests, 16 Node scene/clock tests, and 55 Jest client tests passed.
  Tests cover delayed generation beyond the old 20-second bridge expiry,
  generated pixel changes, moving-to-moving dissolve, cancellation, repeated
  replacement, readiness thresholds, bounded history and bitmap ownership.
- First live test: five requests coalesced to three completed scene replacements.
  Zero shader draws/bridge starts; 182 replay frames; longest radio callback gap
  93 ms; no callback gaps above 125 ms. It finished live and outside transition.
- Distinct-imagery test: two completed changes, adding an indigo/turquoise/ribbon
  instruction and restoring the captured prompt. Zero shader draws/bridge starts;
  139 replay frames; longest radio callback gap 85 ms; no gaps above 125 ms.
- Actual 4K WHEP receiver windows: 82.13s at 23.998 fps
  and 57.08s at 23.999 fps. Both had zero decoded drops,
  packet loss or reported freezes. Maximum estimated display intervals were
  54.4 and 57.4 ms.
  The first window had one skipped compositor callback/media jump; the second had
  none. These are separate local Edge receivers, not physical TV scanout.
- Publisher observations: 93.60s and 66.10s; zero new
  drops, underruns, restarts or delivery gaps in both. Median generation chunks
  were 570.27 and 573.54 ms.
- CPU analysis of the 63-second recorded 4K distinct-scene stream: 1,512 decoded
  frames at 192x108 grayscale analysis resolution, zero identical adjacent frames.
  This checks against encoded repeated stills; it is not a perceptual-quality score.
- Output-GPU allocator retries and OOMs stayed zero. Its final allocation was
  about 16.8 GiB including retained history. Model allocation retries rose by one
  per scene replacement (five total); there were no model OOMs. This does not
  establish indefinite cache stability or resolve every earlier jitter report.

## Deployment details and rejected experiment

Before deployment, a separate synthetic 4K CUDA/NVENC fixture caused contention:
76 fixture input drops and 46 additional live publisher underruns during its
measurement, with further draining immediately afterward. It was rejected and
exited. It was not used as acceptance evidence. Its standalone script now refuses
to run alongside a loaded JoyAI runtime. See `gpu-fixture-result.md` in evidence.

JoyAI was restarted once to load the new code. The first deployment controller
stopped at a capability assertion because the new tile imported a cached
`scene-flow-3` browser module. The service itself was healthy. The manager import
version was changed to `generated-motion-1`, and current client methods/capability
were adopted without a second model restart. Final checks confirm the feature
is active, all settings are preserved, and the operations hold was released.

## Evidence and manual revert

Evidence: `tulip/evidence/generated-handoff-20260925-114939/`.
`validation.json`, `final-checks.json`, receiver summaries and recordings contain
final measurements. `live-scenes.mkv` and `distinct-scenes.mkv` are stream copies,
without re-encoding. `transition-contact.png` samples the first distinct handoff.
`handoff.patch` and `deployed-source-hashes.json` identify the deployed changes.

The original sources are retained in `before/deploy/xvideo/serving/` (four modified
server files), `joyai-client.before.js`, and
`etherea-before/src/web-client/static/js/multistream-manager.js`. To manually
restore the previous shader transition, reconcile these saved source versions
against any subsequent edits, restore only those handoff changes, adopt the
restored browser methods, and reload JoyAI through the existing SDK/supervisor
lifecycle while capturing/restoring the then-current prompt and strength. The
new `motion_bridge.py` can remain unused. Do not restore quality or creative
settings from stale snapshots. No automatic rollback is enabled.

Tests and utilities: `tulip/test_generated_motion_handoff.py`,
`etherea-ai/scripts/streaming/joyai-generated-motion.test.mjs`, and
`etherea-ai/scripts/smoke-tests/joyai-generated-motion-live.cjs NEW.json [distinct]`.
The live utility has bounded auto-cleanup and only restores prompts it still owns.
