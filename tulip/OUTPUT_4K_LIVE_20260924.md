# Live UHD output — 2026-09-24

The user requested the 4K candidate on the live feed after the isolated output
comparison. The existing `joyai-gpu` WebRTC route now delivers **3840×2160 at
24 fps, H.264 NVENC, 15 Mbps**. VSR Ultra upscales GPU RIFE output; native
generation remains 1152×648 at 12 fps with 2× RIFE. This is upscaled UHD.

The publisher accepts exact UHD geometry. A shared output-size helper gives
warmup and live rendering the same 3840-pixel width at 2160p. Existing 1080p
and 1440p geometry is preserved. NDI remains 1944×1080 and desktop JPEG preview
remains 1296×720. The existing mobile preview configuration selects `joyai-gpu`.

One SDK-supervised JoyAI restart loaded the new geometry support. The lifecycle
captured the active creative state, preserved newer creative updates, and
released its operations hold afterward. No automatic quality rollback was added.

## Verification

- 31 tests passed: output geometry, GPU publisher, progressive delivery,
  direct NDI, presentation pipeline, and no automatic rollback.
- A separate local Edge WHEP receiver decoded 3840×2160 at 24.005 fps over
  64.11 measured seconds, excluding its initial five-second connection startup.
  Zero decoder drops, freezes, packet loss, media-time discontinuities, or
  skipped presented-frame counters. This does not measure the actual TV panel.
- Concurrent production observation covered 79.69 seconds: 1,911 frames sent,
  zero publisher drops, underruns, restarts, or new delivery gaps.
- Native generation chunk median 583.33 ms, p95 595.70 ms for eight 12-fps
  frames (666.67 ms budget). Local desktop preview latency median 1,721.5 ms.
- No GPU allocation retries or out-of-memory events. Minimum sampled free
  memory: physical GPU0 5,001 MiB, physical GPU1 2,792 MiB. Sampling can miss
  short-lived peaks, and later scene changes may use more memory.
- RIFE-to-VSR input downloads and raw NVENC-path downloads both remain zero.
  Only the separate NDI/JPEG branch downloads preview pixels.

## Easy manual revert

Change only `gpu_video.height` in `tulip/output-presentation.json` from `2160`
to `1440` using an atomic file replacement. The running service applies this
at a chunk boundary and recreates its publisher; no model restart is needed.
Keep the current bitrate, native preset, interpolation, and creative controls.
Restoring the entire old file can overwrite later settings changes.

## Evidence

`tulip/evidence/4k-live-20260924/` contains the pre-change configuration and
source copies, lifecycle snapshots, code hashes, receiver samples and summary,
production observer data, and `verification-summary.json`. The original
1440p favorites remain available. Earlier isolated 1440p/4K/bitrate comparisons
are in `tulip/evidence/output-4k-20260924/`.

The selected 4K feed remains live; test receivers exited after measurement.
