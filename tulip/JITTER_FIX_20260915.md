# Progressive jitter: findings and deployed correction

The running session fell out of CUDA graph execution after prompt token length
changed. `_apply_prepared_prompt` intentionally deferred capture because VAE
workers remained active, but no later path prepared that new shape. The session
then stayed eager until restarted. Graph keys include the exact text length;
this explains why a session-only restart restores speed without reloading weights.

Before session reset: about 16 fps, 235 ms p95 display gap, 855 ms worst gap;
model chunk processing reached ~896 ms. Restarting only the NDI bridge did not
repair generation (17.8 fps). Restarting only the JoyAI session restored graph_path=1,
~256 ms model chunks, and 32.02 displayed fps.

## Correction

- A submission lock pauses new model jobs during preparation of a missing prompt
  graph. Already-submitted postprocess and pseudo-encode jobs drain before capture.
- A separate presentation lock excludes RIFE/SR calls while capture runs. It is
  acquired inside worker threads, never across an asyncio await on the event loop.
- Synchronize device work before capture. Preserve current embeddings, temporal
  state, pending frames, and KV entries. The next chunk applies/binds the prepared
  prompt. No token padding, truncation, periodic reset or quality reduction added.
- Existing graph shapes skip preparation. Superseded requests and session close
  cancel the drain; a ten-second drain bound prevents indefinite waits.

The 13 stream-control tests cover stage drain ordering, unchanged-shape reuse,
stale request cancellation, exception cleanup and latest-request-wins behavior.
The 5 existing strength tests also pass.

## Live regression

Used exactly the current prompt's words; replaced one internal space with a
newline, then restored the original. Graph text length changed 445 -> 446 -> 445.
Both captures took ~0.4 seconds. Both subsequent revisions reported graph_path=1.
Session epoch stayed 1 and KV cache stayed at two entries. No prompt/scene reset.
Evidence: Etherea logs/jitter-prompt-graph-20260915.json.

## Other output-path findings

The desktop outgoing frame socket also accumulated 8-11 MB and starved fresh NDI
frames while its display continued near 32 fps. A desktop-only SDK app.reload
cleared that transport condition; the precise cause of that socket deterioration
was not isolated. Avoid attaching broad browser Network instrumentation to the
live raw-frame socket when collecting timing evidence.

The always-on microphone branch bypassed the chat history bound and retained
1,217 entries (~50k DOM elements). That branch now trims to the intended 50 entries
and removes retired event-map references. This is a confirmed retention defect;
its causal contribution to the socket slowdown is not established. A 2,000-entry
CPU regression verifies the history and lookup remain bounded.

The separate WebRTC publisher was hard-coded at 30 fps despite JoyAI/NDI 32 fps.
The machine-local webstream launcher now chooses the probed source FPS (overridable
with WEBSTREAM_FPS). Full-resolution WebRTC runs at 32; chandelier output remains
30 via its own filter. The encoder pipeline was restarted through SDK orchestration.

## Verification after deployment

A 120-second control/bridge sample measured 32.23 displayed fps, 32.00 NDI sends/s,
and 32.24 fresh ingests/s, with zero NDI errors (two client output discards during
the sampled period). Latest latency snapshots were 1.29-1.30 seconds.

A separate 90-second local WebRTC receive decoded 2,864 frames at 32 fps with zero
frame drops, freezes, packet loss, or post-startup PLI increases; maximum RTP jitter
was 3 ms. The previous 40-second receive decoded at 30 fps. Local WebRTC validation
does not prove performance of an unobserved phone/network connection.

Evidence: SDK logs/jitter-final-soak-20260915.json; Etherea
logs/jitter-generated-whep-20260915.jsonl and logs/jitter-fixed-whep-20260915.jsonl.
All model quality settings retained: RIFE 2x, RealESRGAN 2x, NR off, full BF16 GPU
text encoder. Live configuration and private snapshots remain local.

Follow-up after a 309-second observation span: 32.40 fresh frames/s into NDI, 31.98 paced NDI sends/s, zero additional client discards and zero NDI errors. Graph execution remained active, queues empty, and latest latency was 1327 ms.
