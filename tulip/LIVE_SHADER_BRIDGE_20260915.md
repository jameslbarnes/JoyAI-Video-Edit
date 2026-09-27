# Moving shader scene transitions

This supersedes the still-endpoint entrance morph for clients supporting the
new bridge. Ordinary RIFE2 interpolation remains enabled.

The previous bridge read `_latest_source_frame` from the16fps model input.
The desktop NDI receiver was also configured for16fps. Fresh32fps transport
therefore hid repeated shader images, while initialization/teardown could
temporarily stop model-input updates. The15-frame RIFE entrance interpolated
two still endpoints, causing a separate motion discontinuity.

Changes:

- Etherea's JoyAI NDI source now requests32fps; model capture remains16fps.
- A temporary `/ws/shader-bridge/{uuid}` connection delivers648x360 JPEGs on
  an independent32fps clock, regardless of model readiness or input credit.
  One frame remains outstanding until decode acknowledgement. The runtime
  stores only the latest immutable CPU frame, rejects incorrect dimensions
  and payloads above512KiB, expires stale data, limits connections to two,
  and closes idle/expired connections. No shader frames enter model caches.
- A successfully established bridge selects a moving600ms crossfade on
  both desktop and direct output. Generated frames and shader frames advance
  throughout the fade. The synthetic still-endpoint entrance is omitted.
  Normal generated-frame RIFE interpolation is unchanged. Legacy clients or
  failed bridge setup keep the old transition behavior.
- Direct output waits for half a second of generated reserve before fading
  in. The existing bounded queue, pacing and socket remain in use.

Deployment via SDK: `logs/restart-20260915-211224/`. Initial client reload
returned before shader controls were initialized; an auto-started tile then
needed replacement. Restoration completed in
`logs/restart-20260915-211633/`, verifying saved prompt/strength, independent
fresh Easel FLUX-IN input, masked prompt graph and GPU-resident text encoder.

Validation:35 Python and5 JavaScript tests passed. Targeted Python tests were
rerun after handling superseded bridge sources. The JS clock test explicitly
blocks model input and still gets32 shader sends in one simulated second.

Live evidence in SDK:

- `logs/restart-20260915-211654/`:25-second observation and15-second recording.
  31.92freshNDIfps;29.39shader input frames/s;97bridge frames with max41.1ms
  interarrival;0output drops; max desktop callback gap44.6ms. An additional
  creative scene request arrived near the test deadline, making the final
  "transition ended" assertion fail. Subsequent readback confirmed that
  second transition completed with0drops and no error.
- `logs/restart-20260915-211928/`:focused12-second test passed.31.98freshNDIfps;
  28.43shader input frames/s;83bridge frames with max40.9ms interarrival;
  0output drops; moving transition completed at2.60s. Max desktop callback
  gap54ms; p9941.4ms. No synthetic entrance frames, RIFE2 unchanged.
- Both directories contain `moving-scene.mp4`, observations and callback gaps.
  Offline pixel-difference inspection shows fewer near-repeated frames in
  the shader interval than the earlier16fps bridge recording. Different
  creative content means this is not a controlled visual-quality comparison.

Etherea `logs/live-shader-bridge-whep-20260915.jsonl`:35seconds of1080p32fps
WebRTC,0decoder drops/freezes/packet loss, maxRTP jitter4ms. This local receiver
does not measure the phone's wireless connection. Actual source delivery
measured28-30fps despite the32fps request; do not equate transport fps with
distinct content frames or claim mathematically jitter-free playback.

Final settings: native648x360/16fps, RIFE2, direct1944x1080/32fps, preview720p,
progressiveSR on, mobile15Mbps, NRoff, no reference image. Temporary frame
callbacks removed; independent bridge sockets expire/close after transitions;
SDK supervision resumed. See client/server capability fallback for rollback;
reverting these source changes requires a runtime/client reload through SDK
and explicit restoration of the configured shader input before tile creation.
