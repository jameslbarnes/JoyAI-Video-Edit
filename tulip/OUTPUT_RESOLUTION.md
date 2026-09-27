# Presentation output resolution

The native model and RIFE remain at 648x360. RealESRGAN general-x4v3 already computes 2592x1440; the final bicubic antialiased resize selects the delivered size.

Optional machine-local `tulip/output-presentation.json`:

```json
{"height":1080}
```

Supported heights: 720 (1296x720, default) and 1080 (1944x1080). Both preserve the model aspect ratio. The setting is read once per presentation chunk, under the existing output lock. Invalid/partial edits retain the last valid setting and expose `config_error` in health. Deleting the file restores 720. Resolution changes do not reset generation, prompt, temporal history, or RIFE.

Rollback: atomically replace the JSON with `{"height":720}`. The client accepts native, 720, and 1080 frames and resizes only upon decoded frame arrival. The JoyAI NDI export permits width up to 1944; other providers retain their existing caps. The existing WebRTC publisher detects NDI geometry changes and restarts/re-probes its fixed-size rawvideo pipe, briefly interrupting delivery.

Keep RIFE at 2x and neural rendering disabled for this trial. The optional NR helper currently supports only 1296x720 and bypasses other shapes. This does not raise native generation resolution or add model detail beyond the existing SR result.

Higher resolution increases JPEG, GPU/CPU transfer, canvas readback, raw NDI transport, and receiver decode costs. Validate fresh NDI ingest and WebRTC playback, not just the paced NDI sender count.

Validation: `python tulip/test_output_geometry.py`, full SR module import, JS syntax checks, actual client drawOutput native/720/1080/rollback acceptance checks. Live trial evidence and outcome are recorded separately.

## 2026-09-15 live trial: reverted to 720

Deployed through SDK JoyAI runtime lifecycle and app.reload, preserving current prompt/strength. The first deployment hit a default-argument initialization typo; corrected it, verified full module import, then started successfully. GPU text residency, RIFE2, NR disabled verified after restart.

At 1944x1080, the 114-second observer measured 30.91 canvas fps, 32.00 paced NDI fps, but only 13.25 fresh NDI ingests/s. Output drops and NDI send errors were zero, which alone did not detect the duplicated frames. Capture-to-canvas latency was approximately 1.5 seconds. A 90-second WHEP probe confirmed actual1944x1080, 2853 decoded frames, zero packet loss/dropped frames, and one237ms freeze. This decoder success did not imply smooth source motion: repeated images still encoded at32fps.

Rolled back by atomically setting output-presentation.json to height720, without restarting/reloading the model. A subsequent34-second observation measured32.29 display/fresh NDI fps and32.00 paced NDI fps, zero output discards/send errors, latency1298ms. Confirmed1296x720, RIFE2, SR enabled, no SR/config errors. Keep720 active. The delivered-frame bottleneck must be addressed before retrying higher resolution; do not simply increase buffering.

Evidence: SDK logs/output-1080-soak-20260915.json and output-720-rollback-soak-20260915.json; Etherea logs/output-1080-whep-20260915.jsonl. The followup1080 probe overlaps the rollback and is not a clean1080 sample. Final rollback WHEP verification: Etherea logs/output-720-rollback-whep-20260915.jsonl.

Final30-second rollback WebRTC check:1296x720 at32fps, zero dropped frames, freezes, or packet loss.
