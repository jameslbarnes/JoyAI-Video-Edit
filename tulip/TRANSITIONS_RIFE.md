# Prepared scenes and optional RIFE output

The existing WebSocket now advertises `scene_prepare_supported` in `started`.
Send `{"type":"prepare_scene","prompt":"..."}` while streaming the old scene.
`scene_prepared` returns a token and preparation timing. Consume it with the
normal `start` message plus `scene_token` and the same prompt. This resets KV/VAE
history but uses the already-encoded prompt/image embeddings. One preparation is
allowed at a time, token/prompt are session-local and validated, failures report
`scene_error` without altering the old scene. Updates still use `update_prompt`.

Optional output: launch `tulip/run-service.ps1` with `-RifeFluxRoot` pointing to an
existing FluxRT checkout containing `RIFE-safetensors/flownet.safetensors`, then
request `rife_multiplier:2` in `start`. `set_rife` with multiplier 1/2 switches
output processing without model/cache reset. `started` / `interpolation_changed`
report output_fps (16 or 32). Input FPS remains 16. No package/model installation
into FluxRT is performed; its implementation and checkpoint are read-only inputs.

RIFE uses FP16, pads the 648x360 input to a multiple of 32, and crops results back.
It inserts one midpoint between adjacent generated frames, including continuous
chunk boundaries, but clears history across scene cuts/reconnection/bypass.
Original JPEGs are unchanged; source sequence endpoints and `interpolated` are
explicit metadata. These frames are never used as model conditioning/history.
`profile.rife_s`, generated_count and display_count separate presentation cost.

Tests in the isolated environment:

```
.venv/Scripts/python.exe tulip/test_stream_controls.py
# Set CUDA_VISIBLE_DEVICES and JOYAI_RIFE_FLUX_ROOT in this test process first:
.venv/Scripts/python.exe tulip/test_rife_output.py 1
```

Live Etherea measurements, WebM evidence, exact Tulip launch commands, primary
switching, hashes and rollback are in the companion
`etherea-ai/docs/JOYAI_TRANSITIONS_RIFE.md`. RIFE is output interpolation, not
additional model generation; scene continuity uses a bounded old-scene buffer
and explicitly labeled animated-shader fallback, not simultaneous model sessions.
