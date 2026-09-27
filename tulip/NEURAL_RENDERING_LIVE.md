# Live Neural Rendering trial on Tulip

September14,2026. User-approved community DLSS NR310.8.0.0 on NVIDIA616.92.

Pipeline: JoyAI648x360 -> native RIFE2x -> Real-ESRGAN2x -> Neural Rendering at1296x720, Natural style, intensity1.0, color/detail1.0 -> existing JPEG/WebSocket/canvas/NDI output. NR never feeds model history.

The optional stage reads `tulip/neural-rendering.json` at chunk boundaries. No file means disabled. The live trial supports adapter0 only. The persistent subprocess is the locally rebuilt `tulip/research/video2dlssnr/out/video2dlssnr.exe`, using the explicitly approved community runtime copied from the successful offline trial. The runtime's invalid NVIDIA Authenticode signature is known and was disclosed before authorization.

Only the upstream video writer was changed: fflush(stdout) after each output frame. Without this, a partial CRT buffer can withhold the end of a frame and deadlock a live one-frame request. The loader/runtime algorithms are unchanged. NVIDIA NGX link library comes from NVIDIA/DLSS, lib/Windows_x86_64/x64/nvsdk_ngx_d.lib. Source and runtime binaries remain local trial assets, not tracked vendor redistribution.

One output chunk of1..16 frames is submitted at a time, with two pipe-I/O workers and the tool's bounded internal queues. There is no growing Python frame queue. The helper is warmed during model startup, then reuses its NR model and optical-flow history. SR bypass discards that history and returns native frames.

A helper failure or15s startup/3s warm chunk deadline terminates only the owned helper, latches NR off and returns the entire original SR chunk. It does not mix partial NR output with SR output. A failure latch requires a JoyAI restart to retry. Ordinary on/off config changes need no model restart.

Rollback while live, in PowerShell from the JoyAI repository:

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./tulip/set-neural-rendering.ps1 -Mode off

Enable again (if no failure was latched):

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./tulip/set-neural-rendering.ps1 -Mode on

This helper updates JSON atomically. Do not edit the file in place during rendering. Removing the config also disables the stage. The SR tile toggle continues to work independently; NR is applied only when SR output is1296x720.

Observability: `/health.upscale.neural_rendering`, `output_frame.profile.upscale.neural_rendering`, and `tulip/neural-rendering-live.log`. Check enabled, loaded, error and increasing frame count; an enabled config alone is not proof of rendered output.

Tests: `tulip/test_neural_rendering_live.py` passed actual single-frame pipe flush, persistent16-frame chunk, matching geometry, nonidentity output, killed-helper fallback and disable cleanup. Measured warm helper chunk0.578s for16 frames under concurrent show load. Existing `tulip/test_upscale_pipeline.py` passed2 tests covering real RIFE/SR, native bypass, geometry and SR failure behavior. Do not run GPU tests during a measured live observation interval.

The model reload is orchestrated through SDK actions/adapters in `easel-agent-sdk/logs/enable-live-nr-20260914.py`; snapshots preserve the current prompt/strength and Easel config. Live measurement evidence and final results are recorded separately after activation.

## Trial outcome

Disabled again at user request because of jitter. Measured29.308s steady interval:20.30 displayed fps,20.27 NDI fps,2169ms median capture-to-canvas latency,196ms p95 callback gap and782ms maximum gap. NR had no inference error; it exceeded the live processing budget. Existing32fps playback pacing drained its buffer and produced irregular frame delivery. The community DLL loaded successfully; this was a throughput/cadence regression. The local config is now disabled and the helper was confirmed stopped without reloading JoyAI.

Activation evidence: etherea-ai/logs/nr-live-20260914.jsonl. Recovery evidence: etherea-ai/logs/nr-rollback-20260914.jsonl. The initial PowerShell rollback wrapper hit execution-policy and null-backup argument errors; the config was then disabled atomically using Python. The wrapper has been corrected to explicit on/off modes with a named backup and successfully tested in off mode.

Recovery measurement:29.323s,30.45 displayed/NDI fps,1368ms median latency,37ms p95 callback gap. NR enabled=false,loaded=false; frame count stopped at1602. RIFE2x and Real-ESRGAN2x remain enabled at1296x720. One683ms maximum callback gap remained in the interval, so this does not establish stall-free playback across scene changes.
