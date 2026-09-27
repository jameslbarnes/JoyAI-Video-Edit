# Direct JoyAI NDI output

Optional local `tulip/output-presentation.json`: `{"height":720,"direct_ndi":true}`. Height1080 enables1944x1080 main output while the browser preview stays1296x720. RIFE stays on native frames before SR. Model input, text placement, prompts, and creative controls remain upstream and SDK-orchestrated.

Data path: raw SR RGB arrays -> bounded paced worker -> local NDI bridge raw TCP -> existing NDI receivers/WebRTC publisher. Each due frame is converted toRGBA with OpenCV, then sent via the existing EVSF framing on127.0.0.1:7780. No browser JPEG decode/readback/export is involved in main output. The preview resize and JPEG encode run together in the existing four-worker pool. The browser continues to receive a720 JPEG preview and to supply model input/control; this is not yet a fully headless inference session.

The worker owns one high-resolution monotonic clock (`perf_counter`; Windows GetTickCount64 is too coarse),350ms initial buffer, up to48 pending frames (plus one in flight; extra headroom accommodates a scene morph), and a500ms socket deadline. It drops oldest queued frames on overload rather than accumulating latency; counters are exposed in `/health` upscale.direct_ndi and frame profile.direct_ndi. Transport failure latches direct publishing off and releases the socket; the client resumes its720 browser NDI fallback. A latched error currently needs a model runtime reload to retry direct publishing. Normal config disable/re-enable does not require a reload. Final session teardown cancels the worker and clears frames; prepared scene changes retain the shader bridge and clear only old model frames; the warm NDI handle is retained by the existing primary lifecycle.

Both paths use the existing primary-joyai sender/ETHEREA JoyAI name. The bridge retires the browser pacer on raw ingress, counts fresh raw frames, and rejects competing browser submissions while the raw owner is connected. Browser output resumes after the direct socket closes. Other model senders retain their existing behavior.

Rollback: atomically set `{"height":720,"direct_ndi":false}`. This switches at a chunk boundary without a model restart. The WebRTC supervisor re-probes when output dimensions change; viewers may briefly reconnect. Scene-transition support was added on 2026-09-15; see the transition correction below.

Tests: `python tulip/test_direct_ndi_output.py` (real loopback protocol/order/pacing/shutdown; bounded queue/shape transition; connection failure latch;1080 publisher +720 preview), `python tulip/test_output_geometry.py`, full server module import. Etherea node tests: scripts/streaming/ndi-direct-owner.test.mjs and ndi-pacer.test.mjs. No offline GPU benchmarks against the live workload.

Deployment uses SDK logs/deploy-direct-ndi-20260915.py, saving the current prompt/strength, holding supervisor recovery, reloading JoyAI and the app, and restarting only the NDI bridge through its existing supervisor API. No SDK public API/adapter contract changes.

## Initial checks and corrections

Initial720 direct trial:39seconds,32.05 browser fps,31.76 fresh NDI fps, no NDI errors; browser NDI pump absent and one raw owner connected. Replaced Python3.11 Windows monotonic/GetTickCount64 (15.625ms resolution) with perf_counter/QueryPerformanceCounter (100ns reported resolution).

Initial1080 direct trial reached22.50 fresh NDI fps. Telemetry identified serial720 preview resizing inside the output thread costing approximately300ms/chunk. Moved preview resize into each existing bounded JPEG worker; directly published1080 pixels remain unchanged. Reverted to720 before reloading corrected modules through the SDK. These initial samples are not final acceptance evidence.

The first corrected180-second1080 sample measured31.89 fresh NDI fps versus32.20 browser frames/s. Two-minute WHEP verification decoded3820 frames at1944x1080 with zero drops/freezes/loss. The small surplus producer rate caused bounded queue discards; client capture used a fractional setInterval period subject to truncation. Replaced it with absolute fractional deadlines and missed-slot skipping; simulated five-minute test produces exactly4800 native captures. Loaded via SDK app.reload with the model resident, also exercising720 browser fallback before final1080 adoption.

## Adopted live state and evidence

Final local config: height1080, direct_ndi true. Main NDI/WebRTC1944x1080, browser preview1296x720; RIFE2, GPU-resident text, NR disabled. The final SDK observer confirmed rawClients1, sourceType joyai-direct, browser output pump absent.

Final114-second observation: 31.13 fresh NDI fps, zero additional direct-publisher queue discards, zero NDI send errors; three browser-preview buffer discards. Queue remained bounded. Two prompt graph updates (approximately0.55/0.61s capture preparation;0.97/1.10s combined encoding/preparation) occurred during this run. Frame-production gaps remained around prompt changes, including a roughly one-second interval with no fresh output; direct transport does not eliminate model-side pauses. Do not equate32fps decoded RTP with continuous new images during those pauses.

Final90-second local WHEP receiver: 2877 decoded frames,1944x1080 at32fps, zero decoder drops/freezes or packet loss. No mobile-device test was available.

Fallback was explicitly exercised: direct disabled, rawClients0, browser-canvas owner restored,31.97 fresh fps at720 and zero send errors. Then direct1080 was re-enabled without another model reload.

Evidence: SDK logs/direct-ndi-1080-adopted-20260915.json, direct-ndi-fallback-20260915.json, direct-ndi-1080-adoption-summary-20260915.json; Etherea logs/direct-ndi-1080-adopted-whep-20260915.jsonl.

## WebRTC bitrate trial, 2026-09-15

After reported pixelation, raised the generated hermes-screen NVENC target from4500 to10000kbps (maxrate/buffer13334k) in the machine-local bin/etherea-webstream.env. This compensates for2.25x the pixels at1944x1080. Only the owned output FFmpeg process was restarted through the SDK-controlled deployment script; model, RIFE, direct publisher and desktop preview were unchanged.

60-second local WHEP sample measured9.994Mbps,1944x1080 at32fps,1906 decoded frames, zero packet loss, decoder drops or freezes, maximum jitter3ms. Parallel59-second fresh NDI observation:31.31fps, zero NDI errors/browser discards, no additional direct publisher discards, start/end capture-to-canvas latency1396ms. Local delivery passed; subjective image quality/mobile link require viewer feedback.

Active target is10Mbps. Backup: SDK logs/restart-20260915-140611/etherea-webstream.env. Evidence: Etherea logs/direct-1080-10mbps-whep-20260915.jsonl; SDK logs/direct-1080-10mbps-soak-20260915.json. To roll back compression settings, restore4500kbps and restart only the owned hermes-screen FFmpeg encoder; no model restart.

## Scene transition correction, 2026-09-15

The original direct-output upgrade omitted the browser shader fade and closed its sender during scene reset. On prepare_scene the publisher now fades current model output into the existing native shader input over 600 ms, on its existing paced CPU worker. A prepared scene restart clears old queued model frames while retaining the socket and shader bridge. The bridge waits for the RIFE morph plus its following real chunk before resuming normal raw output. There is no additional model session, GPU interpolation pass or steady-state browser export. Shader bridge pixels come from the existing 648x360 model input resized to the output geometry; generated scenes retain full SR quality.

The desktop client no longer overlays another 600 ms fade on frames already marked scene-morph. Non-morph transitions retain the ordinary fade. Preparation failure cancels the bridge; teardown clears its retained images/callback; a 20-second lease prevents indefinite retention if control fails. Transition counters and phase are included in direct publisher status. Normal playback retains its prior bounded queue and pacing.

CPU tests cover fade endpoints, live shader updates, waiting for the following real chunk, cancellation/expiry/cleanup and continuous loopback socket cadence through reset. JS tests cover buffer priming and absence of the duplicate morph fade. Deployment: SDK logs/deploy-scene-transition-20260915.py. Live verification is recorded separately below when complete.

First live transition check: all 15 desktop morph frames played; the shader bridge supplied 79 paced frames, the raw socket remained connected throughout, and every 200 ms sample advanced by at least two fresh frames. However, the 32-frame queue discarded seven frames immediately after handoff: the one-time 15-frame morph requires headroom beyond the ordinary two-chunk reserve. Raised the hard queue capacity to 48, leaving its normal 350 ms startup priming and 32 fps clock unchanged. This adds at most 16 retained 1080p RGB frames (about 96 MiB of CPU memory), not GPU work or an unbounded buffer. A new scene clears the prior queue, so the reserve cannot accumulate across resets. Final validation follows.

Final live reset validation (SDK logs/restart-20260915-151902/scene-transition-observation.json): existing prompt unchanged, desktop played all 15 RIFE morph frames and completed transition, direct publisher supplied 73 shader bridge frames and completed one transition. Raw owner remained connected; every 200 ms sample advanced by at least five fresh frames. Measured 31.84 fresh NDI fps, zero queue discards, no direct publisher error, 1944x1080 at 32 fps, RIFE 2x, reference inactive. Hard queue bound is 48; it ended at 36. Eight CPU direct-output tests and three JS tests passed. Runtime deployment backup: SDK logs/restart-20260915-151530/.

Post-transition observation (SDK logs/scene-transition-final-soak-20260915.json): measured 24 seconds after warmup at 32.02 fresh NDI fps, zero browser output drops or NDI errors, direct total drops still zero, queue ended at 28 of 48. Browser capture-to-canvas latency fell from 1516 to 1368 ms. This is a short live check, not a long-term soak.

Texture-control routing correction: the deployment app.reload restored dashboard mode longlive, and manually recreating JoyAI without SDK primary routing left joyaiNdiSource absent. Phone texture changes still landed on Easel FLUX-IN but the model consumed the other visible source. Restored via model.primary.select (resident model, no reload), verified mode ndi and a fresh independent TULIP (Easel - FLUX-IN) input. SDK logs/deploy-scene-transition-20260915.py now reasserts configured_primary.input through shader.setInput after reload, before adding JoyAI, and refuses success unless the tile consumes fresh frames from that source. Evidence: SDK logs/restart-20260915-152818/shader-input-restored.json.
