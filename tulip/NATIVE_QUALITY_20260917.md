# JoyAI native-quality tuning ? 2026-09-17

User request: improve detail before upscaling while balancing resolution, interpolation and delivery speed.

## Current configuration (2026-09-18)

840x480 at 12 native fps plus 2x RIFE on the output GPU for 24 presentation fps, compiled VAE and stable prompt graphs enabled. Every supported prompt-capacity graph is now captured before playback. GPU placement and SR/output geometry below remain unchanged. See [robust playback and current validation](ROBUST_PLAYBACK_20260918.md). The intermediate native-only 24-fps configuration below could not maintain its reserve under longer prompts.

## Previous configuration and baseline measurements

840x480 at requested 24 native fps, two denoising steps, seed 42, temporal limit 8, RIFE off. Native pixel count is 72.8% above the former 648x360. Physical GPU1 holds the FP8 DiT and compiled BF16 VAE; physical GPU0 holds the resident BF16 text encoder, neural SR and NVENC output. Sage variable-length graphs are enabled; timing instrumentation is off. SR denoise .25; desktop preview 1296x720, direct NDI 1944x1080, GPU video 2592x1440 at 15 Mbps.

The 121.94-second warm observation measured 23.59 fresh/model, canvas and GPU frames per second; no counted output drops or service errors, median capture-to-canvas latency 1094 ms, maximum sampled GPU queue five frames. Largest logged sender gap was 116.9 ms, so this is approximately 24 fps, not perfectly uniform frame spacing.

Concurrent local WebRTC receiver: 120 samples over 119.23 seconds; 2,828 frames decoded at 2592x1440; zero decoder drops, freezes or lost packets; final reported rate 24 fps; largest video-frame callback gap 133.9 ms. This verifies the local receiver, not an unobserved phone or external network.

Two prompt revisions changed token length 518 -> 519 -> 518, applied successfully in the same session, and restored the exact original prompt. Largest delivery gap during that test was 903 ms. Variable-length Sage graphs retain this recapture tradeoff; the previous masked graph avoids it at lower measured native throughput. Original prompt and shader strength were preserved, and SDK operations were resumed.

The chosen preset prioritizes real native detail and fresh-frame rate. RIFE2 with neural SR did not sustain 48 fps even after the two-GPU presentation pipeline improvement, and its queued GPU output dropped frames. It is not left enabled merely to advertise a higher presentation frame rate.

Reproduction: performance-settings.json has compileVae=true, fixedPromptGraph=false, vaeDevice=model, rifeDevice=output (inactive with RIFE1; updated in the follow-up below). native-quality.json has width=840, height=480, fps=24, rife_multiplier=1. output-presentation.json retains neural upscaling and original 1080/1440 delivery sizes with denoise=.25. These settings survive SDK-managed restarts.

Final evidence: final-steady.json, final-steady.json.png, final-receiver.jsonl, final-prompt-check.json; session-161915 and deployment-161047 in the evidence directory below.

## Follow-up: interpolation on the other GPU

At the user's request, RIFE was reassigned to physical GPU0 (logical cuda:1), alongside SR, resident text encoding and NVENC, using a managed runtime restart. The improved bounded presentation pipeline was tested with unchanged 840x480/24 input, 1080p NDI and 1440p GPU video. Over 41.727 warm seconds, 2x RIFE delivered 15.48 fresh/native fps and 30.92 GPU output fps, with zero counted drops or service errors, median capture-to-canvas latency 2008 ms and maximum recorded sender gap 149.2 ms. The publisher queue peaked at one frame: delivery was supply-limited rather than dropping a full queue. This configuration does not sustain the requested 24-to-48 fps path. RIFE was disabled again; its placement remains on the output GPU for future tests. Evidence: deployment-195237 and live-pipeline-othergpu-24to48.json in the same evidence directory. The result measures the combined interpolation/upscaling/output path; it does not isolate RIFE's individual cost or establish a hardware ceiling.

Restoration check: live-othergpu-restored-24.json observed 41.615 warm seconds at 23.60 native fps / 23.69 GPU output fps, zero counted drops or service errors, and median capture-to-canvas latency 1101 ms. Persistent native configuration remains 840x480/24 with RIFE1 (disabled).

## Official reference

Final bidirectional V2V foundation training: 720p/24 fps (paper Table 1). Official 5090 deployment: 840x480/24 fps, FP8/Sage/low-VRAM. These are not measured Etherea results.

Sources: https://arxiv.org/html/2608.03974v1#S3.T1 ; https://github.com/jd-opensource/JoyAI-Video-Edit/blob/main/DEPLOYMENT.md#4-launch

## Matched native trials

Same recorded Easel shader, coral prompt, seed 42, two steps, FP8 DiT/BF16 VAE, temporal limit 8. 193 generated frames per size; first 33 excluded from warm throughput. Same source mirror-loop; dimensions change latent noise shape, so images are not pixel-aligned quality pairs. Videos play at nominal 16 fps and do not represent elapsed processing time.

| Native | Warm generated FPS | Chunk p95 ms | Peak allocated GiB |
|---|---:|---:|---:|
| 648x360 | 30.53 | 270.1 | 19.40 |
| 768x432 | 24.46 | 350.5 | 20.50 |
| 840x480 | 20.52 | 431.6 | 21.32 |
| 864x480 | 19.97 | 451.0 | 21.47 |

Native benchmark excludes RIFE/SR, transport and pacing. Each native arm runs for seconds, not an endurance test.

## Matched presentation-stage trials

Same final 32 raw native frames, repeated across eight bounded chunks; first two excluded. RIFE on GPU 1, SR on GPU 0. JoyAI weights were unloaded during these presentation-only trials. SR includes GPU-to-CPU output readback; no NDI/NVENC/browser delivery. These are stage capacities, not full model FPS.

| Native | RIFE | Output | Denoise | Input-equivalent FPS | Presentation FPS |
|---|---:|---:|---:|---:|---:|
| 648x360 | 1x | 1440p | 0.5 | 115.74 | 115.74 |
| 648x360 | 2x | 1440p | 0.5 | 41.78 | 83.57 |
| 648x360 | 4x | 1440p | 0.5 | 21.01 | 84.05 |
| 648x360 | 2x | 1080p | 0.5 | 47.75 | 95.50 |
| 648x360 | 2x | 1440p | 0.25 | 44.50 | 89.00 |
| 648x360 | 2x | 1440p | 0.0 | 43.00 | 86.01 |
| 864x480 | 1x | 1440p | 0.5 | 68.71 | 68.71 |
| 864x480 | 2x | 1440p | 0.5 | 27.37 | 54.73 |
| 864x480 | 4x | 1440p | 0.5 | 12.87 | 51.48 |
| 864x480 | 2x | 1080p | 0.5 | 28.06 | 56.13 |
| 864x480 | 2x | 1440p | 0.25 | 27.16 | 54.31 |
| 864x480 | 2x | 1440p | 0.0 | 27.07 | 54.15 |

## Live trials

First 864x480/16 + RIFE2 + 1440p (.25 denoise): 41.47s warm observation, 13.86 fresh model FPS / 27.68 GPU output FPS; zero counted queue drops, but repeated empty-queue delivery gaps, including 439ms. REJECTED as a smooth 32fps preset. This shows isolated capacities do not prove live performance.

Moving RIFE to the output GPU alone measured 13.03 native / 26.09 output fps and did not help. The 480p/12 + RIFE2 arm measured 11.87 native / 23.76 output fps, still with delivery gaps. Neither establishes a hardware ceiling.

## Upstream-path investigation

The earlier local fixed-capacity prompt graph uses masked cuDNN attention, bypassing Sage for the denoise graph despite JOYOMNI_SAGE_ATTN=1. This was introduced to avoid graph recapture during prompt-length changes. It is a throughput/transition tradeoff, not an upstream requirement.

Matched 481-frame 840x480 trials (first 33 excluded): eager VAE with profiling off and fixed prompt graphs: 20.33 native fps; same setup with variable-length Sage graphs: 25.19 native fps. Profiling removal alone did not materially improve the previous 20.52 result. These remain isolated native tests.

All four native VAE warmup shapes compiled, but the first real streaming encoder call failed with OverflowError in PyTorch's static CUDA launcher. This matches https://github.com/pytorch/pytorch/issues/162430 (Windows stream handle converted through C long). The Windows compatibility path now disables only that static launcher, retaining Torch compilation and Triton's normal launcher. The subsequent 481-frame streaming tests passed.

| Native 840x480 path | Warm native fps |
|---|---:|
| Eager VAE, fixed masked prompt graph, profiling off | 20.33 |
| Eager VAE, variable-length Sage graph | 25.19 |
| Compiled VAE on second GPU, Sage graph | 24.72 |
| Compiled VAE beside DiT, Sage graph | 29.47 |

Compiled VAE beside the DiT had a 296.3 ms chunk p95. Splitting VAE onto the output/text GPU did not improve throughput. The two-GPU placement changes random-number streams used in VAE sampling; identical seeds do not make those generated images pixel-aligned. Reviewed representative frames contain coherent expected content.

Larger native sizes were tested with Sage and eager VAE to avoid another cold compile per size. 1056x600 measured 16.27 fps, median DiT stage 387.5 ms/chunk; 1248x720 measured 10.44 fps, median DiT stage 495.8 ms/chunk. Those stage timings exceed the 333.3 ms eight-frame budget, but include contention with concurrent VAE work. They are not isolated DiT hardware ceilings, compiled-VAE measurements at those sizes, or a claim about future tensor-parallel implementations. Larger compiled configurations could improve; 480p/24 is the demonstrated native-rate target.

Cold compilation plus load initially took approximately nine minutes; the full successful split-VAE arm took 593 seconds including first streaming-shape compilation. A subsequent cached load for compiled VAE on the model GPU took 312 seconds; a later service restart was faster. Compilation startup cost is separate from the warm rates above.

Live scheduling fix: push_frame can leave completed results for the dedicated output pump. Previously the input loop could steal a result and block input reception during RIFE/SR, competing with the output pump. The WebSocket path now has one presentation consumer; offline push_frame retains its original draining default. Focused consumer and stream-control tests: 15 passed.

Explicit machine-local performance-settings.json selects compileVae, fixedPromptGraph, vaeDevice (model/output), and rifeDevice (model/output). The launcher imports the existing MSVC environment when compilation is selected. JOYOMNI_WARMUP_NATIVE_ONLY skips speculative portrait/reference warmups. On Windows, arbitrary reference-image sizes stay eager unless JOYOMNI_COMPILE_REFERENCE=1 is explicitly selected; native video encode/decode remain compiled. Launcher parsing and setting validation: eight checks passed; reference policy: two tests passed.

First-frame prompt/kernel initialization now has a separate 120-second deadline and matching initial idle grace; ongoing frame submissions retain the 15-second guard and established idle sessions retain the 10-second guard. Earlier connections expired at 15 seconds or the independent 10-second idle guard even though initialization completed shortly afterward. The final managed cold restart succeeded without either error and restored the saved prompt and strength.

## Optimized live presentation trials

Each row excludes the first five seconds and observes about 41 seconds of warm output. All rows used 840x480 at a requested 24 native fps, compiled VAE beside DiT, Sage, profile timings off, and one dedicated output consumer. Zero counted queue drops does not imply gap-free delivery.

| RIFE | NDI / GPU height | Fresh fps | GPU output fps | Median latency ms | Largest recorded delivery gap ms |
|---|---|---:|---:|---:|---:|
| 1x | 1080 / 1440 | 23.95 | 23.97 | 1186 | 90.5 |
| 2x on output GPU | 1080 / 1440 | 15.47 | 30.88 | 1520 | 265.9 |
| 2x on output GPU | 720 / 1440 | 19.29 | 38.59 | 1222 | 138.5 |
| 2x on output GPU | 720 / 1080 | 19.80 | 39.57 | 1216 | 127.8 |

RIFE on the model GPU with 720/1080 outputs measured 19.46 native / 38.97 GPU fps, with a 294 ms maximum recorded delivery gap. Moving RIFE alone did not solve the serial presentation bottleneck.

An ordered pipeline now overlaps one chunk of RIFE with SR of the previous chunk. Separate interpolation and SR locks are both held exclusively during prompt graph capture; cancellation drains outstanding interpolation before session reuse. The browser permits at most 32 outstanding native frames when both stages are enabled (24 with SR alone, 16 without SR). Output ordering, overlap bounds, cancellation and prompt-control tests passed; client suite 37 passed, scene/capture-clock checks 7 passed.

Pipelined 480p/24 + RIFE2, 720/1080 outputs: 21.06 native / 42.14 canvas / 40.13 GPU fps. GPU queue reached its 48-frame bound and dropped 80 frames in the warm window. This configuration is rejected as a sustainable 48 fps output.

The CPU/download NDI branch is material overhead at 48 presentation fps. Merely reducing the separate GPU video height provided little additional gain. The final choice prioritizes the requested native detail and fresh-frame rate: 840x480/24, RIFE1, neural SR denoise .25, NDI1080 and GPU1440. Final prompt and receiver results are recorded above.

Optimization evidence: quality-tuning-20260917-141403/deployment-150235 and deployment-153311; live-compiled-*.json in the same evidence root.

## Implementation

Optional machine-local `tulip/native-quality.json` accepts bounded sizes 648x360, 768x432, 840x480, 864x480, FPS 8/10/12/16/20/24, and RIFE 1/2/4 with output <=64 FPS. Changes apply at a new session; weights stay loaded. Invalid settings fail explicitly. Remove the file and restart the session for 648x360/16 baseline.

Client capture geometry and clock follow the server acknowledgement. Independent NDI input allows up to 864px. SR accepts only bounded, matched native frames; output stays 720p preview, 1080p fallback and 1440p mobile. SR denoise is hot-configurable in output-presentation.json under the existing output lock.

Evidence: `C:\Users\Admin\src\joyai-video-edit\tulip\evidence\quality-tuning-20260917-141403`. Current worktrees contained substantial pre-existing changes; no commits made.

## Validation

Client suite 37 passed; provider suite 5 passed; final native/output/control/progressive/shader/pipeline/reference-policy CPU tests 25 passed; scene/capture-clock checks 7 passed; launcher parsing and setting checks 8 passed. Updated stale client test expectations for current 32fps NDI metadata and asynchronous shader preparation. Managed cold restart, live prompt changes, final visual inspection and local receiver validation completed. No commits or pushes made.
