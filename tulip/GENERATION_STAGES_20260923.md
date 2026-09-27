# Generation-stage profiling, 2026-09-23

User-approved follow-up to GPU-only RIFE and the preview latency investigation. Preserve 1152x648, 12 native fps, RIFE 2, VSR Ultra, 2592x1440 NVENC at 24 fps and the 500 ms preview buffer. No automatic quality rollback. Current creative controls are captured immediately before each explicit restart and restored only while still owned.

Evidence: `tulip/evidence/generation-stages-20260923`. `before/` contains source/config snapshots and `before-manifest.json` contains hashes. Existing 648p favorites remain untouched.

## Instrumentation

`deploy/xvideo/serving/stage_timings.py` adds opt-in CUDA event spans for graph replay, input VAE encoding, VAE decoding, and the one-frame continuity encoder. Enable with `tulip/stage-profile.json` containing `{"enabled":true}`; disable with `{"enabled":false}`. The file is re-read at most once a second. `/debug.stage_timings` returns bounded samples and summaries. Completion is queried; no CUDA synchronize is added. Startup chunks below 4 are excluded. Stream spans include launch gaps and contention and can overlap; do not sum them or call them isolated kernel times.

The older `profile_timings` remains off: its stage synchronizations would alter overlap. The new observer is `tulip/observe_stage_timings.py NEW.jsonl --seconds 90` (read-only, 10–300 seconds). It ends only its observation, never the renderer. `tulip/profile_vae_placement.py NEW_DIRECTORY --placement baseline|encode|pseudo|decode` explicitly restarts only JoyAI through the existing SDK/supervisor lifecycle. It refuses a changed native profile or a competing operations hold. It has no automatic rollback.

Validation before deployment: 37 focused tests passed (stage collection, shared streams, warmup, compile policy, prompt graphs/controls, and no automatic rollback), syntax checks passed, and the off-device VAE warmup device-context verifier passed. Mock CUDA events assert completion is deferred and no event synchronization is used.

## Baseline

`baseline/steady.summary.json`: 87.883-second sample window with the current 517-token prompt using the 1024-capacity graph. Median spans per eight native frames: DiT 464.770 ms, input encode 55.168 ms, decode 75.493 ms, continuity encode 11.151 ms. Median complete server chunk 635.302 ms; sampled capture-to-canvas latency 1733 ms. Publisher approximately 24 fps (23.952 by separately polled endpoint counters), zero new drops, underruns, restarts or delivery gaps. Both allocators recorded zero retries and zero OOMs through the window. These are desktop preview measurements, not external TV latency.

Decoder placement on logical cuda:1 / physical GPU0 is the first trial because decoding is the largest VAE stage. Input and continuity encoders stay on logical cuda:0 / physical GPU1 with DiT. Postprocessing follows decode. VSR's required device-wide correctness barrier remains intact; output contention, cross-device transfers and peak VRAM are part of the evaluation.

## Decoder trial: rejected

`decode-output/steady.summary.json`: 67.134-second observed window before an explicit restore. Published delivery fell to 19.722 fps with 112 additional underrun events, zero explicit frame drops/restarts, and repeated delivery gaps. Median server chunk increased to 715.217 ms and sampled preview latency to 2985.5 ms. Decoder median was 114.309 ms, p95 190.524 ms, versus baseline 75.493 / 76.363 ms. DiT barely changed (458.279 ms median). Input encode and continuity encode also developed long tails. No allocation retries or OOMs occurred, but output-GPU allocator reserve grew from 18878 to 22892 MiB.

This rejects this specific decoder/postprocess placement with input and continuity encoding retained on the model GPU. It does not prove every VAE partition impossible. The observed contention/cross-device path outweighs the freed model-GPU work; no isolated kernel trace was collected to attribute every stall. Prompt stress was unnecessary after the steady-state regression. The read-only observer ended with expected connection refusal because JoyAI was explicitly restarted to restore the baseline; all preceding rows were retained and summarized separately. This was a manual decision after measured failure, not an automatic recovery or resolution rollback.

DiT remains the largest measured stage. Further work should target its graph/attention path with fixed-input, same-quality comparisons before attempting a higher resolution. The current prompt has 517 tokens and therefore selects a 1024-capacity graph; finer buckets could reduce padded work, but additional graph-private allocations previously exhausted VRAM, so a bounded-memory design needs an offline benchmark first. No prompts were shortened or truncated for this trial.

## Restored and verified

The original performance settings were restored exactly, with all three VAE roles back on the model GPU. `restored/steady.summary.json`: 89.926 seconds, median server chunk 638.678 ms and preview latency 1741 ms, zero new publisher underruns, drops, restarts or delivery gaps. The 41.347-second whitespace prompt-transition check reported 24.065 published fps, 24.016 canvas fps, no errors or dropped frames, and restored the owned prompt. Subsequent scene changes came from live controls; these were preserved. Both 512- and 1024-capacity graphs appeared in the longer observation.

Independent WebRTC receiver: `restored/whep-summary.json`, 119.191 seconds at 23.995 decoded fps, 2592x1440, zero freezes, dropped frames or packet loss. No OOM occurred. Model allocation retries rose from zero to one in the 90-second window and later to five across additional scene changes; this does not qualify unlimited session replacement or indefinite memory stability. Publisher remained at zero lifetime underruns/drops/restarts/delivery gaps in the final check. Timing collection was disabled live and `/debug` confirmed `enabled:false`. No trial controllers or observation processes remain running.

## Interconnect follow-up

Read-only hardware checks: two RTX 5090s, Windows WDDM driver 616.92, Gigabyte B850 AI TOP, both negotiated PCIe 5.0 x8, 32768 MiB BAR1 aperture each. `nvidia-smi topo -m` reports PXB; `topo -p2p r` and `w` report GNS (GPU not supported) in both directions. This describes the installed driver/platform, not a proof of physical impossibility.

[NVIDIA's 5090 specification](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) lists no NVLink. [NVIDIA's developer response](https://forums.developer.nvidia.com/t/p2p-issue-using-two-rtx-5090-gpus/326776/8) says GeForce RTX 50-series GPU P2P is unsupported. The community [aikitoria Linux driver fork](https://github.com/aikitoria/open-gpu-kernel-modules) reports working PCIe BAR1 P2P for RTX 5090 pairs, requiring appropriate BAR mapping and IOMMU configuration. A separate [rtx-p2p project](https://github.com/kacper-daftcode/rtx-p2p) reports undocumented firmware options with stock Linux drivers. These are experimental claims, not tested on this machine; no verified Windows enablement procedure was found. No driver, BIOS, registry, OS or boot changes were made. A separate Linux benchmark would need transfer correctness, bandwidth/latency and full JoyAI delivery validation before assuming a benefit. P2P alone does not automatically merge VRAM or parallelize the model.
