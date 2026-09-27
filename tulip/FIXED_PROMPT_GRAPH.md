# Fixed-capacity prompt graphs

Current implementation (2026-09-18): see [robust playback](ROBUST_PLAYBACK_20260918.md). All five prompt capacities (512, 640, 768, 896, 1024) are captured before playback, using graph-safe Sage prefix attention and independent graph pools with explicitly shared KV buffers. Live prompt changes select a resident graph without recapture. The current profile generates 12 native fps at 840x480 and interpolates 2x for 24 output fps. The 1024-token/cuDNN design and measurements below describe the original implementation.

Prompt-length changes previously changed the CUDA graph key and forced a live warmup/capture while input submission and GPU presentation were paused. The earlier progressive-slowdown correction ensured graph execution resumed, but did not eliminate this transient production gap.

The graph key now uses the configured prompt capacity (1024 by default). The runner allocates one prompt buffer and one boolean key mask at startup. Binding a prompt validates its dimensions and capacity before mutation, copies valid embeddings, clears the unused tail, and updates the mask in place. Image and cached KV keys remain valid; unused prompt keys contribute nothing. The text encoder retains its previous sequence-length limit and GPU placement.

Only masked attention calls use cuDNN SDPA. The installed SM120 SageAttention dispatcher does not honor an arbitrary attention mask, so it must never receive padded prompts through that unmasked path. Unmasked eager and store passes retain their original backend. Startup still performs the complete warmup and capture. The one-entry graph-cache bound is retained.

`JOYOMNI_FIXED_PROMPT_GRAPH=0` restores the original variable-length graph path on the next runtime start. The default is enabled. Last-chunk telemetry reports graph_prompt_capacity, graph_prompt_tokens, and graph_prompt_masked.

Validation before live deployment:
- 30 CPU stream-control, strength, direct-output, and fixed-prompt tests passed.
- Padding poison tests verify masked keys/values cannot affect valid output; longer-to-shorter updates clear stale data and preserve buffer addresses; overflow fails before changing the active prompt.
- The actual transformer in a small two-layer GPU fixture produced identical padded and unpadded outputs for the tested lengths, including FP8 and capacity 1024. The production StreamingGraphRunner replayed one capture with multiple lengths and stable buffer addresses.
- Isolated GPU attention timings at representative padded dimensions: cuDNN masked 0.49 ms/call, Sage unmasked 0.26 ms/call. These are kernel measurements, not whole-pipeline throughput.

Full-model live validation passed after SDK deployment:
- 110 seconds / six whitespace updates: 31.995 fresh NDI fps, 31.986 desktop fps, zero output drops, no empty-buffer samples, 42 ms worst desktop frame interval (35.5 ms p99). Those whitespace variants mapped to two token lengths, 451 and 452; the test's three-length coverage assertion prompted the wider follow-up below.
- 48 seconds / three wider updates, including repeating the same scene instruction: token lengths 451, 452, and 543; 31.976 fresh NDI and desktop fps; zero output drops or empty-buffer samples; 38 ms worst desktop frame interval. All nine revisions applied, the scene epoch stayed unchanged, every sampled chunk used the fixed masked graph, and no graph captures occurred during updates.
- The 80-second local WebRTC receive decoded 2,559 frames at 1944x1080 / 32 fps with zero frames dropped, freezes, or packet loss; max RTP jitter 4 ms. This verifies the local receiver, not an unobserved phone/network.
- The original scene prompt was restored, independent Easel NDI input remained fresh, and temporal KV entries remained bounded at two. Final playback latency was 1,433 ms with the existing buffer configuration.
- Full-model DiT time was approximately 219 ms/chunk versus about 140 ms in the earlier unpadded Sage path. Total model chunks remained around 330?370 ms, within the 500 ms input budget. The fix uses more GPU computation while maintaining the target output rate.
- RIFE 2x, native generation geometry, SR/output resolution, reference-off, and resident BF16 text settings were retained. A live output frame is saved as fixed-prompt-live-20260915.png.

Live evidence: easel-agent-sdk/logs/restart-20260915-192222/fixed-prompt-observation.json; logs/restart-20260915-192446/fixed-prompt-observation.json; etherea-ai/logs/fixed-prompt-whep-20260915.jsonl. The wider test completed successfully. The main verification script now includes a repeated instruction to ensure wider token-length coverage on future runs.

Source backups: tulip/fixed-prompt-backup-20260915-191126. SDK deployment snapshots: easel-agent-sdk/logs/restart-20260915-191827. GPU proof scripts and JSON artifacts are alongside this document.
