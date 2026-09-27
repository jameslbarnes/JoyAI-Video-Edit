# Prompt update hitch investigation, September 15

Update: the later fixed-capacity masked graph change is deployed and verified. See [FIXED_PROMPT_GRAPH.md](FIXED_PROMPT_GRAPH.md) for the current implementation and live results. The unresolved status below describes the earlier attempts.

The earlier progressive-slowdown fix remained active: graph_path=1 before and after ordinary prompt changes, with the BF16 text encoder fully resident on physical GPU 0. This was not a reversion to eager execution or CPU encoding.

A controlled 40-second observation changed one internal space in the current prompt to a newline, then restored it. Scene epoch stayed constant. Desktop playback exhausted its queue on both updates; the direct bridge also had one 200 ms sample with no fresh frames. Zero queue discards did not imply continuous output.

The original drain waits for model postprocessing and pseudo encoding, but does not wait for completed chunks to finish RIFE, upscaling, and delivery. A tested follow-up added a per-session presentation completion count and waited for it before capture. That ordering change alone was insufficient: capture still drained playback reserves (five empty 200 ms samples across two updates).

A second experiment skipped the full-model warmup when replacing an already-ready graph, with a warmup retry. The actual CUDA backend invalidated capture and the retry did not recover; the session stopped producing frames. This optimization is not usable as implemented. Both experiments were reverted, including their tests. The original working graph, session, and server implementations were restored byte-for-byte from backups. Recovery uses the SDK runtime-only restart with explicit shader-input verification.

The short prompt-change hitch remains unresolved. A safe future change must retain per-shape warmup and remove or hide its interruption without exhausting playback reserves. Do not remove warmup solely because another text shape was previously captured; do not assume a failed capture can be retried in the same CUDA context.

Cue current-prompt context was enabled in the live profile and persisted as CUE_SHOW_CURRENT_PROMPT_IN_FULL=true in cue/.env. No fresh Cue decision trace was available at that check, so subsequent LLM request injection still needs observation.

Evidence: easel-agent-sdk/logs/prompt-update-before-drain-20260915.json, prompt-update-after-drain-20260915.json, prompt-update-hot-capture-20260915.json. Pre-change source copies: logs/restart-20260915-175450. Recovery deployment: logs/restart-20260915-181237.

Recovery verified: exact pre-test prompt restored; ordinary prompt revision applied without error; graph_path=1; fresh independent Easel shader input; RIFE 2x and reference disabled. A 40-second recovery observation (34-second measured interval) delivered 31.98 fresh NDI frames/s at 1944x1080, 32.16 desktop frames/s, zero output drops, and zero NDI errors. Final desktop latency was 1390 ms. Evidence: easel-agent-sdk/logs/prompt-jitter-recovery-20260915.json. This confirms restored normal playback, not elimination of the prompt-change hitch.
