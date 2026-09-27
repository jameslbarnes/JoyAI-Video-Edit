# Explicit return to 648p — September 23, 2026

User requested a return from 1296x720 / 6 native fps / RIFE 4x to 1152x648 /
12 native fps / RIFE 2x. VSR Ultra and 2592x1440 NVENC output at 24 fps remain.
Only JoyAI restarts; current prompt and strength are captured before each change.
No automatic resolution rollback is enabled. Later session-stream, shutdown, and
capture-reserve fixes are retained, and both immutable 648p favorites are untouched.

## Original settings restoration

`native720_vsr_trial.py --restore-648` verifies the favorite's performance-settings
SHA-256 before applying those settings and the 648p native profile. Evidence:
`evidence/native648-vsr-return-20260923-085939`.

Initial observation: 23.996 decoded WebRTC fps over 59.094 seconds, no freezes,
dropped frames, or packet loss. Publisher observation had no gaps, drops, or
underruns; median capture-to-browser-preview latency 1781 ms. These are not
external-TV end-to-end latency measurements.

After that window, a scene restart failed in chunk 2 input VAE encoding with a
406 MiB allocation request. Model GPU reported 28.31 GiB allocated, including
3.94 GiB private graph pools, against the 29.51 GiB allocator limit. The full
five-graph profile therefore does not qualify for robust scene replacement.
The final prompt/strength differ from the original capture because live user
edits occurred, not because the controller restored an older scene.

## Compact graph repair

Explicit command: `native720_vsr_trial.py --restore-648 --compact-prompt-graphs`.
Evidence: `evidence/native648-vsr-return-20260923-090900`.

Keeps prewarmed 512- and 1024-token graphs, removes intermediate 640/768/896
capacities, and preserves supported prompt lengths without truncation or live
graph capture. Short prompts retain the fast 512 path. All VAE work stays with
JoyAI on physical GPU 1; physical GPU 0 retains resident text encoding, RIFE,
VSR Ultra, and NVENC. No quality, cadence, precision, or step-count reduction.

This compact profile is now live. Startup and an explicit same-profile session
replacement both passed. `verify_648_session_replacement.py EVIDENCE_DIRECTORY`
performs that check using the existing runtime and current scene controls.

After replacement, 89.259 seconds of observation measured 11.999 native fps,
23.998 canvas fps, and 23.975 publisher fps, with zero publisher underruns,
drops, delivery gaps, or client errors. Independent WebRTC measured 23.976
decoded fps over 89.129 seconds, zero freezes, dropped frames, or packet loss.
Median capture-to-preview latency was 1738 ms. Active graph capacity during
this observation was 512; this is not a new stress qualification of long prompts.

Model GPU had one allocation retry total (already present after replacement),
output GPU zero; both had zero OOM. Model allocation after replacement was
22.42 GiB, with peak allocation 24.69 GiB. Graph runner count is two.
Prompt/strength changed during live interaction, so the summary's comparisons
against the initial capture are false; newer controls were left intact.
Detailed result: `evidence/native648-vsr-return-20260923-090900/final-summary.json`.

Compared with 720p/6 native/4x, this uses 20% fewer native pixels, twice the native
motion samples, and about 1.2 seconds less measured preview delay. Both GPUs and
VSR Ultra remain in use. Further increases in native resolution at this cadence
need JoyAI/VAE throughput and memory improvements; faster upscaling alone does
not remove those constraints.
