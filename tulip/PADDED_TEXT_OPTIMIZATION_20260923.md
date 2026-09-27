# Padded text rows at 648p

The stable graph family uses 512/1024 text-token capacities. A prompt just over
512 tokens previously performed all text projection/MLP and query-attention
work for 1024 rows. This experiment skips unused rows inside the same captured
graphs. It does not truncate instructions or allocate more graph pools.

## Fidelity constraint

SageAttention quantizes queries in blocks of 128. Removing every padded query
would change the quantization scale shared with the last real query. The new
path preserves the entire final query block, including its padded halo.

At 1152x648, image plus source queries occupy 2592 rows. A 516-token prompt in
the 1024-capacity graph retains 608 text rows. The same projection arithmetic,
FP8 formats, attention precision and weights are used for retained rows.
Inactive outputs are initialized, and query lengths remain GPU scalars.
The video geometry, inference steps, interpolation and upscaler are unchanged.

The feature is opt-in: `trimPaddedText` in `performance-settings.json`, mapped
to `JOYOMNI_TRIM_PADDED_TEXT`. Missing settings explicitly select the original
path. Graph construction selects the implementation; a runtime restart is
needed to change it. `/debug` reports `graph_trim_padded_text` and
`graph_active_text_rows` without reading CUDA scalars back to the CPU.

## Qualification completed before live deployment

- Fourteen CPU regressions and launcher settings/parsing checks passed.
- Three-block small transformer: 14 poisoned-padding cases produced bit-exact
  image and protected-text output while reusing the same graph and addresses.
- Complete streaming graph: 22 cases, two denoise steps, clean-KV store,
  commit, publication, 512/1024 bucket switching, with/without reference KV.
  Latents and all six KV buffers matched bit for bit. Peak allocation 88.5 MiB.
- Actual trained block 0 at 648p geometry: every tested prompt length produced
  bit-exact image and protected-text outputs. Paired cold-cache measurements
  suggested a benefit for 517 tokens but were contended by live output; they
  are not an end-to-end speed claim. Peak allocation 1536 MiB, reserved 2554 MiB.
- Probe processes used physical GPU 0 with bounded memory, stopped only
  themselves if publisher health changed, and left the renderer running.

Separate FP8 launch tuning found only a small isolated MLP-down improvement;
no launch-configuration changes were deployed or combined with this trial.

## Live comparison

Evidence: `evidence/fp8-prefix-20260923/`. The paired live utility temporarily
repeats the current scene's last sentence, uses ownership checks, and restores
only its own prompt. Competing creative updates take precedence. Each phase
runs 65 seconds; summaries discard the first 15 seconds. CUDA event spans are
collected after completion without synchronizing the renderer.

Baseline: 491 tokens in the 512 graph, DiT median 408.47 ms; 516 tokens in the
1024 graph, DiT median 468.26 ms. Server chunk medians 582.23/635.52 ms. Both
settled windows delivered approximately 24 fps without publisher underruns,
drops, restarts or delivery gaps.

The user changed the scene between baseline and restart. The controller
captured and preserved the newer robotic-hummingbird scene; the measurement
refused to apply the old seed. The candidate used a new seed from that scene.
Consequently these are equal-geometry, similar-length observations, not an
identical-prompt A/B. The short-prompt timing acts as a useful control.

| Metric (median) | Baseline | Candidate |
|---|---:|---:|
| Short prompt tokens / capacity | 491 / 512 | 495 / 512 |
| Short DiT | 408.47 ms | 408.99 ms |
| Short server chunk | 582.23 ms | 583.57 ms |
| Longer prompt tokens / capacity | 516 / 1024 | 519 / 1024 |
| Computed text rows, longer prompt | 1024 | 608 |
| Longer DiT | 468.26 ms | 425.90 ms |
| Longer server chunk | 635.52 ms | 600.81 ms |
| Longer preview capture-to-canvas | 1731.5 ms | 1720 ms |

The observed longer-prompt DiT reduction is 42.36 ms (9.0%); the server chunk
reduction is 34.71 ms (5.5%). The 8-frame/12-native-fps budget is 666.67 ms:
median chunk margin increases from 31.15 to 65.86 ms. This is compute headroom,
not a comparable reduction in total preview latency. Both candidate windows
delivered approximately 24 fps without publisher drops, underruns or gaps.
The current 512-capacity scene has essentially unchanged compute cost.

There are still exactly two graphs. Candidate model reserved memory was
28,742 MiB versus 28,614 MiB in the old process (+128 MiB); comparing allocator
totals across a restart is imperfect. Both candidate GPUs had zero allocation
retries and zero OOMs throughout these windows. No extra graph pool was added.
Stage collection was disabled after measurement.

## Final qualification and decision

Keep `trimPaddedText: true` enabled at the existing 1152x648 / 12 native fps,
RIFE 2, VSR Ultra configuration. The main GPU publisher sends 2592x1440 at
24 fps. Prompt and strength edits made during testing were preserved.

- Five-minute local WebRTC receiver: 299.501 measured seconds, 7,188 decoded
  frames, **23.99992 fps**, zero freezes, frame drops or packet loss.
- The 248.982-second prompt-stress observation exercised 512/1024 graphs and
  token counts through 1024. Publisher 23.98969 fps, zero drops/underruns/gaps
  or errors. User scene changes superseded the scripted sequence; it stopped
  its prompt mutations and preserved them. This is not a clean completion of
  the entire scripted stress sequence. Three natural session changes were
  observed in the samples (epochs 1 through 4).
- During those scene changes, canvas counters averaged 23.22 fps and native
  displayed counters 11.62 fps; distinguish these from the continuous 24 fps
  WebRTC publisher. Median preview latency was 1736 ms.
- A final same-prompt scene-start check completed without supersession: prompt
  preparation 527 ms, first generated output 1622 ms, first presentation
  2884 ms, fully revealed 3059 ms. Maximum measured presentation gap 68.3 ms.
  This check was included in the continuous WebRTC observation.
- After subsequent scene starts, the model allocator had six retries but zero
  OOMs; output allocator zero retries/OOMs. There were still only two graphs,
  model reserve 28,626 MiB and output reserve 20,780 MiB. Scene starts retain
  allocation pressure, so this change does not resolve every source of delay.
- Final live health: 648p / RIFE 2 / VSR Ultra, feature active, profiling off.
  No driver, OS, quality preset, inference-step or FP8 launch changes were made.

The finite local receiver test does not establish long-duration or remote-TV
reliability. The gain is less wasted longer-prompt compute and more delivery
margin; total preview latency remains roughly 1.7 seconds.

## Reproduction and explicit reversion

Run from this repository with the existing `.venv/Scripts/python.exe`:

```powershell
$env:PYTHONPATH='deploy'
.venv/Scripts/python.exe -m unittest tulip.test_fp8_prefix_rows tulip.test_fixed_prompt_graph tulip.test_compact_prompt_graph tulip.test_no_automatic_rollback
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tulip/test-runtime-settings.ps1
.venv/Scripts/python.exe tulip/verify_fp8_prefix_streaming.py NEW-PROOF.json
node tulip/measure_padded_prompt.cjs NEW-MEASUREMENT.json PROMPT-SEED.json
.venv/Scripts/python.exe tulip/summarize_padded_prompt.py NEW-MEASUREMENT.json
```

The live measurement requires `stage-profile.json` to contain
`{"enabled":true}`; disable collection after measuring. Never overwrite old
evidence. A second measurement reuses the seed only if the scene still matches.

To explicitly enable/disable this experiment at the existing 648p profile:

```powershell
.venv/Scripts/python.exe tulip/deploy_fp8_prefix.py NEW-DEPLOY-DIRECTORY --enabled yes
.venv/Scripts/python.exe tulip/deploy_fp8_prefix.py NEW-RESTORE-DIRECTORY --enabled no
```

The controller preserves the latest prompt/strength, changes only this
performance flag, restarts only the existing JoyAI runtime through the SDK,
and leaves the process/profile intact if observation times out. There is no
automatic rollback. Pre-change source and settings are saved under `before/`
and `integration-before/` in the evidence directory. The two named 648p
favorites remain untouched.
