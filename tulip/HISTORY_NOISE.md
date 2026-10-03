# History noise: the cause and fix of long-run "mosaic" junk — 2026-10-02

## Symptom

Live scenes degrade over minutes: subjects turn into a glassy, multicolored
mosaic (chest/fur first), edges and backgrounds fill with speckle and "chrome
ladder" structures. Scene resets clear it; sink decay (`CACHE_DECAY.md`) did not.

## Method (`tulip/drift_lab/`)

- `metrics.py`: reference-free per-cell scores against the input frame —
  junk (output detail the input doesn't explain), staleness, texture imprint.
- `live_probe.py`: read-only live recorder of metrics + runtime state.
- `replay.py`: deterministic offline replay, production-faithful
  (1152x648, 12 fps, 2 steps, FP8/Sage, temporal limit 8, combined strength,
  text encoder resident on the second GPU), looping a recorded FLUX-IN clip
  (`drift_lab/inputs/flux-in-20261002-2119.mp4`), one arm per hypothesis.
- `analyze.py`: per-arm trends and contact sheets.

Evidence: `tulip/evidence/drift-offline-20261002-2130/` (batch1..5).

## Findings

Reproduced offline in about 2 minutes (fixed prompt, seed, input), with the
input a featureless gray gradient, so the junk is self-generated, not shader
texture. Junk score at 0 -> 4 min:

| Arm | Memory window | Junk | Notes |
|---|---|---|---|
| baseline / baseline_eager | sink + previous | 0.09 -> 0.22 | dog dissolves by ~2 min |
| clamp_feedback | (carry clamped) | 0.09 -> 0.20 | not the decoder carry |
| steps4 | 4 denoise steps | 0.11 -> 0.25 | not weak denoising |
| no_kv | none | 0.045 flat | clean but pops every chunk (6.2x boundary jump) |
| sink_only | sink | 0.039 flat | clean, pops (4.8x) |
| no_sink | previous | 0.17 by 1.3 min | fastest drift |

The previous-chunk attention memory carries and amplifies the junk: each chunk
is denoised in 2 steps while attending to KV computed from the previous output
as if it were exact (timestep 0), so invented detail is copied forward.

## Fix

Store the history KV from (1 - sigma) * clean + sigma * noise at timestep
sigma * 1000 (flow-matching convention), keeping the scene-start sink clean,
with sink decay off.

| Variant | Junk 0 -> 3 min | Brightness 0 -> 3 min |
|---|---|---|
| sigma .10, noisy sink | +0.10/10 min | 0.47 -> 0.49 |
| sigma .20, noisy sink | flat (15 min: 0.046 -> 0.025) | 0.49 -> 0.77 (washes out to white) |
| **sigma .20, clean sink, decay off** | 0.044 -> 0.052 | 0.50 -> 0.59, keeps color |
| sigma .20, clean sink, decay 60 s | 0.048 flat | 0.50 -> 0.68 |

Continuity (chunk-boundary jump vs typical frame): baseline 2.6x, sigma .10
2.2x, sigma .20 2.9x (noisy sink). No measurable speed cost.

### Fine-detail mode (2026-10-03, now the default)

Full-latent noise at 0.20 made motion visibly less continuous (James noticed).
Noising only the fine detail (latent minus a 5x5 blur, noise energy-matched to
it) keeps broad shapes and motion intact (`scorecard.py`, dog scene):

| Variant | Junk @3 min | Boundary jump | Typical frame change |
|---|---|---|---|
| baseline (no fix) | 0.228 | 2.77 | 1.09 |
| full 0.15 | 0.059 | 2.78 | 1.29 |
| full 0.20 | 0.052 | 4.42 | 1.61 |
| **fine 0.20** | **0.047** | **1.84** | **0.81** |
| fine 0.30 | 0.037 | 2.92 | 1.72 |
| fine 0.30, timestep 0 | 0.074 | 2.75 | 1.32 |

Fine 0.20 is both cleaner and smoother than the original pipeline; the
timestep must stay at sigma * 1000 (telling the model the mix is clean lets
junk back in).

## Production

`deploy/xvideo/serving/history_noise.py`; eager path in `_denoise_chunk`, graph
path via `in_store_noise` / `in_store_sigma` / `in_store_timestep` buffers read
by the captured graph (sigma changes at runtime, no recapture; sigma 0 is
bit-identical). Chunk 0 (sink) is always stored clean.

    GET  http://127.0.0.1:7873/history-noise
    POST http://127.0.0.1:7873/history-noise  {"sigma": 0.2, "mode": "fine"}
    POST http://127.0.0.1:7873/history-noise  {"sigma": 0}      # off

Startup value: `tulip/history-noise.json`. With history noise on, set
`tulip/cache-decay.json` half_life_s to 0. Tests: `tulip/test_history_noise.py`.
