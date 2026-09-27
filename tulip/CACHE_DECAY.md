# Gradual sink decay — 2026-09-27

The user asked for the picture to "forget things over time" without scene
refreshes, which interrupt the flow. Observed live: a 37-minute-old scene kept
chrome ladder / sparkle structures at the frame edges that were not in the
near-black shader input. The streaming window attends to [first chunk of the
scene (sink), previous chunk, current]; the sink is otherwise kept for the whole
scene, and prompt updates deliberately preserve it.

`deploy/xvideo/serving/cache_decay.py` fades the sink toward the latest committed
chunk every chunk, so the old share halves every `half_life_s` seconds:

- `blend`: sink K/V = lerp(sink, current, alpha) on pre-RoPE keys.
- `dissolve`: an alpha fraction of sink tokens is swapped for the current
  chunk's tokens at the same positions (whole real tokens, no averaged vectors).

alpha = 1 - 0.5^(chunk_s / half_life_s), chunk_s = 8 frames / native fps (0.667 s).
The sink is cloned into owned storage before the first in-place update, so shared
graph publication buffers are never modified; the graph pool is re-seeded
(pool_dirty) after each update. Skips (reported, no change) when either entry is
not pre-RoPE. Default OFF. Unlike the failed `JOYAI_SINK_REFRESH_CHUNKS=32`
experiment (hard swap every 16 s, tested while the positional cap of 128 was the
real corruption cause), this is continuous and has not yet been judged on the
live model.

## Controls (live, no restart once the code is loaded)

    GET  http://127.0.0.1:7873/cache-decay
    POST http://127.0.0.1:7873/cache-decay  {"half_life_s": 120, "mode": "blend"}
    POST http://127.0.0.1:7873/cache-decay  {"half_life_s": 0}     # off

POST changes are not persisted. Startup value: `tulip/cache-decay.json`
(missing file = off). `/debug` -> `session.cache.sinkDecay` shows mode,
half-life, per-chunk alpha, decayed chunk count and any skip reason.

Tests: `python -m unittest tulip.test_cache_decay tulip.test_stream_controls`.

## Live deployment 2026-09-27

JoyAI-only SDK restart at 12:13 (`restart_sink_decay.py 0`; evidence
`evidence/sink-decay-restart-20260927-121351/`): 4K/24 fps, settings preserved,
decay loaded OFF, then enabled live at blend / 120 s.

Measured cost (alternating off/on, 30 chunks each): chunk median 0.569 s off vs
0.604-0.607 s on — about +36 ms per update, from the per-chunk graph-pool
re-seed. Still inside the 0.667 s budget (0 underruns, 0 drops) but it cuts the
headroom from ~98 ms to ~60 ms. At 5 min scene age: clean frame, no edge junk,
no visible blend smear.

**On disk, not yet loaded:** `every_chunks` (default 4) applies the update every
Nth chunk with alpha for N chunks, keeping the same half-life at ~1/4 of the
average cost. Live-settable via POST `{"every_chunks": N}` once loaded; takes
effect at the next JoyAI restart.
