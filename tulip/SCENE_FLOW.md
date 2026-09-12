# Output-only scene morph

The Tulip service now advertises `scene_morph_supported` when RIFE2x is active.
A validated prepared-scene token arms one RIFE entrance on a fresh session:
first shader input -> first newly generated frame,15 recursive midpoints plus
the original endpoint. Uses the existing FluxRT FP16 network/checkpoint; no new
dependencies or model files. SR processes the resulting16-frame first chunk.
All morph pixels are output-only; model KV/VAE state is still reset.

The Etherea browser continuously dissolves through its live shader during
preparation/reset, then composites the neural entrance into normal playback.
See `etherea-ai/docs/JOYAI_SCENE_FLOW.md` for exact live evidence, launch/test
commands, bounds, visual findings and rollback. Service logs for this deployment
are `tulip/scene-flow-service-20260912.log` and `.err.log`.

Normal prompt updates, plain resets/reconnections and RIFE toggles do not arm a
cross-scene morph. `rife.reset()` cancels any pending entrance; only an explicit
prepared scene with active RIFE enables it. Native counters exclude midpoints;
metadata marks them `presentation=scene-morph` and `interpolated=true`.
