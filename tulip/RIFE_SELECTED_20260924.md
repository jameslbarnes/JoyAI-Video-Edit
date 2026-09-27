# RIFE selected after live-show comparison ? 2026-09-24

The user preferred RIFE after the actual hummingbird-show A/B. Selected RIFE
at a chunk boundary with set_interpolation_backend.py. No renderer restart
or creative-control writes. Native 1152x648 / 12 fps / 2x interpolation,
VSR Ultra and 2592x1440 / 24 fps / 15 Mbps GPU NVENC remain unchanged.
The output handoff reports GPU input with zero intermediate downloads.

One additional publisher underrun occurred around the first live RIFE load;
there was no publisher restart or dropped frame. After confirmation, the
publisher added 0 underruns, 0 drops and 0 restarts.
The independent receiver measured 23.999 decoded fps over 54.1
seconds after excluding startup, with 0 decoder drops,
0 reported freezes and 0 lost packets. This is a short local
receiver check, not a TV-panel or long-term stability measurement.

Current selection persists in tulip/interpolation-settings.json. DLSSG remains
available through an explicit `python tulip/set_interpolation_backend.py dlssg`;
restore RIFE with `python tulip/set_interpolation_backend.py rife`. Neither
command adds automatic fallback or changes quality presets. The original
648p favorite snapshots remain untouched.

Evidence: tulip/evidence/rife-selected-20260924-102445
