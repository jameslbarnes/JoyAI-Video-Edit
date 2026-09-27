# Live GPU DLSSG trial

The user explicitly requested switching DLSSG on live for visual assessment
and confirmed that a playback stall during the switch is acceptable.

The new interpolation selector supports RIFE and DLSSG at chunk boundaries.
`tulip/interpolation-settings.json` selects `{"backend":"dlssg"}` or
`{"backend":"rife"}`. A missing file defaults to RIFE. There is no automatic
fallback or quality rollback. The existing two 648p favorites are untouched.

The active quality profile remains 1152x648 / 12 native fps, 2x interpolation,
VSR Ultra, 2592x1440 / 24 fps NVENC. The engine selection is the visual change.
DiT padded-text optimization, compiled VAE placement and preview buffers are
preserved. The deployment captures the latest prompt/strength before stopping
only JoyAI; the existing SDK launches the same service and restores controls
only while the initial scene is still owned.

## GPU path

Native JoyAI RGB frames currently reach presentation as CPU arrays. DLSSG
uploads those inputs, then exports CUDA surfaces through DLPack. Generated
and original outputs are copied into independently owned Torch GPU buffers,
then passed directly to VSR and the existing NVENC publisher. The ownership
copy is GPU-to-GPU; there is no intermediate raw-frame CPU download/re-upload.
The CPU NDI/JPEG preview branches still download their final resized pixels.
This is a GPU-only interpolation/output chain after the input upload, not a
claim that the entire application avoids CPU frames.

The bounded native surface pool cannot own the live lookahead/output queues.
Copying on the GPU and waiting for that copy allows native surfaces to be
released promptly while queued frames remain valid through further inputs,
scene resets and native-session closure.

The bridge uses the CUDA primary context already owned by Torch for this
RGB/Torch path. The probe verified production `CUDA_VISIBLE_DEVICES=1,0`,
logical `cuda:1` / physical GPU 0, and installed PyAV 13.1.0. No Python package
upgrade or global dependency change was needed. This integration does not
construct DLSSG FFmpeg AVFrames; the older FFmpeg hardware-context creation
path is not used. Bridge 1.3.0, ABI 3, runtime 310.9, NVOF enabled.

DLSSG's history resets at scene boundaries. When the native engine reports no
intermediate frame at a cut/disabled interval, cadence holds the preceding
frame and reports `cadence_holds`; it never silently runs RIFE. First input
after a reset produces one original. Standard eight-input chunks produce
16 outputs. The existing live shader bridge handles scene transitions;
DLSSG does not advertise the legacy RIFE scene-morph capability.

The live adapter requires the GPU VSR progressive path. It deliberately rejects
incompatible CPU-output/enhancement-order configurations instead of silently
introducing an intermediate CPU transfer. Native 2x/4x multipliers are accepted;
this deployment and real-GPU qualification use 2x.

## Verification before deployment

- Five-frame bridge/context probe: CUDA interop and NVOF active, all original
  frames exact, generated output on `cuda:1`, native raw download bytes zero.
- Production adapter: first frame plus three eight-input chunks, 49 retained
  GPU outputs (more than native pool capacity), exact native originals,
  GPU-only DLSSG-to-VSR with `Tensor.cpu()` prohibited during those calls.
- Retained output remained exact through a scene reset and native-session
  close. New history began with one original frame.
- 24 existing presentation, output, prompt-control and no-automatic-rollback
  checks passed; five new explicit selection/reset/error/close checks passed.
- All predeployment probes ran separately while the old RIFE renderer stayed
  healthy. The change needs one restart to load the new selector; subsequent
  engine selections do not restart JoyAI or change prompts.

## Inspect and manually select

`/debug.session.last_chunk_profile` exposes `interpolation_backend`,
`interpolation_device`, `interpolation_gpu_output`, `interpolation_s`,
`dlssg` diagnostics and `output_order=dlssg_sr_nr`. Existing client cadence
fields retain their `rife_multiplier` name for protocol compatibility.
`rife_gpu_handoff.input_download_bytes=0` remains the existing generic output
handoff diagnostic. Native DLSSG diagnostics independently report raw readbacks.

From this repository, select either engine explicitly:

```powershell
# Revert to the preserved RIFE implementation on the next chunk.
.venv/Scripts/python.exe tulip/set_interpolation_backend.py rife
# Select DLSSG again.
.venv/Scripts/python.exe tulip/set_interpolation_backend.py dlssg
```

Initial engine construction can pause output. Both implementations remain
available; successfully constructed engines are retained within the WebSocket
session for later switching. Closing the session drains output and closes
the native engine. Native engine failures are surfaced rather than triggering
an automatic rollback.

Evidence and source/settings snapshots: `evidence/dlssg-live-20260924/`.

## Live acceptance

DLSSG is selected and producing the live intermediate frames. After the one
planned restart, `/debug` reported `interpolation_backend=dlssg`,
`output_order=dlssg_sr_nr`, logical `cuda:1`, CUDA interop and NVOF active.
Steady chunks reported eight originals plus eight actual DLSSG midpoints,
zero cadence holds, zero native download bytes and zero intermediate handoff
download bytes. GPU publisher geometry is 2592x1440 at 24 fps; browser preview
is 1296x720. VSR Ultra and the 648p/12-native-fps profile remain active.

The brief read-only smoke observation had no preview/publisher drops,
underruns, delivery gaps or errors; median capture-to-preview latency was
1770 ms. Its settled 9.565-second publisher-counter window is too short for
a precise sustained-fps claim. This trial intentionally prioritizes the
user's live visual assessment; it is not a long-duration or independent
WebRTC receiver qualification. Both GPU allocators reported zero retries/OOMs
in the captured live snapshot. The operation hold was released.

The settings file remains on DLSSG for the user to evaluate. Nothing schedules
an automatic return to RIFE. Manual selection commands above take effect on the
next interpolation chunk; first construction of an engine can briefly stall.
