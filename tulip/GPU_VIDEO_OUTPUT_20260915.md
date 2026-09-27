# GPU-resident mobile output

Follow-up: mobile bitrate was briefly raised to 20 Mbps after a short quality trial, then restored to 15 Mbps after the user reported jitter; see GPU_VIDEO_QUALITY_20260915.md. The measurements below describe the original deployment.

Deployed September 15, 2026. Mobile output now uses `joyai-gpu`: 2592 x 1440, 32 fps, H.264 NVENC at 15 Mbps. The SDK preview descriptor prefers this stream and orders `hermes-screen` (1944 x 1080) next as fallback. Desktop JPEG remains 1296 x 720. Generation remains 648 x 360, RIFE 2x. Scene handoffs use the existing moving shader bridge on both publishers.

The two RTX 5090s report no CUDA peer access. Native-resolution frames therefore still cross CPU memory between generation and output GPUs. On the output GPU, upscaling, transition blending, RGBA preparation and NVENC consume CUDA tensors. Only compressed packets leave that branch; FFmpeg remuxes them without decoding. The separate NDI/desktop branch still downloads pixels. This is not an entirely GPU-only stack.

Requires `tulip/requirements-gpu-video.txt` (PyNvVideoCodec 2.2.3), installed into the existing service venv with `uv pip install --python .venv/Scripts/python.exe --no-deps PyNvVideoCodec==2.2.3`. Uses the Torch CUDA context and a dedicated output stream. Runtime CUDA device 1 maps to physical GPU 0 with the current CUDA_VISIBLE_DEVICES=1,0 order.

The installed encoder requires uppercase P4, bf=0 and the packed CUDA array wrapper. Its return value is a list of packet dictionaries. It internally buffers three frames even with no B frames; about 94 ms at 32 fps. Do not claim a measured end-to-end latency reduction from encode duration alone. Full-range BT.709/sRGB metadata is added by a bitstream filter, because the encoder's ABGR conversion omitted matching VUI flags. Decoded color bars differed by at most 2/255 after tagging.

Three-minute live audit, with Cue active and prompt/scene exercises:

| Measurement | Result |
|---|---:|
| GPU fresh frames/sec | 32.002 |
| Independent NDI fresh frames/sec | 31.944 |
| GPU / NDI sender drops | 0 / 0 |
| GPU frame preparation/encode/pipe median | 1.80 ms |
| Same stage p99 | 10.78 ms |
| Maximum bounded GPU queue | 24 of 48 |
| Completed GPU transitions | 4 |
| Local WebRTC decoded frames | 5742 |
| Local WebRTC drops / freezes / packet loss | 0 / 0 / 0 |
| Median desktop capture-to-presentation | 1440 ms |

This succeeds where the previous raw 1440p transport delivered 22.62 fps and dropped 341 frames in about a minute. Audits were operational trials with changing live content, not a controlled identical-content comparison. Actual phone decode smoothness and end-to-end latency were not instrumented. Physical GPU 0 had 3776 MiB free in a post-audit sample; GPU 1 had 8736 MiB free. A three-minute check does not establish long-term leak freedom.

Validation: 17 focused CPU contract/geometry/progressive/transition tests and 56 SDK preview/auth tests passed; compile checks passed; real GPU color decoding and live WebRTC audit passed. The exploratory synthetic publisher script initially failed a strict startup-inclusive frame-count assertion and subsequently a recorder startup race; these diagnostic issues were corrected, but that script was not rerun after its final correction. Live audit results above are the production-path validation. Temporary browser audit hooks and test receivers were removed.

Configuration is local `tulip/output-presentation.json`: keep height=1080, direct_ndi=true, progressive=true; gpu_video contains enabled=true, height=1440, bitrate=15000000, local RTMP URL and existing FFmpeg executable. The independent GPU publisher has a bounded queue and latches errors without stopping NDI. Toggle enabled off/on to reset a latched GPU error. SDK fallback ordering is tested, but a native-device outage/recovery drill has not been run.

Rollback: use SDK `update_config` to set preview.stream to `hermes-screen` first, then reconnect the active mobile receiver so it re-reads the descriptor. Disable gpu_video in the local output config after verifying the phone has the fallback. Do not restore the whole saved nanos.json because unrelated settings may have changed. Configuration backup before promotion: sibling SDK `logs/restart-20260915-230954/`. Runtime-deploy backup: `logs/restart-20260915-225959/`.

Evidence: sibling SDK `logs/restart-20260915-230348/gpu-summary.json` and `audit.jsonl`; sibling Etherea `logs/gpu-video-whep-20260915.jsonl`; local `tulip/gpu-color-validation.json` and `tulip/gpu-color-validation-tagged.flv`. SDK promotion script `logs/switch-gpu-preview-20260915.py` performs atomic surgical config update and verifies both preferred and fallback paths are ready.

For the one-time viewer handoff, the existing old-stream session was closed using MediaMTX's documented session-specific POST endpoint: https://raw.githubusercontent.com/bluenviron/mediamtx/v1.14.0/api/openapi.yaml . Neither publisher nor MediaMTX was restarted. The same remote viewer reconnected to joyai-gpu at 23:12:03 EDT; server telemetry confirmed an established peer connection and increasing transmitted bytes. The final health sample had over 18,000 GPU frames sent, zero drops and no error. This verifies the remote route, not device-side decode performance. Handoff evidence is saved in the promotion backup directory as handoff-verified.json.
