# Stable prompt graphs at 480p

Superseded as the final playback configuration by [robust playback](ROBUST_PLAYBACK_20260918.md): all supported prompt capacities are now prewarmed and retained, and interpolation supplies sustained 24-fps presentation headroom. The native-only measurements below describe the intermediate implementation.

The previous stable-graph path padded all prompts to 1024 tokens and used masked cuDNN SDPA. On the live 840x480 pipeline this produced about 17 fresh fps, despite the previous unpadded Sage configuration producing about 24. A temporary 16-fps target prevented continuous buffer underruns.

The replacement uses SageAttention's variable-length Triton quantization and attention kernels, with single-sequence GPU length metadata and capacity-sized scale buffers. The installed public `sageattn_varlen` wrapper sizes scale allocations from GPU scalars and cannot be captured as-is. This adapter does not modify the installed package. Padding is excluded from key smoothing, quantization, and attention. The prompt mask and sequence-length metadata are updated together when binding the graph. Arbitrary masks without this explicit prefix metadata continue through cuDNN.

Normal prompts share a 512-token graph buffer. Longer prompts round up to 128-token boundaries, up to the existing configured text limit. No prompt is newly truncated. Crossing a capacity bucket still requires capture; the graph cache remains bounded to one entry to fit GPU memory. Changes within a bucket reuse graph and buffer addresses.

`JOYOMNI_PROMPT_ATTENTION=cudnn` at runtime launch disables the prefix adapter for comparison. `JOYOMNI_FIXED_PROMPT_GRAPH=0` restores exact-length unmasked graphs. The live launcher enables SageAttention; prefix attention defaults on only when Sage is enabled. Chunk telemetry identifies `graph_prompt_attention`, capacity and actual token length.

Validation: 31 CPU tests passed across prompt controls, buffer binding, output ownership, presentation, and native transitions. The GPU probe changes input tensors and valid lengths while replaying one capture, poisons padding with values of magnitude 10000, compares against unpadded SDPA, and runs the actual transformer and StreamingGraphRunner. Attention relative L2 error was at most 1.36%; the two-layer transformer comparison against unpadded Sage was below 0.5% BF16 and 0.71% with FP8 image/text projections. This is a numerical regression check, not a full perceptual-quality evaluation.

Deployment `deployment-224156` used the SDK lifecycle, preserved prompt and strength, and resumed operations. Active settings: native 840x480/24, RIFE1, compileVae=true, fixedPromptGraph=true, VAE/model GPU1, resident text/SR/NVENC GPU0. Output sizes remain 720p preview, 1080p NDI, and 1440p GPU video.

First live results under `tulip/evidence/quality-tuning-20260917-141403`:

- `prefix512-steady24.json`: 41.322 warm seconds, 23.668 fresh/model and canvas fps, 23.619 GPU fps, zero drops/errors, 1101 ms median capture-to-canvas latency.
- `prefix-final24.json`: 121.514 warm seconds spanning the prompt test, 23.257 fresh/model fps and 23.265 GPU fps, zero drops/errors, 967 ms median latency, maximum queue four.
- `prefix-final24-prompt.json`: 475 -> 477 -> 475 tokens, both hot revisions applied without graph capture or an intervening scene reset. 23.212 fresh fps, zero drops/errors, original prompt restored, not superseded. Largest retained sender gap 115.99 ms, compared with about 903 ms during the earlier exact-length graph test. These are separate runs rather than a controlled latency distribution comparison.
- `prefix-final24-receiver.jsonl`: 120 samples, 2788 frames decoded at 2592x1440, zero drops, freezes or lost packets. Maximum frame callback gap 117 ms. This validates the local receiver, not an external network.

These results restore approximately the former 24-fps behavior but do not establish a locked 24.000 fresh fps or guarantee no stutter for every prompt. Longer prompts crossing buckets still trigger capture and do more computation.

A final bounded comparison of six launch policies for the Sage Triton kernel found no improvement over its existing eight-warps/four-stages policy. The tested policy was retained; no further runtime restart was needed. See `prefix-kernel-tuning-20260918.json`. Timings were collected with the live workload active and are diagnostic, not isolated GPU benchmark claims.
