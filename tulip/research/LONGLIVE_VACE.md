# LongLive, VACE, and expressive realtime shader transformation

## Etherea-specific correction and additional evidence (September 10)

The initial analysis below examines upstream Scope. Subsequent inspection of
`C:\Users\Admin\src\etherea-ai` at HEAD
`399072a29f8821b289c12c83759e586fc1960c1e` found a materially different historical
deployment. Existing user changes were preserved; Etherea was only read.
**Do not assume current upstream Scope reproduces the old Etherea B300 setup.**

### Recorded quality recipe

Etherea's `CLAUDE.md`, under "Current Pod", records a B300 running the custom
`generalsemantics/scope-etherea:20260319-c42ac0a` image, LongLive at 832x480,
LightTAE, `[1000, 875, 750]`, and flat-color LoRA scale 0.8. This is a historical
note, not a live-session measurement. Other launch paths set two steps; neither
default establishes the settings of the user's favorite session.

The checked-in custom pipeline manager explicitly constructs LongLive in BF16,
defaults generator quantization to None, loads its required performance LoRA,
and separately enables the optional VACE module if its file exists. Its status
can still retain `vace_enabled=true` after warning that the weights were missing,
so that boolean alone is not proof of loaded conditioning weights.
See [pipeline manager](../../../etherea-ai/docker/scope/patches/pipeline_manager.py).

Additional artistic adapters are concrete, not hypothetical:
[lora_stacks.json](../../../etherea-ai/src/backend/configs/lora_stacks.json)
defines LANA 1.0 and LANA 1.0 + flat-color 0.9. The LANA prompt version deliberately
matches painting-caption vocabulary. The legacy backend prompt sender prepends
loaded-adapter trigger words using `lora_registry.py`; the newer Cue/browser path
can bypass that sender, so automatic trigger insertion must not be assumed for
every version or transport. These style choices are independent of VACE.

### Temporal memory was customized, not merely preserved

Commit `86b76a6d` introduced an attention-score bias experiment. The subsequent
commit `8800769d` explicitly replaced it: its message reports performance falling
from 27 to 1.4 FPS with repeated FlexAttention compilation, versus 27.4 to 27.3
FPS using sink refresh at decay 0.5. These are developer-recorded historical
results without a complete benchmark manifest, not new Tulip measurements.

The selected [Dockerfile](../../../etherea-ai/docker/scope/Dockerfile) installs
[sink_refresh_patch.py](../../../etherea-ai/docker/scope/patches/sink_refresh_patch.py),
not the older attention-bias patch. The replacement blends recent cached K/V into
initial sink positions after ordinary cache updates:

`new_sink = (1 - sink_decay) * old_sink + sink_decay * recent_context`

The UI's Memory control sends `sink_decay = 1 - slider/100`. This supplies a
mechanism for an evolving scene anchor: neither permanent fixation on the first
scene nor repeated hard resets. Its artistic benefit is plausible, not proven
by the throughput figures. The patch is inference-time cache modification;
its quality and long-run stability must be tested rather than assumed.

There is also a wiring caveat: the HTML slider defaults to 50, but the module's
initial payload still contains `kv_cache_attention_bias: 0.5`, not
`sink_decay: 0.5`. The inspected path sends sink decay when the Memory handler
runs; a displayed default is not evidence the server received that value.

### VACE's role needs more careful attribution

The historical [frame processor](../../../etherea-ai/docker/scope/patches/frame_processor.py)
passes incoming shader frames as `call_params["video"]` and prepares with
`video=True`. It does not itself route them to `vace_input_frames`, unlike the
modern upstream graph path discussed below. This establishes the caller's
interface, not all internal behavior of the historical base Docker image, whose
complete pipeline code is not vendored here. Consequently, **VACE enabled does
not establish that the old shader feed used VACE structural conditioning**.

Etherea does explicitly support VACE reference-image guidance and resends its
reference path with prompt updates. Commit `72ab200a` documents the February 8
change connecting the legacy abstraction slider to VACE reference strength.
Today the repo contains overlapping control paths:

- `AudioUI.updateDenoisingA` -> `ScopeUI.setScopeNoiseScale` actually sends
  `vace_context_scale = slider/49 * 2`, despite the function's name.
- `ScopeUI.updateScopeDenoising` sends conventional `noise_scale` and disables
  the motion-aware noise controller.
- The newer multistream `applyScopeAbstraction` likewise sends `noise_scale`.

Therefore my initial shorthand that the main abstraction control simply sends
noise scale was incomplete. The active UI/version matters. The DA3 sidecar is
also **post-generation output depth**, not evidence of depth conditioning input:
its documented path is Scope RGB output -> depth estimation -> RGB/depth stream.

### Smooth display and smooth generation must remain separate

Historical launch code requests RIFE 8x, but the vendored frame processor does
not itself establish that those requests reach a working interpolator in every
image. It does explicitly repeat the last RTMP frame during queue gaps. Its input
collector samples across the accumulated buffer and discards frames through the
last sampled index. These behaviors affect perceived cadence and motion sampling,
and must not be counted as additional native generated FPS.

The historical processor also consumes one parameter update per chunk, unlike
the current upstream drain-all implementation. The modern coalescing behavior
described below cannot be credited retroactively to this older deployment.

### Revised experiment priority

First reconstruct **Etherea's actual artistic/temporal recipe**: its matching
image, BF16 generator, LightTAE, 832x480, exact denoising schedule, artistic LoRA
and trigger/prompt version, shader routing, and confirmed sink-decay setting.
Then ablate sink decay, style adapter, and VACE independently. Full WanVAE remains
a useful quality comparison, but LightTAE is now the historically grounded
baseline rather than an assumed downgrade from the successful B300 run.

Current Etherea deployment defaults to upstream Scope with custom-image and
runtime-patch switches off unless overridden (`scope_deploy.py`). Merely selecting
LongLive today does not guarantee the custom memory behavior is present. The
user's exact session image/configuration remains the smallest missing evidence.

## Findings

VACE is a strong candidate for the distinctive controllability of a Scope-based
LongLive video-to-video workflow. The strongest explanation is a combination,
however: **a generative video prior, a separate structural-control pathway,
persistent generated-history memory, and deliberate prompt-transition behavior**.
No saved configuration or recording of the specific B300 session was available
for this analysis, so attributing its subjective quality to any single component
would be premature.

The architectural distinction matters more than the V2V label. In Scope's usual
VACE route, incoming frames supply learned conditioning hints; they are not simply
the image being partially erased with noise. This can preserve a moving shape
while permitting a substantially different material, subject, or environment.
That is a plausible match for expressive shader-driven generation. It does not
guarantee that every shader or prompt will work well.[^1][^2][^3]

LongLive need not be treated as intrinsically B300-only. Scope identifies its
LongLive implementation as a Wan2.1 **1.3B** derivative, not a large 14B/16B model,
and estimates approximately 20 GB VRAM. This is an estimate, not a measurement on
Tulip and not a promise of B300-equivalent resolution or speed.[^4]

## Scope version examined

The installed checkout at `C:\Users\Admin\src\scope` is clean and is pinned to
`44ccdd5c24d7249f5c79dbd9648d5b046970805d`, dated July 22, 2026. Current upstream
HEAD was `2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0` when checked September 10, 2026.
Sixteen selected LongLive/VACE implementation and documentation files were
downloaded at that exact revision and compared with the installed copies; all
sixteen were identical after newline normalization. The snapshot and SHA256
manifest are saved alongside this note. Scope itself was not updated or modified.

Additional implementation observations below cite the installed commit where
that file was not part of the sixteen-file comparison. Neither checkout proves
which version or preset was used on the B300.

## Two substantially different V2V paths

| Property | Standard latent-initialization V2V | Scope VACE control-video route |
|---|---|---|
| Input port | `video` | `vace_input_frames` |
| Initial generation latents | Encoded input mixed with noise | Fresh noise, unless a separate `video` input is also connected |
| Main control | Noise scale / denoising schedule | Learned VACE hint injection and context scale |
| Structural input | Entangled with source appearance | May be RGB, contours, depth, grayscale, or other supported control representation |
| Appearance freedom | Increasing change also removes source information | Generator can create appearance while receiving separate structural guidance |
| Temporal continuity | Depends on the running video model and caches | Also depends on the running video model and caches; VACE is not a replacement for memory |

The server's graph builder explicitly routes frames to `vace_input_frames` for
selected VACE-enabled pipelines. The pipeline's routing block chooses the
source-latent initialization only when `video` is present. Otherwise it uses the
fresh-noise preparation block. An explicit custom graph can connect both paths,
so the actual port wiring must be recorded, not inferred from a UI label.[^1][^2]

In the conventional path, Scope computes a linear mixture of encoded source and
noise: `noise * noise_scale + source_latents * (1 - noise_scale)`. Raising noise
loosens adherence but also discards source information in the starting latent.
This is a different control from changing the amplitude of learned structural
hints.[^5]

The implementation explicitly clears ordinary noise controls for the VACE-only
route, which internally resolves to the fresh-noise mode. There is an exception
for explicitly modulated noise parameters. Thus the presence of a noise slider
or a stale noise value in a session payload does not establish that conventional
source/noise mixing was active. Ordinary motion-aware noise control is associated
with the `video` preprocessing path, not automatically the VACE-only path.[^6]

## What VACE actually contributes

### A learned influence inside the generator

Scope constructs parallel VACE context blocks and injects their outputs as
residual additions into selected base-transformer blocks. The implementation
uses a context multiplier and defaults to injection at every other block.
The first context block also receives current generation features; this is not
a static overlay applied to decoded images.[^3][^7]

For a shader, the useful possibility is: a moving contour can influence where a
tentacle bends, while the base model supplies texture, lighting, and semantics.
This is an interpretation of the architecture, not evidence that VACE perfectly
preserves geometry or optical flow. Stronger hints can overconstrain the output;
weaker hints can let it drift away. The balance is an artistic control, not a
universal quality slider.

### Generate everywhere does not mean ignore the input

When a control video is provided without a mask, Scope supplies an all-white
mask. White means a region may be generated; black marks a preserved region.
The source still enters the conditioning representation even when the entire
frame is marked for generation. Treating an all-white mask as “conditioning off”
would therefore be incorrect.[^8]

The encoder constructs preserved and reactive streams and combines their latent
channels with an encoded mask. The standard conditioning representation has
96 channels: two 16-channel latent streams plus 64 mask channels. With an
all-white mask, the reactive stream carries the control content. This is not a
pixel-copy blend.[^9]

### Persistent state requires more than one cache

Scope allocates distinct inactive/reactive encoder caches where the chosen
autoencoder needs them. Its encoding utility uses different cache behavior for
all-white conditioning and mixed-mask editing, avoiding temporal contamination
between conceptually different streams. These details are especially relevant
to TAE's stateful encoding. Resetting or accidentally sharing those states can
change the output even when the prompt and model weights are unchanged.[^9]

An important implementation caveat: “cached reference” is not synonymous with
“all VACE compute cached.” The inspected denoising loop passes VACE context into
each generator call. Dynamic control-video context is encoded for each chunk,
and the context blocks execute when that context is supplied. Their attention
blocks do not maintain the base model's historical KV cache. The running video
generator provides the main long-range history.[^7][^10]

## What LongLive contributes independently of VACE

### A model trained to continue its own generated video

LongLive's original training combines few-step causal generation with long
rollouts and prompt-switch examples. The key distinction from a frame-by-frame
image editor is that generated history is part of the generation problem. The
training explicitly addresses the mismatch between short training clips and long
inference streams.[^11]

The likely artistic consequence is persistence: an invented form can remain an
object with an evolving trajectory instead of being independently rediscovered
in every frame. A running KV cache alone does not prove this quality—training and
the learned video prior matter too. The old B300 session is needed to establish
how strongly this quality was expressed in that particular workflow.

### Local motion history plus an initial anchor

Scope's LongLive configuration uses three latent frames per block, temporal
compression by four, a local attention budget of twelve latent frames, and a
three-latent-frame sink. The attention implementation reserves part of the
budget for initial sink tokens and rolls the remaining local history. The
nominal output block is therefore twelve video frames; these are not twelve
independent image generations.[^12][^13]

The initial anchor can preserve scene identity while recent history carries
current motion. This also creates inertia: retaining an early visual state may
resist a drastic new direction. Good live control involves choosing when to
preserve that anchor and when a deliberate cut is preferable.

The VAE also decodes with temporal caching. Visual continuity therefore depends
on both generator state and the pixel decoder, not just on retaining a prompt
string between calls.[^14]

### Prompt changes are a state-management operation

Scope's LongLive recache block keeps a sliding buffer of generated latents. When
conditioning changes, it can recompute history under the new conditioning;
with the global sink enabled, the sink is preserved. This is more sophisticated
than replacing text while leaving an old semantic cache untouched.[^15]

However, the installed cache-setup block can reset the cache when conditioning
changes in V2V outside a transition and automatic cache management is enabled.
The smooth-transition path signals its state to avoid that hard reset. Therefore
“LongLive supports smooth prompt changes” must not be translated into “every API
prompt update in every Scope mode is seamless.”[^16]

Scope also supports interpolation between prompt embeddings over successive
generation calls. Those transition steps are chunk-level updates, not display
frames. A slow morph, a direct prompt replacement, and an explicit cache reset
are different actions with different visual and timing consequences.[^17]

Control delivery matters too. The installed processor drains queued parameter
updates before a generation call, merging them so the latest value for each key
wins. Scalar input ports similarly take their latest value. This prevents a burst
of prompt edits from being replayed one stale edit per chunk. It does not remove
chunk-boundary delay or establish a latest-frame policy for video: video inputs
use a separate collection path. These are application behaviors worth preserving
when moving the model into another provider.[^26]

## Why this combination could feel better with shaders

The hypothesis is not that VACE adds detail by itself. Rather, it allows the
generation system to divide responsibilities:

- **Control input:** the shape, spatial layout, and motion to respond to.
- **Prompt and optional compatible style LoRA:** what the scene should become.
- **Generated history:** what the scene has already become.
- **Transition policy:** how much of that history to retain as the direction changes.

An abstract shader contains relatively little semantic information. A system
that strongly preserves its final RGB appearance may faithfully reproduce smooth
color blobs and bloom. A system that largely ignores it may invent detailed
imagery but lose the performance's gesture. Separate structural conditioning
offers a way to explore the space between those failures. This remains a
hypothesis to test with matched inputs, not an explanation established by a
single successful demo.

The current test clip is especially difficult: broad pink/green luminous forms
include clipped near-white areas. Once pixels are clipped, an exposure operation
cannot recover their hidden structure. A shader-native control pass captured
before bloom or tone mapping would be a cleaner experiment than trying to
recover structure from the final RGB output.

## LoRA and decoder choices can change the answer

Scope loads the LongLive base, adds VACE, then installs LongLive's built-in
performance adapter and any additional user adapters. The built-in adapter is
not the same thing as an optional artistic LoRA. A faithful comparison must
preserve the required adapter and record additional style adapters separately.
Scope's code explicitly treats this ordering as important.[^18]

A style LoRA may be responsible for much of a striking demo's material or
illustrative character. VACE can help align that style with motion, but it should
not receive sole credit for the appearance. A matched no-style-LoRA run is needed
before drawing that conclusion.

Scope exposes four autoencoder choices. Its documentation describes full WanVAE
as the quality-first default, LightVAE as a pruned alternative, TAE as a faster
preview-oriented architecture, and LightTAE as another speed/quality compromise.
Replacing the decoder can materially change fine detail and texture. It is not
equivalent to changing the MP4 playback rate.[^19]

## What is actually established about RTX 5090 performance

Daydream's VACE adaptation paper reports the following LongLive 1.3B results on
an RTX 5090 at **640x368**, four steps, BF16, SageAttention, and a TAE decoder.
It measures fifteen chunks after three warmups, with twelve frames per chunk.
These are author-reported inference measurements, not Tulip results.[^20]

| Configuration | Average chunk time | Generated FPS | Reported peak VRAM |
|---|---:|---:|---:|
| No VACE | 539 ms | 22.3 | 13.2 GB |
| VACE depth control | 698 ms | 17.2 | 14.6 GB |
| VACE inpainting | 698 ms | 17.2 | 14.6 GB |

The paper excludes streaming overhead and warns that reference-image fidelity
degrades relative to batch VACE. These numbers do not establish capture-to-display
latency, full-WanVAE speed, or performance at 832x480.[^20]

Scope's documentation identifies 832x480 as LongLive's training resolution and
notes the quality benefit of using it. That has about 1.70 times the pixels of
640x368; throughput cannot be extrapolated linearly because attention, VAE work,
memory traffic, and fixed costs scale differently.[^4]

### Chunk latency still matters

A twelve-frame input chunk takes eleven frame intervals from its first to its
last captured frame: approximately 688 ms at 16 FPS or 458 ms at 24 FPS. These
are arithmetic accumulation delays for an ordinary full-chunk collector, not
measured Scope latency. Initial-frame padding, frame selection, overlap, and
transport behavior can alter the actual correspondence.

Processing time, preprocessing, queueing, and delivery come after or overlap that
collection depending on the implementation. A stream can sustain a useful
throughput while responding noticeably late to a gesture. Conversely, feeding
24 distinct frames every second into a 17-FPS generator cannot remain lossless
and bounded-latency indefinitely: it must slow capture, drop/select frames,
reduce work, or accumulate delay.

The B300 may have enabled a preferred combination of full-resolution generation,
decoder quality, and prompt responsiveness. The evidence does not establish
that the GPU itself supplied an otherwise unavailable aesthetic; it also does
not establish that the same settings will fit the 5090's desired latency budget.

## Transferability to other models

### VACE weights are not a generic video plugin

The weight loader filters for VACE context-block and patch-embedding parameters,
checks dimensions, and rejects incompatible shapes or missing VACE keys. It is
designed around compatible Wan architectures. Adding a similarly named API field
to SANA or JoyAI would not supply trained conditioning behavior.[^21]

The transferable idea is a separate, trained structural-control branch combined
with correct streaming state. Transplanting actual Wan VACE weights into a
different transformer family requires architectural compatibility and evidence
of learned-feature compatibility; otherwise new adapter training or distillation
is likely required. Even two models with “Wan” in their lineage are not
automatically checkpoint-compatible.

### Candidate directions

| Direction | Why it is relevant | Main unresolved issue |
|---|---|---|
| LongLive 1.3B + Scope VACE | Closest controlled reference for the existing successful experience | Matching the old preset and acceptable 5090 quality/latency |
| RewardForcing + the same VACE/control setup | Scope-supported Wan2.1 1.3B alternative with reward-based training | Whether its visual preferences improve abstract shader transformation |
| MemFlow + the same VACE/control setup | Scope-supported relative of LongLive with additional memory | Memory can improve persistence but may increase inertia or cost |
| JoyAI isolated trial | Actual incremental input and persistent streaming implementation already under test | Shader reinterpretation and usable VRAM/performance remain unverified |
| Different backbone plus a new VACE-like adapter | Potentially preserves the right control abstraction | A training/architecture project, not a small provider integration |

Scope explicitly supports VACE and compatible 1.3B LoRAs for RewardForcing and
MemFlow. Those facts justify comparative tests, not a ranking of output quality.
The MemFlow documentation also warns that its discussed sparse-memory activation
is not implemented in the referenced release; paper efficiency should not be
assumed to exist in the installed runtime.[^22][^23]

There is also a newer **LongLive 2.0**, released in May 2026, with a 5B Wan2.2
path and FP8/NVFP4 inference options. It is distinct from Scope's inspected
1.3B integration. Its headline 45.7 FPS is on GB200, not RTX 5090, and a
drop-in Scope VACE adaptation was not established by the inspected materials.
It deserves a separate compatibility investigation, not automatic substitution
of its advertised speed into this comparison.[^24][^25]

## Recommended controlled experiment

The most informative next test is a **conditioning ablation**, not another
unmatched model demo. This is a proposed experiment; it has not been run here.

1. Recover the successful B300 preset where possible: exact pipeline/version,
   resolution, decoder, denoising schedule, VACE routing/scale, style LoRAs,
   input preprocessing, prompt-transition settings, and output interpolation.
2. Keep the same source clip, prompt schedule, seed, and native output sampling.
   Use a no-style-LoRA baseline plus the known artistic adapter separately.
3. Compare standard source-latent V2V with VACE RGB, grayscale, and contour
   conditioning. Use shader-native depth when it exists. Estimated monocular
   depth of abstract emissive blobs is not necessarily meaningful geometry.
4. Sweep VACE context scale, for example 0, 0.35, 0.7, 1.0, and 1.3. These are
   exploratory values, not validated presets. Scale zero is a useful negative
   control for whether the structural pathway is actually influencing output.
5. Compare smooth prompt transitions with direct updates and intentional resets.
   Inspect both response to the new prompt and retention of visual identity.
6. Establish a quality-first full-WanVAE reference before testing TAE/LightTAE or
   fewer denoising steps for speed. Otherwise two independent quality changes
   can be incorrectly blamed on the new generator.
7. Once the best control representation is known, repeat it on RewardForcing,
   MemFlow, and the JoyAI candidate where their interfaces genuinely support
   an equivalent experiment. Do not label ordinary RGB conditioning as depth
   control unless that model was trained to interpret it that way.

The evaluation should score five properties independently: **motion response,
semantic transformation, temporal persistence, detail/material quality, and
prompt responsiveness**. Save timestamped input/output pairs and observe chunk
boundaries. Report warm native generated FPS separately from displayed FPS,
first-output delay, p50/p95 input-to-output latency, prompt-switch stalls,
preprocessor cost, and process GPU-memory peaks. Include longer runs to expose
drift that a two-second clip cannot reveal.

The decision criterion is not maximum source similarity. For a generative
instrument, success means the output follows a performed gesture while creating
visually interesting new content and retaining enough history to feel like an
evolving scene. VACE is promising precisely because it provides a separate
control dimension for that balance.

## Sources

Code links are pinned to the inspected commits. Documentation and papers were
accessed September 10, 2026. Publications' performance results remain attributed
to their authors and were not reproduced on Tulip during this investigation.

[^1]: Daydream Scope, [graph input routing](https://github.com/daydreamlive/scope/blob/2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0/src/scope/server/graph_schema.py), `build_linear_graph`.
[^2]: Daydream Scope, [latent-preparation routing](https://github.com/daydreamlive/scope/blob/2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0/src/scope/core/pipelines/wan2_1/blocks/auto_prepare_latents.py).
[^3]: Daydream Scope, [VACE residual hint injection](https://github.com/daydreamlive/scope/blob/2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0/src/scope/core/pipelines/wan2_1/vace/models/attention_blocks.py).
[^4]: Daydream, [LongLive pipeline reference](https://docs.daydream.live/scope/reference/pipelines/longlive), undated current documentation.
[^5]: Daydream Scope, [source-latent/noise mixing](https://github.com/daydreamlive/scope/blob/2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0/src/scope/core/pipelines/wan2_1/blocks/prepare_video_latents.py).
[^6]: Daydream Scope, [mode and noise-control defaults](https://github.com/daydreamlive/scope/blob/44ccdd5c24d7249f5c79dbd9648d5b046970805d/src/scope/core/pipelines/defaults.py).
[^7]: Daydream Scope, [causal VACE wrapper](https://github.com/daydreamlive/scope/blob/2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0/src/scope/core/pipelines/wan2_1/vace/models/causal_vace_model.py).
[^8]: Daydream Scope, [VACE input encoding and mode handling](https://github.com/daydreamlive/scope/blob/2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0/src/scope/core/pipelines/wan2_1/vace/blocks/vace_encoding.py).
[^9]: Daydream Scope, [dual-stream encoding, masks, and caches](https://github.com/daydreamlive/scope/blob/2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0/src/scope/core/pipelines/wan2_1/vace/utils/encoding.py).
[^10]: Daydream Scope, [denoising loop](https://github.com/daydreamlive/scope/blob/44ccdd5c24d7249f5c79dbd9648d5b046970805d/src/scope/core/pipelines/wan2_1/blocks/denoise.py).
[^11]: Shuai Yang et al., [LongLive: Real-time Interactive Long Video Generation](https://arxiv.org/html/2509.22622v1), September 2025, sections 3–4.
[^12]: Daydream Scope, [LongLive model configuration](https://github.com/daydreamlive/scope/blob/2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0/src/scope/core/pipelines/longlive/model.yaml).
[^13]: Daydream Scope, [causal attention and sink handling](https://github.com/daydreamlive/scope/blob/44ccdd5c24d7249f5c79dbd9648d5b046970805d/src/scope/core/pipelines/longlive/modules/causal_model.py).
[^14]: Daydream Scope, [cached video decoding](https://github.com/daydreamlive/scope/blob/44ccdd5c24d7249f5c79dbd9648d5b046970805d/src/scope/core/pipelines/wan2_1/blocks/decode.py).
[^15]: Daydream Scope, [LongLive recache block](https://github.com/daydreamlive/scope/blob/2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0/src/scope/core/pipelines/longlive/blocks/recache_frames.py).
[^16]: Daydream Scope, [cache reset and transition conditions](https://github.com/daydreamlive/scope/blob/44ccdd5c24d7249f5c79dbd9648d5b046970805d/src/scope/core/pipelines/wan2_1/blocks/setup_caches.py).
[^17]: Daydream Scope, [embedding transition implementation](https://github.com/daydreamlive/scope/blob/44ccdd5c24d7249f5c79dbd9648d5b046970805d/src/scope/core/pipelines/wan2_1/blocks/embedding_blending.py).
[^18]: Daydream Scope, [LongLive initialization and adapter ordering](https://github.com/daydreamlive/scope/blob/2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0/src/scope/core/pipelines/longlive/pipeline.py).
[^19]: Daydream, [VAE types](https://docs.daydream.live/scope/reference/vae), current documentation.
[^20]: Ryan Fosdick / Daydream, [Adapting VACE for Real-Time Autoregressive Video Diffusion](https://arxiv.org/html/2602.14381v1), February 16, 2026, table 5 and limitations.
[^21]: Daydream Scope, [VACE weight validation/loading](https://github.com/daydreamlive/scope/blob/2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0/src/scope/core/pipelines/wan2_1/vace/utils/weight_loader.py).
[^22]: Daydream, [RewardForcing pipeline](https://docs.daydream.live/scope/reference/pipelines/reward-forcing), current documentation.
[^23]: Daydream, [MemFlow pipeline](https://docs.daydream.live/scope/reference/pipelines/memflow), current documentation including sparse-memory implementation caveat.
[^24]: NVIDIA, [LongLive repository and release history](https://github.com/NVlabs/LongLive), current main branch; 2.0 released May 13, 2026 and FP8 support announced July 8, 2026.
[^25]: NVIDIA, [LongLive 2.0 project results](https://nvlabs.github.io/LongLive/LongLive2/), GB200-specific headline performance.
[^26]: Daydream Scope, [parameter coalescing and scalar-port delivery](https://github.com/daydreamlive/scope/blob/44ccdd5c24d7249f5c79dbd9648d5b046970805d/src/scope/server/pipeline_processor.py), `process_chunk` and `_drain_non_video_inputs`.
