# JoyAI prompting audit — September 16, 2026

Research and read-only inspection; no prompting configuration or live stream was changed.

## Primary references

- [Official current prompt enhancer](https://github.com/jd-opensource/JoyAI-Video-Edit/blob/main/deploy/xvideo/serving/pe.py), retrieved through its raw main-branch file. Search results also exposed an older version, so this audit uses the current file.
- [Official repository and examples](https://github.com/jd-opensource/JoyAI-Video-Edit).
- [Technical report](https://arxiv.org/html/2608.03974v1).

The enhancer favors a cohesive English editing instruction, prioritizing the requested change, concrete entities, physical interaction, and consistent tracking. Preservation clauses should cover only untouched content and must not contradict transformations. It supports explicit removal commands and requested exact text. Its webcam-specific chair/glasses recipes should not be copied indiscriminately into our shader workflow. The report describes conditioning on the source frame and editing instruction. These sources do not establish an optimal 70-word ceiling or guarantee that one wording improves our particular shader-driven output.

## Actual live path

Cue house session -> canonical prompt distribution -> JoyAI streaming runtime. Live Cue status and the actual recent LLM request both identify the active output as FluxRT/FLUX.2 Klein, with an Odyssey-mix description. This is not merely an obsolete file: it is in the messages sent to the director. The provider resolver in `cue/examples/etherea-prompt/server.config.ts:486` and guide selector in `prompts.ts:243` support longlive/ltx/fluxrt/odyssey but have no JoyAI branch. Selecting JoyAI by name without implementing that branch would fall back to LongLive.

The live guide imposes 40–70 words, positive-only wording, one hero, and one slow motion. It says to remix even when speech adds nothing new. The base system and final user instruction also permit passing on filler, while the runtime policy forbids that during speech. This is contradictory creative routing, independent of JoyAI model behavior. Do not silently change the user's intended show pacing as part of model guidance.

JoyAI's separate prompt enhancer is disabled (`debug.ws.use_pe=false`), so the upstream reference enhancer does not repair the mismatch. Keeping one author in Cue avoids an additional rewrite stage; a JoyAI-specific Cue guide is the recommended intervention.

## What already works

- Actual live prompts use editing verbs, concrete materials/colors/light, and preservation clauses. The butterfly/caterpillar examples were 53 and 58 words.
- Both sampled Cue requests recorded `multimodalFrameAttached=true`. This proves attachment, not that Cue sees the identical source frame used by JoyAI.
- The actual update request included the preceding full butterfly prompt. Runtime profile: `showCurrentPromptInFull=true`, priorPromptCount=3; compact context preserves the latest full prompt. Transcript window is 75 seconds, up to eight segments, newest four seconds highlighted.
- JoyAI independently encodes the instruction with a source image; prompt updates use the latest source-frame copy. The inspected live prompt used 437 of 1,024 conditioning tokens. An earlier example used 444. No evidence of capacity pressure in these samples.

## Recommended next experiment

Add and correctly select a JoyAI Cue profile. Put the requested operation and affected subject first; state the desired result and the specific content that stays. Retain the operator's chosen style, meaningful subjects, and requested motion. Use concise prose without an arbitrary hard word cap; spend additional words only on details needed to satisfy the request. Permit direct object removal and explicitly requested text. Do not automatically invent extra camera motion or replace requested multi-subject scenes with one subject.

Compare matched inputs and source frames for style change, removal, background change, subject replacement, and a multi-subject request. Judge instruction adherence, preservation, temporal stability, and prompt latency. This audit establishes a configuration mismatch; visual benefit from replacing the guide is still unmeasured.

Selective live evidence: `prompt-audit-20260916.json`. No model/stream restart or benchmark was performed for this audit.
