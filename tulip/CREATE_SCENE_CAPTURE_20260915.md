# Prepared scene capture

The desktop client's initial one-frame input gate remained active during a
prepared scene reset. It waited for initial generation, RIFE scene morph and SR
delivery before capturing the next eight inputs. The direct NDI bridge reads
the session's latest source frame, so this gate also froze its shader input.

Fix in Etherea `src/web-client/static/js/joyai-client.js`: a prepared RIFE scene
uses the existing bounded input credit (24 frames with SR, otherwise 16) before
first output. Cold startup retains the one-frame gate. No model, graph, prompt,
resolution, RIFE multiplier or presentation-buffer changes.

Applied to the running client prototype through minimal Runtime.evaluate after
four Node tests passed. The same code is saved in the client source for future
reloads. SDK ops hold/resume surrounded live checks; no runtime restart needed.

Evidence in the SDK repository:

- `logs/restart-20260915-204748/scene-transition-observation.json`: baseline
  handoff about 3.0 seconds; 31.90 fresh NDI frames/s; no drops.
- `logs/restart-20260915-204942/scene-transition-observation.json`: patched
  handoff 2.00 seconds; 31.96 fps; all 15 morph frames; no drops. Initial summary
  script had a KeyError after saving observations; summary was recovered from
  those observations and the script corrected.
- `logs/restart-20260915-205109/scene-transition-observation.json`: second
  25-second observation, first handoff 2.20 seconds, 31.88 fps, no output drops,
  minimum five fresh NDI ingests per 200 ms sample. Another scene reset occurred
  during observation (epoch 11 to 13); both completed, with 30 morph frames.
- Same directory `desktop-gaps.json`: max presentation callback gap 63 ms,
  p99 37.1 ms, including resets.
- Same directory `warm-scene.mp4`: 15-second copy of existing RTMP encoded feed,
  480 frames at 32 fps. Offline 216x120 RGB mean absolute frame differences:
  longest run below 0.1/255 was two frames (62.5 ms). This checks visual motion,
  unlike transport counts alone; it is not a perceptual-quality benchmark.

Runtime readback confirmed masked capacity-1024 prompt graph still active and
all workers alive. RIFE remained 2x, reference image disabled, 1944x1080 direct
output. Probe callbacks were removed and SDK supervision resumed.

Validation: from Etherea run `node --test
scripts/streaming/joyai-scene-transition.test.mjs
scripts/streaming/joyai-capture-clock.test.mjs` (four passing tests).
