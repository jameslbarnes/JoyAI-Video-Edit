# Reference image trial, 2026-09-15

User requested the supplied portrait with a man walking a dog.

Two delivery problems were found and corrected:
- Etherea now forwards multistream referenceImage to JoyAI start.ref_image, and exposes referenceImageActive in runtime stats.
- LocalPreset in tulip/serve.py had discarded ref_image. It now passes bounded string values through. The runtime was reloaded through SDK lifecycle endpoints; all existing presentation settings were preserved.
- A seedPrompt alone was overwritten by canonical Cue state. Sending the requested scene via SDK cue.transcript.send updated the actual live scene. Subsequent guidance asked to preserve the reference man's facial features.

Verified: runtime has_ref_image=true, initialized=true, graph_path=1, three bounded KV entries including the static reference, text encoder fully resident on GPU. Direct output remains 1944x1080, RIFE 2x, neural rendering off; encoder remains 10 Mbps. A 39-second measured window yielded 30.68 fresh NDI frames/s, no output drops or NDI errors, latency 1393 to 1404 ms. This short window included a scene transition and is not a long-term benchmark.

Visual result is NOT a successful likeness test. After fixing canonical prompt delivery, captured output showed a man with a leashed dog, but the man's face was poorly defined. Later explicit reference guidance yielded distorted human/animal facial blending. Technical activation does not establish faithful identity preservation. No transport quality change was made to compensate.

Evidence: SDK logs/reference-man-dog-soak-20260915.json, logs/reference-final-status-20260915.json; local PNG captures reference-trial-cue-output-20260915.png and reference-trial-explicit-output-20260915.png. Original portrait is local reference-man-dog-20260915.jpg. Trial image is held in the current browser stream; persistence across tile recreation/app reload is not implemented.

Validation: Node syntax checks and actual beginSession payload forwarding with/without reference; Python compile; isolated LocalPreset ASGI check for valid/absent/invalid/oversized reference values, pinned geometry and RIFE. Live operations hold was resumed successfully.
