# Native live-show A/B capture

The user asked to compare interpolation using the actual live show. The
production interpolation facade now has an optional native RGB capture tap.
It receives the existing CPU arrays before interpolation/VSR, so recording
adds no GPU download. It does not change engine selection, creative settings,
native resolution, cadence, or fallback behavior.

Capture is off when `tulip/native-capture.json` is missing or `{}`. To capture
again without restarting JoyAI, atomically write a unique command such as:

```json
{"id":"live-ab-new-unique-id","max_frames":192}
```

Output is `tulip/evidence/native-captures/<id>/`, with lossless NumPy chunks
and a final manifest. IDs are restricted to lowercase letters/digits/dashes/
underscores and never overwrite an existing directory. Maximum capture is
600 native frames. A four-chunk asynchronous disk queue bounds memory; queue
pressure marks capture invalid rather than blocking the renderer. Observe
`native_capture` in the existing debug profile. A valid manifest requires
`complete:true`, no error, and the requested written-frame count. Scene resets
are recorded as segment boundaries. Disarm with `{}` after completion.

The one-time `capture_live_interpolation.py` deployment requested the saved
prompt/strength through the existing lifecycle helper and retained
1152x648/12-native/2x-DLSSG/VSR Ultra/1440p NVENC settings. Its original facade
backup is SHA-256 verified against the previous benchmark manifest. No
automatic rollback was added. Evidence: `evidence/live-show-ab-20260924/`.
The show supplied newer prompts during startup/capture; the helper's ownership
check left that newer state in place. The snapshots retain this history.

`live_show_interpolation_ab.py prepare` selects a contiguous captured scene
segment of at least 97 inputs. `rife` and `dlssg` modes feed identical inputs
through the production GPU adapters, assert native originals remain exact,
and preserve profiles/counts. `present --method rife|dlssg` uses identical
VSR Ultra/NVENC settings at 24 fps. `report` randomizes A/B labels, copies
encoded files without re-encoding, and creates a single-player local viewer
with an optional engine reveal. This is a subjective comparison: there is
no native 24 fps ground truth for the live clip, so do not claim PSNR accuracy.

Validation: native-capture ownership, cap, disabled path, bounded queue,
scene segmentation, path constraints, existing-ID protection, cancellation,
backend selection and presentation/fallback checks (19 tests). The real
capture and comparison have their own manifests and before/after health data.
