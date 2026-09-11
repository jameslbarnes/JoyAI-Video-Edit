"""Decode every frame, quantify changes and create labeled input/output evidence."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--chunk-size", type=int, default=8)
p.add_argument("--first-chunk", type=int, default=1)
p.add_argument("--loop-input", action="store_true", help="Match a diagnostic run that loops its fixed input")
a = p.parse_args()
dest = Path(a.output).parent
src, gen = cv2.VideoCapture(a.input), cv2.VideoCapture(a.output)
assert src.isOpened() and gen.isOpened()
fps = gen.get(cv2.CAP_PROP_FPS)
count = int(gen.get(cv2.CAP_PROP_FRAME_COUNT))
indices = sorted(
    {
        0,
        1,
        a.first_chunk - 1,
        a.first_chunk,
        a.first_chunk + a.chunk_size - 1,
        a.first_chunk + a.chunk_size,
        count // 3,
        2 * count // 3,
        count - 1,
    }
)
rows, changes, source_distances, deviations = [], [], [], []
prev = None
decoded = 0
while True:
    ok, output = gen.read()
    if not ok:
        break
    ok, inp = src.read()
    if not ok and a.loop_input:
        src.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, inp = src.read()
    if not ok:
        raise RuntimeError("Input ended before output")
    inp = cv2.resize(inp, (output.shape[1], output.shape[0]))
    source_distances.append(float(np.abs(output.astype(float) - inp).mean()))
    deviations.append(float(output.std()))
    if prev is not None:
        changes.append(
            {
                "frame": decoded,
                "mean_abs_delta": float(np.abs(output.astype(float) - prev).mean()),
                "chunk_start": decoded >= a.first_chunk
                and (decoded - a.first_chunk) % a.chunk_size == 0,
            }
        )
    if decoded in indices:
        panels = [cv2.resize(x, (432, 240)) for x in (inp, output)]
        row = np.zeros((267, 864, 3), dtype=np.uint8)
        row[27:, :432], row[27:, 432:] = panels
        cv2.putText(
            row,
            f"Input {decoded} / {decoded / fps:.2f}s",
            (8, 20),
            0,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(row, f"Generated {decoded}", (440, 20), 0, 0.5, (255, 255, 255), 1)
        rows.append(row)
        cv2.imwrite(str(dest / f"review-frame-{decoded:04d}.png"), output)
    prev = output.astype(float)
    decoded += 1
src.release()
gen.release()
assert decoded == count and decoded > 1, (decoded, count)
for page in range(0, len(rows), 3):
    cv2.imwrite(
        str(dest / f"comparison-{page // 3 + 1}.jpg"),
        np.concatenate(rows[page : page + 3]),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
report = {
    "decoded_frames": decoded,
    "container_fps_not_generation_fps": fps,
    "source_pixel_MAD_mean": float(np.mean(source_distances)),
    "output_std_min": min(deviations),
    "consecutive_delta": changes,
    "exact_frozen_pairs": sum(x["mean_abs_delta"] == 0 for x in changes),
    "note": "Pixel statistics are diagnostic, not perceptual-quality or motion-fidelity scores.",
}
(dest / "decode-audit.json").write_text(json.dumps(report, indent=2))
print(
    json.dumps({k: v for k, v in report.items() if k != "consecutive_delta"}, indent=2)
)
