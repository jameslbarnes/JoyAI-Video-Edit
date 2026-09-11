"""Package browser-generated JPEG evidence; playback FPS is not a benchmark."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
frames = json.loads((args.directory / "frames.json").read_text())
report = {"playback_fps": 16, "timing_note": "Unique generated frames at nominal playback rate; elapsed stalls are omitted. Use metrics.json for real latency/throughput.", "epochs": []}
for epoch in sorted({frame["epoch"] for frame in frames}):
    selected = [frame for frame in frames if frame["epoch"] == epoch]
    target = args.directory / f"epoch-{epoch}.mp4"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 16, (648, 360))
    if not writer.isOpened():
        raise RuntimeError("Video writer failed")
    previous = None
    repeated = 0
    differences = []
    try:
        for frame in selected:
            pixels = cv2.imread(str(args.directory / frame["filename"]))
            if pixels is None or pixels.shape != (360, 648, 3):
                raise RuntimeError(f"Invalid generated frame {frame['filename']}")
            if previous is not None:
                repeated += int(np.array_equal(previous, pixels))
                differences.append(float(np.abs(previous.astype(np.float32) - pixels).mean()))
            previous = pixels
            writer.write(pixels)
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(target))
    decoded = 0
    while True:
        ok, pixels = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    if decoded != len(selected):
        raise RuntimeError(f"Decoded {decoded}, expected {len(selected)}")
    report["epochs"].append({"epoch": epoch, "video": target.name,
                             "decoded_frames": decoded, "exact_adjacent_repeats": repeated,
                             "mean_adjacent_pixel_difference": float(np.mean(differences))})
(args.directory / "video-audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
