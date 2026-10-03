"""Junk vs continuity scorecard across replay arms (same input/prompt/seed).

Usage: python tulip/drift_lab/scorecard.py RUN_DIR arm_path [arm_path ...]
arm_path is relative to RUN_DIR, e.g. batch2/baseline_eager.
  junk m0/m3      mean junk in minute 0 and minute 3 of output
  jump            mean frame-to-frame change at chunk boundaries (8-frame chunks)
  typical         median frame-to-frame change across chunk positions
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def junk(path, lo, hi):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    vals = [r["junk"] for r in rows if lo <= r["t"] < hi]
    return sum(vals) / len(vals) if vals else float("nan")


def continuity(video, frames=2800):
    cap = cv2.VideoCapture(str(video))
    prev, diffs = None, []
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, (288, 162)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(np.abs(g - prev).mean())
        prev = g
    d = np.array(diffs)
    idx = np.arange(1, len(d) + 1)
    by = [d[(idx % 8) == k][3:].mean() for k in range(8)]
    return by[1], float(np.median(by))


def main():
    root = Path(sys.argv[1])
    print(f"{'arm':26s} {'junk m0':>8s} {'junk m3':>8s} {'jump':>6s} {'typical':>8s}")
    for arm in sys.argv[2:]:
        d = root / arm
        jump, typical = continuity(d / "output.mp4")
        print(f"{arm:26s} {junk(d / 'metrics.jsonl', 0, 60):8.4f} {junk(d / 'metrics.jsonl', 180, 240):8.4f} "
              f"{jump:6.2f} {typical:8.2f}")


if __name__ == "__main__":
    main()
