"""Summarize drift replay arms: metric trends over output time + contact sheets.

Usage: python tulip/drift_lab/analyze.py RUN_DIR [RUN_DIR ...]
Finds every */metrics.jsonl below the given dirs; prints per-arm trend table
(first 2 min vs last 2 min, slope per 10 min) and writes contact.jpg per arm.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

KEYS = ("junk", "detail", "staleness", "imprint", "brightness")


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def trend(rows, key):
    pts = [(r["t"], r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    if len(pts) < 10:
        return None
    t, v = np.array(pts).T
    early = v[t <= t.min() + 120].mean()
    late = v[t >= t.max() - 120].mean()
    slope = np.polyfit(t / 600.0, v, 1)[0]  # per 10 minutes
    return early, late, slope


def contact(arm_dir):
    shots = sorted(arm_dir.glob("f*.jpg"))
    if not shots:
        return
    picks = [shots[int(i)] for i in np.linspace(0, len(shots) - 1, min(8, len(shots)))]
    tiles = []
    for p in picks:
        img = cv2.imread(str(p))
        img = img[:, : img.shape[1] // 2]  # output half only
        cv2.putText(img, f"t={int(p.stem[1:]) // 12}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        tiles.append(img)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(arm_dir / "contact.jpg"), np.vstack(rows))


def main():
    for root in map(Path, sys.argv[1:]):
        for mpath in sorted(root.rglob("metrics.jsonl")):
            rows = load(mpath)
            if not rows:
                continue
            arm = mpath.parent
            span = rows[-1]["t"]
            parts = [f"{arm.relative_to(root)}  ({len(rows)} samples, {span / 60:.1f} min)"]
            for key in KEYS:
                tr = trend(rows, key)
                if tr:
                    parts.append(f"  {key:10s} early {tr[0]:.4f}  late {tr[1]:.4f}  slope/10min {tr[2]:+.4f}")
            chunks = arm / "chunks.jsonl"
            if chunks.exists():
                cs = load(chunks)
                for field in ("latent", "sinkKey", "sinkValue"):
                    vals = [c[field]["rms"] for c in cs if isinstance(c.get(field), dict)]
                    if len(vals) > 20:
                        parts.append(f"  {field + ' rms':10s} early {np.mean(vals[:20]):.3f}  late {np.mean(vals[-20:]):.3f}")
                clip = [c.get("clipFraction") or 0 for c in cs]
                if clip:
                    parts.append(f"  clipFrac   early {np.mean(clip[:20]):.4f}  late {np.mean(clip[-20:]):.4f}")
            print("\n".join(parts))
            contact(arm)


if __name__ == "__main__":
    main()
