"""Deterministic, code-rendered structured-motion input; no generated AI frames."""

import json
from pathlib import Path
import cv2
import numpy as np

dest = Path(__file__).resolve().parent / "evidence" / "structured-input"
dest.mkdir(parents=True, exist_ok=True)
if (dest / "input.mp4").exists():
    raise SystemExit("Refusing to overwrite existing input")
w, h, fps, count = 648, 360, 16, 97
writer = cv2.VideoWriter(
    str(dest / "input.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)
assert writer.isOpened()
vertices = np.array(
    [[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]], dtype=float
)
faces = [
    [0, 1, 3, 2],
    [4, 6, 7, 5],
    [0, 4, 5, 1],
    [2, 3, 7, 6],
    [0, 2, 6, 4],
    [1, 5, 7, 3],
]
for frame in range(count):
    t = frame / fps
    canvas = np.full((h, w, 3), (45, 40, 35), np.uint8)
    cv2.rectangle(canvas, (0, 260), (w, h), (76, 68, 59), -1)
    for x in range(-w, w * 2, 60):
        cv2.line(canvas, (w // 2, 260), (x, h), (95, 87, 76), 1)
    for y in [276, 300, 334]:
        cv2.line(canvas, (0, y), (w, y), (95, 87, 76), 1)
    angle = t * 0.6
    ry = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    rx = np.array([[1, 0, 0], [0, 0.94, -0.342], [0, 0.342, 0.94]])
    xyz = vertices @ ry @ rx
    screen = (
        xyz[:, :2] * 58
        + np.array([205 + 35 * np.sin(t * 0.8), 174 + 12 * np.sin(t * 1.2)])
    ).astype(int)
    cv2.ellipse(canvas, (205, 276), (87, 14), 0, 0, 360, (35, 32, 29), -1)
    for idx in sorted(range(6), key=lambda i: xyz[faces[i], 2].mean()):
        color = (80 + idx * 12, 125 + idx * 13, 45 + idx * 6)
        cv2.fillConvexPoly(canvas, screen[faces[idx]], color)
        cv2.polylines(
            canvas, [screen[faces[idx]]], True, (200, 205, 160), 2, cv2.LINE_AA
        )
    bx = int(447 + 54 * np.sin(t * 1.1))
    by = int(204 + 36 * np.sin(t * 1.7))
    cv2.ellipse(canvas, (bx, 279), (53, 10), 0, 0, 360, (35, 32, 29), -1)
    for radius in range(39, 0, -1):
        level = 1 - radius / 39
        cv2.circle(
            canvas,
            (bx, by),
            radius,
            (int(50 + 30 * level), int(65 + 50 * level), int(150 + 65 * level)),
            -1,
        )
    cv2.line(canvas, (bx - 25, by), (bx + 25, by), (110, 140, 235), 3, cv2.LINE_AA)
    writer.write(canvas)
    if frame in (0, 32, 64, 96):
        cv2.imwrite(str(dest / f"frame-{frame:04d}.png"), canvas)
writer.release()
(dest / "input-manifest.json").write_text(
    json.dumps(
        {
            "type": "deterministic synthetic 3D-like primitives",
            "width": w,
            "height": h,
            "fps": fps,
            "frames": count,
            "interpolation": False,
            "motion": "rotating translating cube and independently oscillating sphere",
        },
        indent=2,
    )
)
print(dest / "input.mp4")
