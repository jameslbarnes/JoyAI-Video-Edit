"""Score dial-sweep replay arms: how closely output follows the (aligned) input.

Usage: python tulip/drift_lab/dial_score.py RUN_DIR INPUT_CLIP arm [arm ...]
  struct   mean correlation of blurred output vs input luminance (layout)
  edges    mean correlation of blurred edge maps (shapes)
  rot/pan  median |rotation| deg/s and pan %width/s, output vs input
  track    correlation of per-step output pan vectors with input pan vectors
           (1 = moves with the shader, 0 = unrelated motion)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

STEP = 3  # 12 fps output -> 4 fps samples


def frames(path, step, size=(480, 270), limit=None, src_step=1):
    cap = cv2.VideoCapture(str(path))
    out, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % src_step == 0:
            out.append(cv2.cvtColor(cv2.resize(f, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
        i += 1
    return out[::step][:limit]


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def step_motion(a, b):
    p0 = cv2.goodFeaturesToTrack(a, 400, 0.01, 8)
    if p0 is None or len(p0) < 20:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None)
    ok = st.ravel() == 1
    if ok.sum() < 20:
        return None
    m, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC)
    if m is None:
        return None
    return np.degrees(np.arctan2(m[1, 0], m[0, 0])), m[0, 2], m[1, 2]


def main():
    root, clip, arms = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    cap = cv2.VideoCapture(clip)
    src_step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 24) / 12))
    cap.release()
    inp = frames(clip, STEP, src_step=src_step)
    print(f"{'arm':10s} {'struct':>6s} {'edges':>6s} {'rot out/in':>11s} {'pan out/in':>11s} {'track':>6s}")
    for arm in arms:
        out = frames(root / arm / "output.mp4", STEP)
        n = min(len(out), len(inp))
        low = lambda f: cv2.GaussianBlur(f.astype(np.float32), (0, 0), 6)
        edg = lambda f: cv2.GaussianBlur(cv2.Canny(f, 50, 150).astype(np.float32), (0, 0), 2)
        struct = np.mean([corr(low(out[i]), low(inp[i])) for i in range(n)])
        edges = np.mean([corr(edg(out[i]), edg(inp[i])) for i in range(n)])
        mo = [step_motion(out[i], out[i + 1]) for i in range(n - 1)]
        mi = [step_motion(inp[i], inp[i + 1]) for i in range(n - 1)]
        pairs = [(o, x) for o, x in zip(mo, mi) if o is not None and x is not None]
        fps = 12 / STEP
        rot_o = np.median([abs(o[0]) for o, _ in pairs]) * fps
        rot_i = np.median([abs(x[0]) for _, x in pairs]) * fps
        pan_o = np.median([np.hypot(o[1], o[2]) for o, _ in pairs]) * fps / 480 * 100
        pan_i = np.median([np.hypot(x[1], x[2]) for _, x in pairs]) * fps / 480 * 100
        vo = np.array([[o[1], o[2]] for o, _ in pairs]).ravel()
        vi = np.array([[x[1], x[2]] for _, x in pairs]).ravel()
        track = corr(vo, vi) if len(vo) > 4 else float("nan")
        print(f"{arm:10s} {struct:6.3f} {edges:6.3f} {rot_o:5.1f}/{rot_i:4.1f} {pan_o:5.1f}/{pan_i:4.1f} {track:6.3f}")


if __name__ == "__main__":
    main()
