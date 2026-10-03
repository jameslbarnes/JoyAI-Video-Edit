"""How strongly the live output follows the shader input (read-only).

Usage: python tulip/drift_lab/dial_probe.py [seconds=15]
Captures output and input at 4 fps, low-passes both (structure, not texture)
and reports the best correlation over output lags 0..3 s (the pipeline delays
output ~1.5-2 s), plus edge-map correlation at that lag. Higher = the dial is
pulling the picture toward the shader.
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from motion_probe import FPS, grab, motion  # noqa: E402
import threading  # noqa: E402


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-9 else 0.0


def adherence(out, inp):
    low = lambda f: cv2.GaussianBlur(f.astype(np.float32), (0, 0), 6)
    edges = lambda f: cv2.Canny(f, 50, 150).astype(np.float32)
    best = (-1, 0, 0)
    n = min(len(out), len(inp))
    for lag in range(0, 3 * FPS + 1):
        pairs = [(out[i + lag], inp[i]) for i in range(0, n - lag)]
        if len(pairs) < 8:
            continue
        c = float(np.mean([corr(low(o), low(x)) for o, x in pairs]))
        if c > best[0]:
            e = float(np.mean([corr(cv2.GaussianBlur(edges(o), (0, 0), 2), cv2.GaussianBlur(edges(x), (0, 0), 2))
                               for o, x in pairs]))
            best = (c, lag / FPS, e)
    return best


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 15
    res = {}
    threads = [threading.Thread(target=lambda n=n, u=u: res.__setitem__(n, grab(u, seconds)))
               for n, u in (("output", "rtmp://127.0.0.1:1935/joyai-gpu"), ("input", "rtmp://127.0.0.1:1935/shader"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    c, lag, e = adherence(res["output"], res["input"])
    oz, orot, op = motion(res["output"])
    iz, irot, ip = motion(res["input"])
    print(f"structure corr {c:5.3f} (lag {lag:.2f}s)  edge corr {e:5.3f}  | rot out/in {orot:5.1f}/{irot:4.1f} deg/s"
          f"  pan out/in {op:4.1f}/{ip:4.1f} %/s")


if __name__ == "__main__":
    main()
