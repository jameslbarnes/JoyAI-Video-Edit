"""Global camera motion of the live output vs the shader input (read-only).

Usage: python tulip/drift_lab/motion_probe.py [seconds=20]
Samples both RTMP streams at 4 fps, tracks features, fits a similarity
transform per frame pair and prints median |zoom| (%/s), |rotation| (deg/s)
and pan (% of width/s). Output spinning far more than the input = the model
is inventing camera motion instead of following the shader.
"""
import subprocess
import sys
import threading

import cv2
import numpy as np

FFMPEG = "C:/Users/Admin/src/easel/external/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"
W, H, FPS = 480, 270, 4


def grab(url, seconds):
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", url, "-t", str(seconds + 1),
                        "-vf", f"fps={FPS},scale={W}:{H}", "-pix_fmt", "gray", "-f", "rawvideo", "-"],
                       capture_output=True, timeout=seconds + 60)
    a = np.frombuffer(p.stdout, np.uint8)
    k = len(a) // (W * H)
    return a[:k * W * H].reshape(k, H, W)


def motion(frames):
    zs, rs, ps = [], [], []
    for a, b in zip(frames[:-1], frames[1:]):
        p0 = cv2.goodFeaturesToTrack(a, 400, 0.01, 8)
        if p0 is None or len(p0) < 20:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None)
        ok = st.ravel() == 1
        if ok.sum() < 20:
            continue
        m, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC)
        if m is None:
            continue
        zs.append(abs(np.log(np.hypot(m[0, 0], m[1, 0]))) * FPS * 100)
        rs.append(abs(np.degrees(np.arctan2(m[1, 0], m[0, 0]))) * FPS)
        ps.append(np.hypot(m[0, 2], m[1, 2]) * FPS / W * 100)
    return [float(np.median(v)) if v else float("nan") for v in (zs, rs, ps)]


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20
    res = {}
    threads = [threading.Thread(target=lambda n=n, u=u: res.__setitem__(n, grab(u, seconds)))
               for n, u in (("output", "rtmp://127.0.0.1:1935/joyai-gpu"), ("input", "rtmp://127.0.0.1:1935/shader"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for name in ("input", "output"):
        z, r, p = motion(res[name])
        print(f"{name:6s} zoom {z:5.2f} %/s  rotation {r:5.2f} deg/s  pan {p:5.2f} %width/s")


if __name__ == "__main__":
    main()
