"""Read-only live artifact recorder: samples JoyAI output + shader input,
computes drift_lab metrics, and logs them with the runtime state.

Usage: python tulip/drift_lab/live_probe.py --out DIR [--seconds N] [--fps 2]
Writes DIR/samples.jsonl and DIR/thumbs/<t>.jpg (output | input) every 10 s.
Never sends controls; safe during shows.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import frame_metrics  # noqa: E402

FFMPEG = "C:/Users/Admin/src/easel/external/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"
W, H = 480, 270


class Reader:
    """Latest frame of an RTMP stream, decoded at a fixed rate into RGB."""

    def __init__(self, url, fps):
        self.frame, self.at, self.lock = None, 0.0, threading.Lock()
        self.proc = subprocess.Popen(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-fflags", "nobuffer", "-i", url,
             "-vf", f"fps={fps},scale={W}:{H}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            stdout=subprocess.PIPE)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        size = W * H * 3
        while True:
            buf = self.proc.stdout.read(size)
            if len(buf) < size:
                return
            with self.lock:
                self.frame, self.at = np.frombuffer(buf, np.uint8).reshape(H, W, 3), time.time()

    def latest(self):
        with self.lock:
            return self.frame, self.at

    def close(self):
        self.proc.kill()


def runtime_state():
    try:
        d = json.load(urllib.request.urlopen("http://127.0.0.1:7873/debug", timeout=3))
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    s = d.get("session") or {}
    p = s.get("last_chunk_profile") or {}
    st = p.get("shader_strength") or {}
    c = s.get("cache") or {}
    pc = s.get("prompt_control") or {}
    return {"sceneAge": s.get("age_s"), "chunk": s.get("chunk_idx_next"), "steps": p.get("steps"),
            "temporalPosition": p.get("temporal_position"), "strength": st.get("value"),
            "referenceGain": st.get("referenceGain"), "blend": st.get("blend"),
            "decay": (c.get("sinkDecay") or {}).get("half_life_s"),
            "allocatedMiB": c.get("allocatedMiB"), "promptApplied": pc.get("applied"),
            "prompt": (s.get("raw_prompt") or "")[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=0, help="0 = run until killed")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--output-url", default="rtmp://127.0.0.1:1935/joyai-gpu")
    ap.add_argument("--input-url", default="rtmp://127.0.0.1:1935/shader")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    o, i = Reader(args.output_url, args.fps), Reader(args.input_url, args.fps)
    inputs = deque(maxlen=int(4 * args.fps) + 1)  # ~4 s of input history for lag search
    prev_o = prev_i = None
    state, state_at, last_thumb = {}, 0.0, 0.0
    started = time.time()
    with (out / "samples.jsonl").open("a", encoding="utf-8") as log:
        while not args.seconds or time.time() - started < args.seconds:
            time.sleep(1 / args.fps)
            fo, at_o = o.latest()
            fi, at_i = i.latest()
            if fo is None or fi is None:
                continue
            now = time.time()
            if now - state_at > 2:
                state, state_at = runtime_state(), now
            m = frame_metrics(fo, fi, prev_o, prev_i, list(inputs)[::-1])
            m.update(t=round(now, 2), outputAge=round(now - at_o, 2), inputAge=round(now - at_i, 2), **state)
            log.write(json.dumps(m) + "\n")
            log.flush()
            if now - last_thumb >= 10:
                cv2.imwrite(str(out / "thumbs" / f"{int(now)}.jpg"),
                            cv2.cvtColor(np.hstack([fo, fi]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                last_thumb = now
            prev_o, prev_i = fo, fi
            inputs.append(fi)
    o.close()
    i.close()


if __name__ == "__main__":
    main()
