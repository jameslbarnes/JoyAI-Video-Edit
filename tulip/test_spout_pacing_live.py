"""Real 1440p queue-handoff test. Run with live generation stopped."""
import ctypes
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
from xvideo.serving.spout_output import GpuSpoutPublisher, NativeStats
from xvideo.serving.gpu_video_output import GpuFrame
from xvideo.serving.cadence_metrics import require_native_cadence


class DelayedSubmit:
    """Hold a feeder call before native submission without pausing its clock."""
    def __init__(self, dll):
        self.dll = dll
        self.method = None
        self.entered = threading.Event()
        self.release = threading.Event()

    def __getattr__(self, name):
        return getattr(self.dll, name)

    def _send(self, method, *args):
        if self.method == method:
            self.method = None
            self.entered.set()
            if not self.release.wait(3):
                raise TimeoutError("Test did not release delayed submit")
        return getattr(self.dll, method)(*args)

    def jsp_submit(self, *args):
        return self._send("jsp_submit", *args)

    def jsp_replace(self, *args):
        return self._send("jsp_replace", *args)

    def arm(self, method):
        self.entered.clear()
        self.release.clear()
        self.method = method


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    frame = GpuFrame.record(torch.full((1440, 2592, 3), 180, dtype=torch.uint8, device="cuda"))
    source = np.full((360, 648, 3), 60, dtype=np.uint8)
    p = GpuSpoutPublisher(device="cuda:0", name="ETHEREA Spout pacing test")
    proxy = DelayedSubmit(p.dll)
    p.dll = proxy
    results = {}

    def native():
        s = NativeStats()
        proxy.jsp_stats(p.native, ctypes.byref(s))
        return {k: getattr(s, k) for k, _ in s._fields_}

    def await_condition(predicate, seconds=3):
        until = time.monotonic() + seconds
        while not predicate():
            if time.monotonic() >= until:
                raise AssertionError("Timed out waiting for test condition")
            time.sleep(.002)

    try:
        p.warmup((1440, 2592))
        p.submit([frame] * 32, 24)
        await_condition(lambda: p.native is not None and native()["sent"] >= 4)

        # A CUDA/native submission wait must not block the producer or health.
        proxy.arm("jsp_submit")
        if not proxy.entered.wait(2):
            raise AssertionError("Feeder did not reach delayed call")
        finished = threading.Event()
        def concurrent_producer():
            p.submit([frame] * 12, 24)
            results["statusDuringCopy"] = p.status()
            finished.set()
        producer = threading.Thread(target=concurrent_producer)
        producer.start()
        responsive = finished.wait(.1)
        proxy.release.set()
        producer.join(3)
        if not responsive:
            raise AssertionError("GPU copy held the producer/status lock")

        # Keep old motion available while the replacement is being prepared.
        await_condition(lambda: native()["queued"] >= 14)
        before = native()
        proxy.arm("jsp_replace")
        if not p.begin_transition(lambda: source, moving=True):
            raise AssertionError("Transition refused")
        if not proxy.entered.wait(2):
            raise AssertionError("Transition did not reach replacement")
        time.sleep(.16)
        waiting = native()
        proxy.release.set()
        results["beforeReplacement"] = before
        results["waitingForReplacement"] = waiting
        if waiting["sent"] - before["sent"] < 3:
            raise AssertionError("Old motion stopped before replacement was ready")
        if waiting["underruns"] != before["underruns"]:
            raise AssertionError("Replacement emptied the playback queue")
        await_condition(lambda: native()["discarded"] > before["discarded"])

        # Reset preserves the already queued shader bridge.
        time.sleep(.5)
        reset_before = native()
        if not p.reset_scene():
            raise AssertionError("Scene reset refused")
        time.sleep(.7)
        reset_after = native()
        if reset_after["discarded"] != reset_before["discarded"]:
            raise AssertionError("Reset discarded the moving bridge queue")
        for _ in range(5):
            p.submit([frame] * 16, 24)
            time.sleep(16/24)
        result = p.status()
        results["final"] = result
        n = result["native"]
        results["cadence"] = require_native_cadence(n, 24, max_gap_frames=1.8)
        if result["error"] or n["underruns"] != before["underruns"] or n["max_gap_ms"] > 75:
            raise AssertionError("Presentation stalled during tested handoffs")
        if result["transitions_completed"] != 1:
            raise AssertionError("Transition did not finish")
        # Stop while textures remain queued, then reuse the same publisher.
        # Historical native counters must not prevent creation of its new ring.
        p.submit([frame] * 24, 24)
        await_condition(lambda: native()["queued"] >= 14)
        p.close()
        if p.status()["queued"] != 0:
            raise AssertionError("Closed publisher reports retired textures as queued")
        p.submit([frame] * 32, 24)
        await_condition(lambda: p.native is not None and native()["sent"] >= 12)
        results["afterRestart"] = p.status()
        if results["afterRestart"]["error"]:
            raise AssertionError("Publisher did not recover after stop/resume")
        print(json.dumps({"ok": True, "native": n}))
    finally:
        proxy.release.set()
        p.close()
        (root / "pacing-test.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
