"""Exercise production publisher bursts and scene handoffs with generation stopped."""
import json
from pathlib import Path
import sys
import time
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
from xvideo.serving.spout_output import GpuSpoutPublisher
from xvideo.serving.gpu_video_output import GpuFrame
from xvideo.serving.cadence_metrics import require_native_cadence

root = Path(sys.argv[1])
torch.cuda.set_device(0)
frame = GpuFrame.record(torch.full((1440, 2592, 3), 180, dtype=torch.uint8, device="cuda"))
source = np.full((360, 648, 3), 60, dtype=np.uint8)
publisher = GpuSpoutPublisher(device="cuda:0", name="ETHEREA Spout publisher test")
publisher.warmup((1440, 2592))
samples = []
try:
    for i in range(30):
        start = time.monotonic()
        publisher.submit([frame]*16, 24)
        time.sleep(max(0, 16/24 - (time.monotonic()-start)))
        samples.append(publisher.status())
        if samples[-1]['error']:
            raise RuntimeError(samples[-1]['error'])
    if not publisher.begin_transition(lambda: source):
        raise RuntimeError("Could not begin scene transition")
    time.sleep(1)
    if not publisher.reset_scene():
        raise RuntimeError("Could not reset scene while retaining presentation")
    time.sleep(1)
    for i in range(9):
        publisher.submit([frame]*16, 24)
        time.sleep(16/24)
    samples.append(publisher.status())
    print(json.dumps(samples[-1]))
    last = samples[-1]['native']
    samples[-1]['cadence'] = require_native_cadence(last, 24)
    duration = last['last_send_at'] - last['first_send_at']
    if samples[-1]['error'] or last['sent'] < 500 or (last['sent']-1)/duration > 24.05:
        raise RuntimeError("Publisher failed to sustain presentation")
finally:
    publisher.close()
    (root / "publisher-integration.json").write_text(json.dumps(samples, indent=2))
