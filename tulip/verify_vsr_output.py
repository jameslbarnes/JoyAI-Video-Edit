"""GPU integration check: explicit second device, owned buffers and size changes."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'deploy'))
from xvideo.serving.vsr_output import VsrUpscaler
model = VsrUpscaler('cuda:1')
try:
    frame = np.asarray(Image.open(root/'tulip/source-frame.png').convert('RGB').resize((1152,648))).copy()
    first = model.process_gpu(frame, (1440, 2592))
    with torch.cuda.device(model.device), torch.cuda.stream(model.stream):
        saved = first.wait().clone()
    durations = []
    for _ in range(12):
        start = time.perf_counter()
        other = model.process_gpu(255-frame, (1440, 2592))
        model.stream.synchronize()
        durations.append((time.perf_counter()-start)*1000)
        time.sleep(.08)
    with torch.cuda.device(model.device), torch.cuda.stream(model.stream):
        assert torch.equal(first.wait(), saved), 'SDK overwrote retained output'
        assert not torch.equal(other.wait(), saved), 'Different inputs produced identical output'
        assert other.tensor.device.index == 1
        assert other.tensor.dtype == torch.uint8
        assert saved.float().std() > 1, 'Initial output is black'
    small = model.process(frame, (1080, 1944))
    large = model.process(frame, (1440, 2592))
    assert small.shape == (1080, 1944, 3) and large.shape == (1440, 2592, 3)
    assert large.std() > 1, (float(small.std()), float(large.std()), float(large.mean()))
    burst = [model.process_gpu(frame if i % 2 == 0 else 255-frame, (1440,2592)) for i in range(8)]
    model.stream.synchronize()
    with torch.cuda.device(model.device), torch.cuda.stream(model.stream):
        reference = torch.from_numpy(large).to(model.device).float()
        for i, result in enumerate(burst):
            difference = (result.wait().float()-burst[i % 2].wait().float()).abs()
            assert torch.equal(result.wait(), burst[i % 2].wait()), ('Burst mismatch', i, float(difference.mean()))
            if i % 2 == 0:
                assert torch.equal(result.wait().float(), reference), 'Frame/input correspondence mismatch'
    print(json.dumps({'ok': True, 'device': str(model.device), 'medianMs': float(np.median(durations)),
        'ownedOutput': True, 'sizeChanges': True}))
finally:
    model.close()
