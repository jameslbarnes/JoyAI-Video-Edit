"""Actual GPU handoff through production SR, NDI pixels and JPEG preparation.

Local collectors replace network publishers; no live settings are written.
"""
import argparse
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy'))
os.environ['JOYAI_RIFE_FLUX_ROOT'] = str(ROOT.parent / 'FluxRT')
import numpy as np
import torch
from PIL import Image
from xvideo.serving.rife_output import RifeOutput
from xvideo.serving.upscale_output import OutputUpscale
from xvideo.serving.vsr_output import VsrUpscaler

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('report', type=Path)
args = parser.parse_args()
torch.set_num_threads(2)
inputs = np.load(ROOT / 'tulip/evidence/dlssg-rife-20260923/inputs.npy', mmap_mode='r')
rife = RifeOutput('cuda:0')
vsr = VsrUpscaler('cuda:0')
output = OutputUpscale('cuda:0', denoise=.25)
published, ndi = [], []
output.model = vsr
output.models = {'vsr_ultra': vsr}
output.backend = 'vsr_ultra'
output.direct_publisher = SimpleNamespace(error=None, submit=lambda frames, fps: ndi.extend(frames),
    status=lambda: {'error': None}, close=lambda: None)
output.gpu_publisher = SimpleNamespace(error=None, submit=lambda frames, fps: published.extend(frames),
    status=lambda: {'error': None}, close=lambda: None)
output._configure_gpu_publisher = lambda: None
output.neural_rendering = SimpleNamespace(_settings=lambda: {'enabled': False},
    status=lambda: {'enabled': False}, process=lambda frames: frames, close=lambda: None)
downloads = []
original_cpu = torch.Tensor.cpu
def checked_cpu(tensor, *a, **kw):
    downloads.append(list(tensor.shape))
    assert tuple(tensor.shape[:2]) == (1080, 1944), 'Unexpected raw inter-stage download'
    return original_cpu(tensor, *a, **kw)

try:
    with tempfile.TemporaryDirectory() as directory:
        output.output_config = Path(directory) / 'output.json'
        config = {'height': 1080, 'denoise': .25, 'backend': 'vsr_ultra',
                  'direct_ndi': True, 'progressive': True, 'rife_gpu': True,
                  'gpu_video': {'enabled': True, 'height': 1440, 'bitrate': 15000000,
                                'url': 'rtmp://127.0.0.1:1935/test-unused', 'ffmpeg': 'unused'}}
        output.output_config.write_text(json.dumps(config))
        output._refresh_output_size()
        assert output.gpu_interpolation_enabled(rife.device)
        assert not output.gpu_interpolation_enabled('cuda:1')
        profile = {'presentation_fps': 24}
        with patch.object(torch.Tensor, 'cpu', checked_cpu):
            frames, metas = rife.process(list(inputs[:8]), [{'seq': i} for i in range(8)],
                profile, raw_output=True, gpu_output=True)
            previews = output.process(frames, profile)
        assert len(previews) == len(ndi) == len(published) == len(metas) == 15
        assert profile['rife_gpu_handoff'] == {'gpu_input': True, 'input_download_bytes': 0}
        assert len(downloads) == 15
        assert all(f.shape == (1440, 2592, 3) for f in published)
        assert all(Image.open(io.BytesIO(f)).size == (1296, 720) for f in previews)
        # Same GPU versus NumPy source through the full presentation branch.
        saved = published[0].wait().clone()
        expected_ndi = ndi[0].copy()
        expected_preview = previews[0]
        published.clear(); ndi.clear()
        again = output.process([inputs[0]], {'presentation_fps': 24})
        assert torch.equal(saved, published[0].wait())
        assert np.array_equal(expected_ndi, ndi[0])
        assert expected_preview == again[0]
        # Manual off toggle drains already-prepared GPU frames without a restart.
        config['rife_gpu'] = False
        output.output_config.write_text(json.dumps(config))
        published.clear(); ndi.clear()
        output.process(frames[:1], {'presentation_fps': 24})
        assert not output.gpu_interpolation_enabled(rife.device)
        assert len(published) == len(ndi) == 1
        # Consumer contract change between prepare and consume is explicit.
        profile = {}
        raw = output.process(frames[:1], profile, raw_output=True)
        assert profile['rife_gpu_handoff']['input_download_bytes'] == 1152*648*3
        assert raw[0].shape == (1080, 1944, 3)
        report = {'ok': True, 'frames': 15, 'interStageDownloads': 0,
                  'previewNdiDownloads': len(downloads), 'exactGpuNdiJpegAgreement': True,
                  'manualToggleDrainsInFlight': True, 'consumerChangeFallback': True}
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
finally:
    output.close()
