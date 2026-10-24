"""Paused-show VSR compatibility test; separate process per release.

Never stops/restarts a service. Refuses a live JoyAI session before importing
Torch, and rechecks between batches. GPU tests need the operator's approval
to pause the show. Completed-work timing is not a projector cadence result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy'))
from xvideo.serving.vfx_runtime import VsrSettings


def require_idle(debug):
    if not isinstance(debug, dict) or debug.get('ok') is not True or 'session' not in debug:
        raise RuntimeError('Cannot establish the live model state; no GPU test started')
    if debug['session'] is not None:
        raise RuntimeError('Live JoyAI generation is active; pause only with operator approval')


def check_model_idle(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as response:
        require_idle(json.load(response))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--release', required=True, choices=['0.1.0.1', '0.2.0.0'])
    parser.add_argument('--quality', type=int, default=4)
    parser.add_argument('--strength', type=float)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--input', type=Path, required=True,
                        help='Same actual 1152x648 native RGB frame for each release/mode')
    parser.add_argument('--reference', type=Path, help='Previous run directory for pixel comparison')
    parser.add_argument('--debug-url', default='http://127.0.0.1:7873/debug')
    parser.add_argument('--generation-paused', action='store_true',
                        help='Operator approved interruption; script verifies the session is idle')
    args = parser.parse_args()
    if not args.generation_paused:
        parser.error('GPU tests require --generation-paused after operator approval')
    settings = VsrSettings(args.release, args.quality, args.strength)
    check_model_idle(args.debug_url)
    import numpy as np
    from PIL import Image
    import torch
    from xvideo.serving.vsr_output import VsrUpscaler
    from xvideo.serving.gpu_video_output import GpuFrame
    from xvideo.serving.cadence_metrics import interval_metrics

    args.output.mkdir(parents=True, exist_ok=False)
    frame = np.asarray(Image.open(args.input).convert('RGB')).copy()
    if frame.shape != (648, 1152, 3):
        raise ValueError('Use an actual 1152x648 native frame; this test does not resize input')
    input_hash = hashlib.sha256(frame.tobytes()).hexdigest()
    device = torch.device(args.device)
    report = {'ok': False, 'release': args.release, 'quality': args.quality,
              'inputSha256': input_hash, 'device': str(device),
              'timingScope': 'Completed upscaling work in a paused-show fixture; not display cadence',
              'requiresLiveProjectorGate': True}
    model = None
    try:
        torch.cuda.set_device(device)
        model = VsrUpscaler(device, settings=settings)
        originals = (frame, 255-frame)
        baselines = [model.process(x, (1440, 2592)).copy() for x in originals]
        assert all(x.std() > 1 for x in baselines), 'Black or empty output'
        assert np.abs(baselines[0].astype(float)-baselines[1]).mean() > 5, 'Stale/inverted input mismatch'
        first = model.process_gpu(frame, (1440, 2592))
        first.ready.synchronize()
        retained = first.tensor.cpu().numpy().copy()
        producer = torch.cuda.Stream(device=device)
        times = []
        for i in range(64):
            if i % 8 == 0:
                check_model_idle(args.debug_url)
            # Alternating cross-stream inputs catch stale conversion and reuse.
            with torch.cuda.stream(producer):
                source = GpuFrame.record(torch.from_numpy(originals[i % 2].copy()).to(device))
            start = time.perf_counter()
            result = model.process_gpu(source, (1440, 2592))
            result.ready.synchronize()
            if i >= 8:
                times.append((time.perf_counter()-start)*1000)
            pixels = result.tensor.cpu().numpy()
            diff = np.abs(pixels.astype('int16')-baselines[i % 2].astype('int16'))
            if diff.max() > 1:
                report['mismatch'] = {'iteration': i, 'max': int(diff.max()),
                                      'mean': float(diff.mean())}
                Image.fromarray(pixels).save(args.output/'mismatch.png')
                Image.fromarray(baselines[i % 2]).save(args.output/'expected.png')
            assert diff.max() <= 1, 'Wrong/stale frame'
        assert np.array_equal(first.tensor.cpu().numpy(), retained), 'Retained output was overwritten'
        check_model_idle(args.debug_url)
        small = model.process(frame, (1080, 1944))
        large = model.process(frame, (1440, 2592))
        assert small.shape == (1080, 1944, 3) and large.shape == baselines[0].shape
        assert np.abs(large.astype(int)-baselines[0].astype(int)).max() <= 1, 'Resize return changed output'
        assert np.array_equal(first.tensor.cpu().numpy(), retained), 'Resize altered owned output'
        report.update(status=model.status(), completedFrameMs=interval_metrics(times, 1000/24),
                      alternatingFrames=True, ownedOutput=True, crossStreamInput=True, geometryChanges=True)
        Image.fromarray(frame).save(args.output/'input.png')
        Image.fromarray(large).save(args.output/'output.png')
        if args.reference:
            old_report = json.loads((args.reference/'result.json').read_text())
            if old_report['inputSha256'] != input_hash:
                raise ValueError('Reference used a different native input')
            old = np.asarray(Image.open(args.reference/'output.png').convert('RGB'))
            diff = np.abs(large.astype(int)-old.astype(int))
            report['comparison'] = {'meanAbsoluteError': float(diff.mean()),
                                    'maxAbsoluteError': int(diff.max()),
                                    'note': 'Pixel difference is not a perceptual-quality score'}
        model.close()
        model = None
        report.update(ok=True, cleanTeardown=True)
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        try:
            if model is not None:
                model.close()
        finally:
            (args.output/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
