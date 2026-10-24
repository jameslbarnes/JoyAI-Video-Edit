"""Run only while live generation is paused; compare the pinned VSR implementations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
from xvideo.serving.direct_vsr import DirectVsrUpscaler
from xvideo.serving.gpu_video_output import GpuFrame
from xvideo.serving.vsr_output import LegacyVsrUpscaler
from xvideo.serving.upscale_output import OutputUpscale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quality", type=int, default=4)
    parser.add_argument("--compare-legacy", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    frame = np.asarray(Image.open(ROOT / "tulip/source-frame.png").convert("RGB").resize((1152, 648))).copy()
    report = {"device": str(device), "quality": args.quality}
    legacy_output = None
    if args.compare_legacy:
        legacy = LegacyVsrUpscaler(device)
        samples = []
        try:
            legacy_input = torch.from_numpy(frame).to(device)
            for i in range(16):
                t = time.perf_counter()
                result = legacy.process_gpu(legacy_input, (1440, 2592))
                result.ready.synchronize()
                if i >= 4:
                    samples.append((time.perf_counter() - t) * 1000)
            legacy_output = result.tensor.cpu().numpy()
            report["legacyMs"] = samples
            Image.fromarray(legacy_output).save(args.output / "legacy.png")
        finally:
            legacy.close()
            del legacy
            torch.cuda.empty_cache()
    model = DirectVsrUpscaler(device, args.quality)
    try:
        first = model.process_gpu(frame, (1440, 2592))
        first.ready.synchronize()
        saved = first.tensor.cpu().numpy().copy()
        assert saved.std() > 1, "Black or empty first frame"
        samples = []
        # Bursts alternate strongly different images, including a separate producer stream.
        producer = torch.cuda.Stream(device=device)
        for i in range(48):
            source = frame if i % 2 == 0 else 255 - frame
            with torch.cuda.stream(producer):
                gpu_input = GpuFrame.record(torch.from_numpy(source.copy()).to(device))
            t = time.perf_counter()
            result = model.process_gpu(gpu_input, (1440, 2592))
            result.ready.synchronize()
            elapsed = (time.perf_counter() - t) * 1000
            output = result.tensor.cpu().numpy()
            if i >= 4:
                samples.append(elapsed)
            if i % 2 == 0:
                assert np.max(np.abs(output.astype(int) - saved)) <= 1, ("Stale/mismatched frame", i)
            else:
                assert np.abs(output.astype(float) - saved).mean() > 5, ("Repeated frame", i)
        assert np.array_equal(first.tensor.cpu().numpy(), saved), "Retained frame was overwritten"
        burst = [model.process_gpu(frame if i % 2 == 0 else 255-frame, (1440, 2592)) for i in range(8)]
        burst[-1].ready.synchronize()
        for i, result in enumerate(burst):
            pixels = result.tensor.cpu().numpy()
            difference = np.abs(pixels.astype(int) - saved)
            assert difference.max() <= 1 if i % 2 == 0 else difference.mean() > 5, ("Burst order", i)
        del burst
        # Reconfiguration must finish effect work without altering retained frames.
        small = model.process(frame, (1080, 1944))
        large = model.process(frame, (1440, 2592))
        assert small.shape == (1080, 1944, 3) and large.shape == saved.shape
        assert np.max(np.abs(large.astype(int) - saved)) <= 1
        # Both work submission and output ordering are checked under unrelated CUDA work.
        unrelated = torch.cuda.Stream(device=device)
        with torch.cuda.stream(unrelated):
            torch.cuda._sleep(300_000_000)
            done = torch.cuda.Event()
            done.record(unrelated)
        start = time.perf_counter()
        output = model.process_gpu(frame, (1440, 2592))
        submit_ms = (time.perf_counter() - start) * 1000
        unrelated_finished_at_return = done.query()
        output.ready.synchronize()
        unrelated.synchronize()
        assert np.max(np.abs(output.tensor.cpu().numpy().astype(int) - saved)) <= 1
        assert not unrelated_finished_at_return, "Submission waited for unrelated GPU work"
        report.update(ok=True, nativeMs=samples, nativeMedianMs=statistics.median(samples),
                      submitWithUnrelatedWorkMs=submit_ms,
                      unrelatedFinishedAtReturn=unrelated_finished_at_return,
                      ownedOutput=True, crossStreamInput=True, sizeChanges=True, status=model.status())
        if legacy_output is not None:
            difference = np.abs(large.astype(int) - legacy_output.astype(int))
            report["comparison"] = {"meanAbsoluteError": float(difference.mean()),
                                    "maxAbsoluteError": int(difference.max())}
        # The operator restored serial GPU preview preparation. Only CPU JPEG
        # encoding runs on workers; its chunk-boundary wait is intentional in
        # this rollback. Do not claim independent cross-chunk preview delivery.
        output = OutputUpscale.__new__(OutputUpscale)
        output.enabled, output.error = True, None
        output.device, output.output_size = str(device), (1440, 2592)
        output.direct_requested, output.gpu_publisher = True, None
        output.model = model
        output.neural_rendering = SimpleNamespace(status=lambda: {"enabled": False})
        submitted, published, release = [], threading.Event(), threading.Event()
        def submit(frames, fps):
            assert fps == 24
            submitted.extend(frames)
            if len(submitted) == 8:
                published.set()
        output.direct_publisher = SimpleNamespace(gpu_input=True, available=lambda: True,
            submit=submit, status=lambda: {"error": None, "sent": len(submitted)})
        encode = output.encode_preview
        def slow_preview(frame, quality):
            assert isinstance(frame, np.ndarray), "GPU preview work escaped onto a JPEG worker"
            assert release.wait(15), "GPU publishing was blocked by preview work"
            return encode(frame, quality)
        output.encode_preview = slow_preview
        with ThreadPoolExecutor(max_workers=4) as output.encoder, ThreadPoolExecutor(max_workers=1) as producer:
            job = producer.submit(output._process_progressive,
                [frame if i % 2 == 0 else 255-frame for i in range(8)],
                {"presentation_fps": 24}, 95, time.perf_counter(), defer_preview=True)
            try:
                assert published.wait(10), "Local delivery waited for JPEG workers"
                submitted[-1].ready.synchronize()
                assert not job.done(), "Fault injection did not actually block previews"
            finally:
                release.set()
            previews = job.result(30)
            second_previews = output._process_progressive(
                [frame if i % 2 == 0 else 255-frame for i in range(8)],
                {"presentation_fps": 24}, 95, time.perf_counter(), defer_preview=True)
        assert len(submitted) == 16
        assert output.last["preview_deferred"] is False
        assert len(previews) == 8 and all(isinstance(x, bytes) and x.startswith(b"\xff\xd8") for x in previews)
        assert previews[0] == previews[2] and previews[1] == previews[3] and previews[0] != previews[1]
        assert second_previews == previews
        report["preview"] = {"publishedWhileJpegBlocked": 8, "serialGpuPreparation": True,
                             "independentAcrossChunks": False,
                             "orderedJpegs": len(previews), "bytes": [len(x) for x in previews]}
        Image.fromarray(large).save(args.output / f"native-mode-{args.quality}.png")
        model.close()
        model.close()  # Repeated teardown must be harmless.
        report["cleanTeardown"] = True
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        model.close()


if __name__ == "__main__":
    main()
