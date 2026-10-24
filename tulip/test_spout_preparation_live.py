"""Paused-show fixture for the production Spout feeder and optional GPU spans.

Unique source, no video recording. Never pauses/restarts the show itself.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--generation-paused', action='store_true')
    args = parser.parse_args()
    if not args.generation_paused:
        parser.error('Requires an authorized pause')
    from test_vsr_release_live import check_model_idle
    check_model_idle('http://127.0.0.1:7873/debug')
    import numpy as np
    import torch
    from xvideo.serving.gpu_video_output import GpuFrame
    from xvideo.serving.spout_output import GpuSpoutPublisher
    from xvideo.serving.cadence_metrics import require_native_cadence

    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    p = GpuSpoutPublisher('cuda:0', name='ETHEREA private preparation test')
    p.configure_transition_reserve(3)
    p.configure_preparation_profile(True)
    p.feeder_trace.threshold_ms = 0  # Fixture only; production retains slow calls.
    p.warmup((1440, 2592))
    source = np.full((360, 648, 3), 60, np.uint8)
    frame = GpuFrame.record(torch.full((1440, 2592, 3), 180, device='cuda', dtype=torch.uint8))
    spans, report = {}, dict(ok=False, producer=[])
    # An absolute clock includes diagnostic/HTTP overhead in the batch period.
    # Sleeping 16/fps anew after each request slowly undersupplied the queue.
    clock = time.monotonic()
    try:
        for i in range(36):
            check_model_idle('http://127.0.0.1:7873/debug')
            start = time.monotonic()
            if i == 10:
                if not p.begin_transition(lambda: source, moving=True) or not p.reset_scene():
                    raise AssertionError('Scene handoff did not begin')
            if i == 22:
                p.configure_transition_reserve(2)
            p.submit([frame] * 16, 24)
            report['producer'].append(dict(batch=i, at=time.monotonic(),
                scheduled=clock+i*16/24, lateMs=(start-clock-i*16/24)*1000))
            time.sleep(max(0, clock+(i+1)*16/24-time.monotonic()))
            state = p.status()
            if state['error']:
                raise RuntimeError(state['error'])
            for event in state['feeder']['events']:
                details = event.get('details', {})
                if 'prepare_gpu_span_ms' in details:
                    spans[event['serial']] = event
                    a, b, total = (details[k] for k in
                        ('select_gpu_span_ms', 'pack_gpu_span_ms', 'prepare_gpu_span_ms'))
                    if min(a, b, total) < 0 or abs(a + b - total) > .01:
                        raise AssertionError('Invalid/reused GPU timing event')
        # Keep timings even when the cadence assertion fails. A failed run is
        # the evidence needed to diagnose queue starvation, not disposable data.
        report.update(state=state, spans=list(spans.values()))
        report['cadence'] = require_native_cadence(state['native'], 24)
        if len(spans) < 300 or not any(e['transition'] for e in spans.values()):
            raise AssertionError('Incomplete timing/transition coverage')
        p.configure_preparation_profile(False)
        if p.status()['profile_preparation']:
            raise AssertionError('Could not disable profiling')
        report.update(ok=True, state=state, spans=list(spans.values()),
                      scope='Private feeder timing/ownership; requires subsequent real-show acceptance')
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        report.update(state=p.status(), spans=list(spans.values()))
        p.close()
        (args.output / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'ok': report['ok'], 'spans': len(spans), 'cadence': report['cadence']}))


if __name__ == '__main__':
    main()
