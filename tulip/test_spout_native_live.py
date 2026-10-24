"""Integration check. Stop live generation before running this GPU test."""
import ctypes
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--receiver', type=Path, default=Path(__file__).resolve().parents[2] /
                        'easel-installation/build/Release/test_spout_receive.exe')
    parser.add_argument('--generation-paused', action='store_true')
    args = parser.parse_args()
    if not args.generation_paused:
        parser.error('This GPU fixture requires an authorized pause')
    if not args.receiver.is_file():
        parser.error('Build test_spout_receive from the current Easel checkout, or pass --receiver EXE')
    from test_vsr_release_live import check_model_idle
    check_model_idle('http://127.0.0.1:7873/debug')
    import torch
    from xvideo.serving.spout_output import library, NativeStats, NativeCopyTrace, NativePresentSample
    evidence = args.output
    evidence.mkdir(parents=True, exist_ok=False)
    dll = library(args.library)
    torch.cuda.set_device(0)
    w, h = 2592, 1440
    pixels = torch.empty((h, w, 4), device="cuda", dtype=torch.uint8)
    event = torch.cuda.Event()
    name = "ETHEREA Spout ownership test"
    sender = dll.jsp_create(name.encode(), w, h, 24., 18, 12)
    if not sender:
        raise RuntimeError(dll.jsp_error(None).decode())
    log = (evidence / "native-receiver.log").open("w")
    receiver = subprocess.Popen([
        str(args.receiver),
        name, "32", str(evidence / "native-receiver.json"), "pattern"],
        stdout=log, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    count, samples, copy_samples, priority_changes = 0, [], [], []
    priority = 0
    try:
        started = time.monotonic()
        while time.monotonic() - started < 34:
            s = NativeStats()
            dll.jsp_stats(sender, ctypes.byref(s))
            if s.queued >= 16:
                time.sleep(.002)
                continue
            count += 1
            if count in (240, 480, 720):
                priority = -1 if priority == 0 else 0
                if dll.jsp_set_copy_priority(sender, priority) != 1:
                    raise RuntimeError(dll.jsp_error(None).decode())
                priority_changes.append(dict(frame=count, priority=priority))
            pixels[:h//2, :, 0].fill_(220)
            pixels[h//2:, :, 0].fill_(10)
            pixels[:h//2, :, 2].fill_(10)
            pixels[h//2:, :, 2].fill_(220)
            pixels[:, :, 1].fill_(count % 256)
            pixels[:, :, 3].fill_(255)
            event.record()
            if dll.jsp_submit(sender, pixels.data_ptr(), event.cuda_event, count) != 1:
                raise RuntimeError(dll.jsp_error(sender).decode())
            trace = NativeCopyTrace()
            if dll.jsp_copy_trace(sender, ctypes.byref(trace), ctypes.sizeof(trace)) != 1:
                raise RuntimeError('Native copy diagnostics ABI mismatch')
            if trace.token != count or trace.priority != priority or trace.gpu_span_ms < 0:
                raise RuntimeError('Copy trace does not match the completed submission')
            # Poison the input immediately: queued textures must retain their copy.
            pixels.fill_(127)
            if count % 24 == 0:
                samples.append({k: getattr(s, k) for k, _ in s._fields_})
                copy_samples.append({k: getattr(trace, k) for k, _ in trace._fields_})
            if count % 120 == 0:
                check_model_idle('http://127.0.0.1:7873/debug')
                # Hold Python's GIL; native presentation must keep its own clock.
                until = time.perf_counter() + .15
                while time.perf_counter() < until:
                    pass
        dll.jsp_stats(sender, ctypes.byref(s))
        result = {k: getattr(s, k) for k, _ in s._fields_}
        result["error"] = dll.jsp_error(sender).decode()
        from xvideo.serving.cadence_metrics import native_sender_metrics, require_native_cadence
        result["cadence"] = native_sender_metrics(result, 24)
        result["samples"] = samples
        result['copy_samples'] = copy_samples
        result['priority_changes'] = priority_changes
        (evidence / "native-sender.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ("samples", "copy_samples")}))
        receiver.wait(timeout=5)
        require_native_cadence(result, 24)
        if receiver.returncode or result["error"] or result["access_misses"]:
            raise RuntimeError("Spout integration test failed; inspect receiver and sender evidence")
        # A separate intentional supply outage validates diagnostics. It must
        # not be mixed into the steady ownership/cadence verdict above.
        time.sleep(1)
        count += 1
        event.record()
        if dll.jsp_submit(sender, pixels.data_ptr(), event.cuda_event, count) != 1:
            raise RuntimeError(dll.jsp_error(sender).decode())
        until = time.monotonic()+2
        while time.monotonic()<until:
            dll.jsp_stats(sender, ctypes.byref(s))
            if s.last_token==count:
                break
            time.sleep(.005)
        events = (NativePresentSample*64)()
        events[0].serial=999
        if dll.jsp_present_trace(sender,events,64,ctypes.sizeof(NativePresentSample)-8)!=-1 or events[0].serial!=999:
            raise RuntimeError('Mismatched presentation ABI wrote to caller memory')
        size=dll.jsp_present_trace(sender,events,64,ctypes.sizeof(NativePresentSample))
        if not 1<=size<=64:
            raise RuntimeError('Expected bounded native starvation evidence')
        rows=[{k:getattr(e,k) for k,_ in e._fields_} for e in events[:size]]
        (evidence/'intentional-starvation.json').write_text(json.dumps(rows,indent=2))
        last=rows[-1]
        if last['token']!=count or last['empty_wait_ms']<200 or last['gap_ms']<200 or not last['advanced']:
            raise RuntimeError('The diagnostic did not attribute the intentional supply outage')
    finally:
        dll.jsp_close(sender)
        if receiver.poll() is None:
            receiver.terminate()
            receiver.wait(timeout=5)
        log.close()


if __name__ == "__main__":
    main()
