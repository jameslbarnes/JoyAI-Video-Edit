"""Deliberately deny the named shader texture for <=1 second; no GPU workload.

Requires explicit live-test authorization and the existing interactive session.
Uses the SDK's existing named texture mutex, never its frame counter. Windows
releases mutex ownership if this process exits; normal release is in finally.
Run only as an approved live-show fault test, followed by a normal cadence gate.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--authorized-live-test', action='store_true')
    parser.add_argument('--source', required=True)
    parser.add_argument('--easel-pid', type=int, required=True)
    parser.add_argument('--model-pid', type=int, required=True)
    parser.add_argument('--milliseconds', type=int, default=800)
    args = parser.parse_args()
    if not args.authorized_live_test or not 600 <= args.milliseconds <= 1000:
        parser.error('Explicit authorization and a bounded 600..1000 ms source interruption are required')
    if not args.source or len(args.source)>200:
        parser.error('An exact existing shader sender is required')
    import psutil
    processes = [psutil.Process(pid) for pid in (args.easel_pid,args.model_pid)]
    if Path(processes[0].exe()).name.lower()!='easel.exe':
        raise RuntimeError('Wrong Easel process')
    if 'serve.py' not in ' '.join(processes[1].cmdline()):
        raise RuntimeError('Wrong JoyAI process')
    created = [process.create_time() for process in processes]
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenMutexW.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_wchar_p]
    kernel.OpenMutexW.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    kernel.WaitForSingleObject.restype=ctypes.c_ulong
    kernel.ReleaseMutex.argtypes=[ctypes.c_void_p]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    def session(pid):
        value=ctypes.c_ulong()
        if not kernel.ProcessIdToSessionId(pid,ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        return value.value
    desktop=session(os.getpid())
    if not desktop or any(session(process.pid)!=desktop for process in processes):
        raise RuntimeError('Use the same interactive session as Easel and JoyAI')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def sample():
        health=json.load(opener.open('http://127.0.0.1:7873/health',timeout=3))
        debug=json.load(opener.open('http://127.0.0.1:7873/debug',timeout=3))
        state=debug.get('session') or {}
        local=health['upscale']['direct_ndi']
        web=health['upscale']['gpu_video']
        return dict(at=time.time(),input=health.get('input_transport') or {},
            local={key:local.get(key) for key in ('sent','error','last_send_at','underruns')},
            firstSendAt=(local.get('native') or {}).get('first_send_at'),
            web={key:web.get(key) for key in ('sent','error','last_send_at','restart_count')},
            sessionAge=state.get('age_s'),chunk=state.get('chunk_idx_next'))
    before=sample()
    inp=before['input']
    if (inp.get('source')!=args.source or inp.get('transport')!='spout' or
        inp.get('fresh_frame_grace_s',0)<2 or (before['sessionAge'] or 0)<10 or
        time.time()-inp.get('last_frame_at',0)>1):
        raise RuntimeError('A healthy active Spout input with the tested grace policy is required')
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'before.json').write_text(json.dumps(before,indent=2))
    # Open, never create. This exact SDK name was inspected in SpoutFrameCount.
    handle=kernel.OpenMutexW(0x00100000|0x0001,False,args.source+'_SpoutAccessMutex')
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    owned=False
    held=None
    try:
        result=kernel.WaitForSingleObject(handle,1000)
        if result in (0,0x80):
            owned=True
        if result!=0:
            raise RuntimeError(f'Expected a healthy available shader mutex, result={result}')
        begin=time.perf_counter()
        time.sleep(args.milliseconds/1000)
        held=(time.perf_counter()-begin)*1000
    finally:
        released = not owned or kernel.ReleaseMutex(handle)
        error = ctypes.get_last_error()
        kernel.CloseHandle(handle)
        if not released:
            raise ctypes.WinError(error)
    samples=[]
    for _ in range(10):
        time.sleep(.5)
        samples.append(sample())
    after=samples[-1]
    failures=[]
    if any(not process.is_running() or process.create_time()!=birth
           for process,birth in zip(processes,created)):
        failures.append('Application or model process changed')
    if after['firstSendAt']!=before['firstSendAt']:
        failures.append('Publisher was replaced')
    if after['web']['restart_count']!=before['web']['restart_count']:
        failures.append('Web encoder was restarted')
    if after['sessionAge'] is None or after['sessionAge']<before['sessionAge']:
        failures.append('Generation session was reset')
    if after['input'].get('fresh_wait_recoveries',0)<=inp.get('fresh_wait_recoveries',0):
        failures.append('The intended short-timeout recovery was not exercised')
    if after['input'].get('frames',0)<=inp.get('frames',0):
        failures.append('Fresh shader input did not resume')
    for name in ('local','web'):
        if after[name]['error'] or after['at']-(after[name]['last_send_at'] or 0)>1:
            failures.append(name+' output did not resume')
    report=dict(passed=not failures,failures=failures,heldMs=held,
        source=args.source,easelPid=args.easel_pid,modelPid=args.model_pid,
        before=before,after=after,samples=samples,
        scope='Intentional short shader interruption. This tests scene/session survival, not normal display cadence; run the normal live gate separately.')
    (out/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({key:report[key] for key in ('passed','failures','heldMs')}),flush=True)
    return 0 if report['passed'] else 2


if __name__=='__main__':
    sys.exit(main())
