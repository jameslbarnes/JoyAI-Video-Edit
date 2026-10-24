"""Isolated, bounded encoded-frame clock. No Torch, CUDA, or tokenizer imports.

stdin: little-endian uint32 length + payload, first record is the FLV header.
stdout: paced FLV bytes. stderr: compact JSON telemetry every 250 ms.
"""
import collections
import json
import queue
import struct
import sys
import threading
import time


def pace(source, sink, report, fps, capacity=12, prefill=8):
    frames=queue.Queue(maxsize=capacity)
    def read_exact(n):
        result=bytearray()
        while len(result)<n:
            part=source.read(n-len(result))
            if not part:raise EOFError
            result.extend(part)
        return bytes(result)
    def record():
        n=struct.unpack('<I',read_exact(4))[0]
        if n>16*1024*1024:raise ValueError('Encoded frame exceeds bound')
        return read_exact(n)
    sink.write(record());sink.flush()
    failure=[]
    def receive():
        try:
            while True:frames.put(record())
        except EOFError:pass
        except Exception as exc:failure.append(str(exc))
        finally:frames.put(None)
    reader=threading.Thread(target=receive,daemon=True);reader.start()
    while frames.qsize()<prefill and reader.is_alive():time.sleep(.002)
    deadline=time.perf_counter();last=None;sent=underruns=0;starved=False
    last_report=0;gap_count=0;max_gap=0;last_gap=None
    while True:
        delay=deadline-time.perf_counter()
        if delay>0:time.sleep(delay)
        try:payload=frames.get_nowait()
        except queue.Empty:
            if not starved:underruns+=1;starved=True
            payload=frames.get()
        if payload is None:break
        start=time.perf_counter();sink.write(payload);sink.flush();now=time.perf_counter()
        gap=(now-last)*1000 if last is not None else 0
        max_gap=max(max_gap,gap)
        if gap>62.5:
            gap_count+=1;last_gap={'at':time.time(),'gap_ms':gap,'queued':frames.qsize(),'stages':{'pipe':(now-start)*1000},'transition':False}
        last=now;sent+=1;starved=False
        state={'sent':sent,'queued':frames.qsize(),'underruns':underruns,
               'last_send_at':time.time(),'last_send_ms':(now-start)*1000,
               'gap_count':gap_count,'max_gap_ms':max_gap,'last_gap':last_gap}
        if now-last_report>=.25:report(state);last_report=now
        period=1/fps
        deadline=deadline+period if now-deadline<period else now+period
    if sent:report(state)
    if failure:raise RuntimeError(failure[0])


if __name__=='__main__':
    if __package__:
        from .windows_priority import configure_delivery_priority
    else:
        from windows_priority import configure_delivery_priority
    priority = configure_delivery_priority()
    pace(sys.stdin.buffer,sys.stdout.buffer,
         lambda state:print(json.dumps({**state, 'delivery_priority': priority},separators=(',',':')),file=sys.stderr,flush=True),
         float(sys.argv[1]))
