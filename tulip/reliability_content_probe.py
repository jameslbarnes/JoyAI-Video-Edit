import subprocess,time,json,sys
from pathlib import Path
import numpy as np
out=Path('tulip/reliability-content-20260915.jsonl')
cmd=[r'C:/Users/Admin/src/easel/external/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe','-hide_banner','-loglevel','error','-i','rtmp://127.0.0.1:1935/hermes-screen','-t','480','-an','-vf','scale=64:36','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL);n=64*36*3;last=None;index=0;run=best=0;started=time.monotonic()
try:
 with out.open('w') as f:
  while True:
   raw=bytearray()
   while len(raw)<n:
    b=p.stdout.read(n-len(raw))
    if not b:break
    raw.extend(b)
   if len(raw)!=n:break
   a=np.frombuffer(raw,np.uint8).astype(np.int16)
   diff=float(np.mean(np.abs(a-last))) if last is not None else None
   if diff is not None:
    run=run+1 if diff<.1 else 0;best=max(best,run)
   f.write(json.dumps({'at':time.time(),'frame':index,'meanPixelDifference':diff,'nearIdenticalRun':run})+'\n')
   if index%1920==0:f.flush();print(json.dumps({'seconds':round(time.monotonic()-started),'frames':index,'longestNearIdenticalFrames':best}),flush=True)
   index+=1;last=a
finally:
 if p.poll() is None:p.terminate()
 p.wait(timeout=10)
print(json.dumps({'frames':index,'longestNearIdenticalFrames':best,'longestNearIdenticalSeconds':best/32}),flush=True)
