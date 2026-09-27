import json,pathlib,sys,tempfile,time,statistics,subprocess
import numpy as np
import torch
from PIL import Image
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'deploy'))
from xvideo.serving.upscale_output import OutputUpscale
from xvideo.serving.rife_output import RifeOutput
stage=pathlib.Path('C:/Users/Admin/Downloads/etherea-dlss5-20260914')
torch.set_num_threads(2);torch.set_num_interop_threads(1)
raw=subprocess.check_output([str(stage/'trial-tool/out/ffmpeg.exe'),'-v','error','-i',str(root/'tulip/evidence/upscale-pixel-same-input-20260912/input.mp4'),'-f','rawvideo','-pix_fmt','rgb24','-'])
frames=np.frombuffer(raw,np.uint8).reshape(-1,360,648,3)
results=[]
with tempfile.TemporaryDirectory() as td:
 cfgpath=pathlib.Path(td)/'nr.json'
 cfg={'enabled':False,'executable':str(root/'tulip/research/video2dlssnr/out/video2dlssnr.exe'),'adapter':0,'intensity':1.0}
 cfgpath.write_text(json.dumps(cfg))
 sr=OutputUpscale('cuda:0');sr.neural_rendering.config_path=cfgpath
 rife=RifeOutput('cuda:0');sr.warmup();assert sr.enabled,sr.error
 try:
  for order,mult,nr in [('legacy',2,False),('reordered',2,False),('reordered',2,True),('reordered',4,False),('reordered',4,True)]:
   cfg['enabled']=nr;cfgpath.write_text(json.dumps(cfg));rife.reset();records=[]
   for j in range(6):
    batch=list(frames[j*8:(j+1)*8]);metas=[{'seq':j*8+k} for k in range(8)];profile={};t=time.perf_counter()
    if order=='legacy':
     pixels,mapped=rife.process(batch,metas,profile,raw_output=True,multiplier=mult)
     jpg=sr.process(pixels,profile)
    else:
     enhanced=sr.process(batch,profile,raw_output=True)
     pixels,mapped=rife.process(enhanced,metas,profile,raw_output=True,multiplier=mult)
     originals=[p for p,m in zip(pixels,mapped) if not m['interpolated']]
     assert all(np.array_equal(a,b) for a,b in zip(originals,enhanced))
     jpg=list(sr.encoder.map(sr.encode,pixels,[95]*len(pixels)))
    elapsed=time.perf_counter()-t
    assert len(jpg)==(8*mult if j else 7*mult+1),(order,mult,j,len(jpg))
    assert sr.neural_rendering.error is None,sr.neural_rendering.status()
    records.append({'seconds':elapsed,'frames':len(jpg),'profile':profile})
   warm=records[2:];total=sum(r['seconds'] for r in warm)
   report={'order':order,'rife':mult,'nr':nr,'sourceFps':32/total,'outputFps':sum(r['frames'] for r in warm)/total,'chunkMedianMs':statistics.median(r['seconds'] for r in warm)*1000,'cudaPeakMiB':torch.cuda.max_memory_allocated()/2**20,'samples':records}
   results.append(report);print(json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)
 finally:sr.close()
(stage/'optimization-benchmark.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
