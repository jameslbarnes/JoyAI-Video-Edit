"""Matched saved native frames through RIFE and SR; no live publishers."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deploy'))
p=argparse.ArgumentParser();p.add_argument('--out',required=True)
p.add_argument('--target-native-fps',type=float,default=16)
a=p.parse_args();out=Path(a.out)
import numpy as np
import torch
from PIL import Image
from xvideo.serving.rife_output import RifeOutput
from xvideo.serving.upscale_output import PixelUpscaler
from xvideo.serving.cadence_metrics import interval_metrics
torch.set_num_threads(2)
rife=RifeOutput('cuda:0')
rows=[]
for size in ['648x360','864x480']:
    frames=np.load(out/size/'native-last32.npy')
    for denoise in [.5,.25,0.]:
        sr=PixelUpscaler('cuda:1',denoise)
        combinations=[(2,1440)] if denoise!=.5 else [(1,1440),(2,1440),(4,1440),(2,1080)]
        for mult,height in combinations:
            rife.reset();records=[]
            for j in range(8):
                chunk=list(frames[(j%4)*8:(j%4+1)*8]);meta=[{'seq':j*8+k} for k in range(8)];profile={}
                tick=time.perf_counter()
                pixels,_=rife.process(chunk,meta,profile,raw_output=True,multiplier=mult) if mult>1 else (chunk,meta)
                rife_s=time.perf_counter()-tick;tick=time.perf_counter()
                enhanced=[sr.process(f,(height,height*9//5)) for f in pixels]
                sr_s=time.perf_counter()-tick
                records.append({'rife_s':rife_s,'sr_s':sr_s,'seconds':rife_s+sr_s,'frames':len(enhanced)})
                if j==3:
                    Image.fromarray(enhanced[-1]).save(out/size/f'sr-{height}-denoise{denoise}-rife{mult}.png')
            warm=records[2:];seconds=sum(r['seconds'] for r in warm)
            row={'native':size,'rife':mult,'height':height,'denoise':denoise,
                'input_fps':len(warm)*8/seconds,'output_fps':sum(r['frames'] for r in warm)/seconds,
                'rife_median_ms':statistics.median(r['rife_s'] for r in warm)*1000,
                'sr_median_ms':statistics.median(r['sr_s'] for r in warm)*1000,
                'chunk_p95_ms':float(np.percentile([r['seconds'] for r in warm],95))*1000,'records':records}
            row['chunkTiming'] = interval_metrics([r['seconds']*1000 for r in warm], 8/a.target_native_fps*1000)
            row['cadenceScope'] = 'Offline chunk compute only; does not certify paced playback. Run the live projector acceptance probe.'
            rows.append(row);(out/'presentation-benchmark.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!='records'}),flush=True)
        del sr
        torch.cuda.empty_cache()
