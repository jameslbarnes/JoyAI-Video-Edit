"""Matched offline SR comparison. Never changes live service settings."""
import json,sys,time,statistics,gc
from pathlib import Path
import av,cv2,numpy as np,torch
from PIL import Image
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'deploy'),str(root/'tulip/research/vfx-probe')]
from xvideo.serving.upscale_output import PixelUpscaler
from nvvfx import VideoSuperRes,__version__,get_sdk_version
from nvvfx.effects.video_super_res import QualityLevel
out=root/'tulip/evidence/vsr-matched-20260922';plan=json.loads((out/'plan.json').read_text())
torch.set_num_threads(2);cv2.setNumThreads(1)
assert torch.cuda.mem_get_info()[0]>2*2**30,'Need 2GiB free on output GPU'
frames=[]
with av.open(str(root/plan['source'])) as c:
 for i,f in enumerate(c.decode(video=0)):
  if i>=plan['startFrame']:frames.append(f.to_ndarray(format='rgb24'))
  if len(frames)==plan['frames']:break
assert len(frames)==plan['frames']
reports=[]
for method in plan['methods']:
 times=[];gpu_times=[];luma=[];model=None;effect=None
 stream=torch.cuda.Stream(device=0,priority=0)
 if method=='favorite_realesrgan':
  import os
  os.environ['JOYAI_SR_GRAPH']='1'
  model=PixelUpscaler('cuda:0',.25);model.stream=stream
  model.warmup_graph(frames[0].shape[:2],(1440,2592));assert model.graph is not None,model.graph_error
 else:
  q=QualityLevel.ULTRA if method=='vsr_ultra' else QualityLevel.HIGHBITRATE_ULTRA
  effect=VideoSuperRes(quality=q,device=0);effect.output_width=2592;effect.output_height=1440;effect.load()
 def render(frame):
  with torch.inference_mode(),torch.cuda.stream(stream):
   if model is not None:return model.process_gpu(frame,(1440,2592)).wait()
   x=torch.from_numpy(frame).permute(2,0,1).contiguous().to('cuda:0',dtype=torch.float32)/255
   y=torch.from_dlpack(effect.run(x,non_blocking=True,stream_ptr=stream.cuda_stream).image)
   return y.clamp(0,1).mul(255).byte().permute(1,2,0).contiguous()
 for _ in range(4):pixels=render(frames[0]);stream.synchronize();time.sleep(.12)
 with av.open(str(out/(method+'.mp4')),'w') as c:
  enc=c.add_stream('libx264',rate=plan['fps']);enc.width=2592;enc.height=1440;enc.pix_fmt='yuv420p';enc.options={'crf':'16','preset':'veryfast','threads':'2'}
  for i,frame in enumerate(frames):
   tick=time.perf_counter();pixels=render(frame);stream.synchronize();times.append((time.perf_counter()-tick)*1000)
   with torch.cuda.stream(stream):array=pixels.cpu().numpy()
   assert array.shape==(1440,2592,3) and array.std()>1,'Invalid output'
   if i in (0,32,64,95):Image.fromarray(array).save(out/(method+'-'+str(i)+'.png'))
   v=av.VideoFrame.from_ndarray(array,format='rgb24');v.pts=i
   for packet in enc.encode(v):c.mux(packet)
   time.sleep(.12)
  for packet in enc.encode():c.mux(packet)
 if effect is not None:effect.close()
 report={'method':method,'frames':len(times),'medianMs':statistics.median(times),'p95Ms':float(np.percentile(times,95)),'timesMs':times}
 reports.append(report);print(json.dumps({k:v for k,v in report.items() if k!='timesMs'}),flush=True)
 del model,effect,pixels;gc.collect();torch.cuda.empty_cache()
 (out/'results.json').write_text(json.dumps({'plan':plan,'package':__version__,'sdk':get_sdk_version(),'reports':reports,'note':'Matched native frames, same physical output GPU, low-priority stream. Live workload competes; timings include upload/output ownership but exclude CPU readback and encoding. No high-resolution ground truth.'},indent=2))
