"""Bounded GPU encode/color check; no inference and no live stream mutation."""
import json,sys,time,subprocess
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deploy'))
from xvideo.serving.gpu_video_output import CudaRgba,FlvH264,GpuVideoPublisher,GpuFrame,H264_COLOR_METADATA

ffmpeg=Path('C:/Users/Admin/src/easel/external/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe')
p=GpuVideoPublisher('cuda:0',str(ffmpeg),'rtmp://127.0.0.1:1935/joyai-gpu-validation')
colors=np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,255],[0,0,0],[128,128,128]],dtype=np.uint8)
with torch.cuda.device(0):
    p.stream=torch.cuda.Stream()
    with torch.cuda.stream(p.stream):
        p._open((2592,1440,32.))
        pixels=torch.from_numpy(np.repeat(np.repeat(colors[None],432,axis=1),1440,axis=0)).cuda()
        frame=GpuFrame.record(pixels)
        p.stream.synchronize()
        output=Path('tulip/gpu-color-validation.flv')
        written=[]
        p._write=lambda b:written.append(bytes(b))
        timings=[]
        try:
            for i in range(32):
                start=time.perf_counter()
                for packet in p._encode(frame,32):p._write(packet)
                timings.append((time.perf_counter()-start)*1000)
                time.sleep(max(0,1/32-(time.perf_counter()-start)))
            output.write_bytes(FlvH264.header+b''.join(written))
        finally:p._release_encoder()
tagged=output.with_name('gpu-color-validation-tagged.flv')
subprocess.run([str(ffmpeg),'-hide_banner','-loglevel','error','-y','-i',str(output),'-c:v','copy','-bsf:v',H264_COLOR_METADATA,str(tagged)],check=True)
raw=subprocess.check_output([str(ffmpeg),'-hide_banner','-loglevel','error','-i',str(tagged),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
decoded=np.frombuffer(raw,np.uint8).reshape(1440,2592,3)
actual=np.array([decoded[720,i*432+216] for i in range(6)])
error=np.abs(actual.astype(int)-colors).max()
result={'colorsExpected':colors.tolist(),'colorsDecoded':actual.tolist(),'maxChannelError':int(error),'encodeAndGpuPreparationMs':timings,'packets':p.packet_count}
Path('tulip/gpu-color-validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
assert error<=5,'Color conversion/range mismatch'
assert p.packet_count>=28
