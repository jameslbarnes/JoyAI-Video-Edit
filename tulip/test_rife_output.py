"""Real-GPU smoke/benchmark of the optional read-only FluxRT RIFE adapter."""
import io
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import torch
from PIL import Image, ImageDraw
from xvideo.serving.rife_output import RifeOutput
if len(sys.argv) > 1:
    torch.set_num_threads(int(sys.argv[1]))

model = RifeOutput("cuda:0")
frames = []
for i in range(8):
    im = Image.new("RGB", (648, 360), (15, 20, 45))
    ImageDraw.Draw(im).ellipse((90+i*6, 80, 230+i*6, 220), fill=(230, 80, 35))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95)
    frames.append(buf.getvalue())
metas = [{"seq":i+1, "t_capture_ms":i*62.5} for i in range(8)]
profiles=[]
for _ in range(4):
    model.reset()
    profile={}
    out, mapped = model.process(frames, metas, profile)
    assert len(out)==15 and len(mapped)==15
    assert out[::2]==frames, "Original JPEGs must remain byte-identical"
    assert sum(bool(m.get("interpolated")) for m in mapped)==7
    assert mapped[1]["source_seq_previous"]==1 and mapped[1]["seq"]==2
    assert all(Image.open(io.BytesIO(j)).size==(648,360) for j in out)
    assert out[1]!=out[0] and out[1]!=out[2]
    profiles.append(profile)
model.reset()
out,_=model.process(frames[:1],metas[:1],{})
assert len(out)==1, "Reset must prevent cross-scene interpolation"
print(json.dumps({"device":torch.cuda.get_device_name(),"cpu_threads":torch.get_num_threads(),"profiles":profiles,
    "peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,
    "peak_reserved_MiB":torch.cuda.max_memory_reserved()/2**20}))
