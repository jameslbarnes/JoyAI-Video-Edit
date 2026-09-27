"""Explicit GPU integration check: exact pixels, output ownership and size fallback."""
import json,os,sys
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/"deploy"))
from xvideo.serving.upscale_output import PixelUpscaler
assert torch.cuda.mem_get_info(0)[0]>2*2**30
u=PixelUpscaler("cuda:0",.25);rng=np.random.default_rng(42)
frames=[rng.integers(0,256,(648,1152,3),dtype=np.uint8) for _ in range(3)]
expected=[u.process(f,(1440,2592)) for f in frames]
os.environ["JOYAI_SR_GRAPH"]="1";u.warmup_graph((648,1152),(1440,2592))
assert u.graph is not None,u.graph_error
# Reproduce production ordering: SR captured first, then a no-grad model graph
# on the other GPU. An implicit process-global capture stream is unsafe here.
with torch.no_grad(),torch.cuda.device(1):
    test_input=torch.ones(16,device="cuda:1")
    test_graph=torch.cuda.CUDAGraph();test_stream=torch.cuda.Stream(device=1)
    with torch.cuda.graph(test_graph,stream=test_stream):
        test_output=test_input*2
    test_graph.replay();torch.cuda.synchronize(1)
    assert torch.equal(test_output,torch.full_like(test_input,2))

retained=[u.process_gpu(f,(1440,2592)) for f in frames]
actual=[g.wait().cpu().numpy() for g in retained]
assert len({g.tensor.data_ptr() for g in retained})==3,"Publisher outputs alias"
for a,b in zip(actual,expected): assert np.array_equal(a,b),int(np.abs(a.astype(int)-b.astype(int)).max())
assert u.process(frames[0],(1080,1944)).shape==(1080,1944,3)
result=dict(exactPixelCases=3,distinctOutputStorage=True,crossDeviceCapture=True,alternateSizeEager=True,peakMiB=torch.cuda.max_memory_allocated()/2**20)
(root/"tulip/evidence/shared-vae-20260922/sr-graph-verification.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))
