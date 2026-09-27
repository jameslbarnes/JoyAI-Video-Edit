"""Confirm warmup/autotuning uses the tensor's GPU and restores the caller device."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deploy'))
from xvideo.models.vae import vae_compile
assert torch.cuda.device_count()==2
torch.cuda.set_device(0)
seen=[]
def encode(x):
    seen.append(('encode',torch.cuda.current_device(),x.device.index))
    return x
def decode(x,**kwargs):
    seen.append(('decode',torch.cuda.current_device(),x.device.index))
    return (x,)
with patch.object(vae_compile,'maybe_setup_encode'),patch.object(vae_compile,'maybe_setup_decode'):
    vae_compile.warmup_encode(SimpleNamespace(encode=encode),3,24,24,
        device=torch.device('cuda:1'),dtype=torch.bfloat16,temporal_lens=(1,))
    vae_compile.warmup_decode(SimpleNamespace(decode=decode),2,2,2,
        device=torch.device('cuda:1'),dtype=torch.bfloat16,temporal_lens=(1,))
assert seen==[('encode',1,1),('decode',1,1)],seen
assert torch.cuda.current_device()==0
print(json.dumps(dict(ok=True,stages=seen,callerDeviceRestored=True)))
