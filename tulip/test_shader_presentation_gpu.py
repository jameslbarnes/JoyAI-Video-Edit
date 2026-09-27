"""Bounded GPU proof. Run while JoyAI is stopped; exits and releases its context."""
import importlib.util
from pathlib import Path
import json
import torch

source=Path(__file__).resolve().parents[1]/'deploy/xvideo/serving/shader_strength.py'
spec=importlib.util.spec_from_file_location('strength_gpu_test',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert torch.cuda.is_available()
with torch.cuda.device(0):
    generated=torch.randint(256,(8,648,1152,3),device='cuda:0',dtype=torch.uint8)
    source=torch.randint_like(generated,256)
    generated_before=generated.clone();source_before=source.clone()
    for previous,current in [(0,0),(0,.5),(.5,1),(1,1),(1,0)]:
        result=module.mix_shader_presentation(generated,source,{'outputMix':current,'outputMixPrevious':previous})
        assert result.is_cuda and result.dtype==torch.uint8
        if previous==current==1:assert result is source and torch.equal(result,source_before)
        if previous==current==0:assert result is generated
        assert torch.equal(generated,generated_before) and torch.equal(source,source_before)
    torch.cuda.synchronize()
print(json.dumps({'gpu_proof':'passed','geometry':[1152,648],'frames':8,'raw_endpoint_exact':True,
                  'history_unmodified':True,'mix_remained_on_gpu':True}))
