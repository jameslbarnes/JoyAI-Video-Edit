"""Small real-GPU compatibility gate before downloading any model weights."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JOYOMNI_FP8_FAST_ACCUM"] = "1"
import importlib.util
import json
from pathlib import Path
import torch
import joyomni_ops as ops

torch.manual_seed(123)
device = "cuda:0"
x = torch.randn(32, 1024, device=device, dtype=torch.bfloat16)
w = torch.randn(1024, device=device, dtype=torch.bfloat16)
y = ops.rmsnorm(x, w)
reference = (
    x.float()
    * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    * w.float()
).bfloat16()
torch.testing.assert_close(y, reference, rtol=0.01, atol=0.015625)
shift = torch.randn_like(x) * 0.1
scale = torch.randn_like(x) * 0.1
normalized = ops.fused_norm_scale_shift(x, None, None, scale, shift, "layer", 1e-6)
refnorm = (
    torch.nn.functional.layer_norm(x.float(), (1024,), eps=1e-6) * (1 + scale.float())
    + shift.float()
)
torch.testing.assert_close(normalized, refnorm.bfloat16(), rtol=0.02, atol=0.03125)
q = torch.randn(1, 32, 128, device=device, dtype=torch.bfloat16)
k = torch.randn_like(q)
qw = torch.ones(128, device=device, dtype=torch.bfloat16)
cos = torch.ones(8, 64, device=device, dtype=torch.bfloat16)
sin = torch.zeros_like(cos)
rq = (
    q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + 1e-6)
).bfloat16()
rk = (
    k.float() * torch.rsqrt(k.float().square().mean(-1, keepdim=True) + 1e-6)
).bfloat16()
ops.fused_qk_norm_rope_3d_paired(q, k, 8, 4, 1e-6, qw, qw, cos, sin)
torch.testing.assert_close(q, rq, rtol=0.02, atol=0.03125)
torch.testing.assert_close(k, rk, rtol=0.02, atol=0.03125)
q = torch.randn(1, 32, 128, device=device, dtype=torch.bfloat16)
k = torch.randn_like(q)
qw = torch.randn(128, device=device, dtype=torch.bfloat16)
kw = torch.randn_like(qw)
angle = torch.randn(8, 64, device=device)
cos = angle.cos().bfloat16()
sin = angle.sin().bfloat16()


def rope_reference(value, weight):
    xf = value.float().view(1, 8, 4, 128)
    norm = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * weight.float()
    c = cos.float().view(8, 1, 64)
    s = sin.float().view(8, 1, 64)
    result = torch.empty_like(norm)
    result[..., 0::2] = norm[..., 0::2] * c - norm[..., 1::2] * s
    result[..., 1::2] = norm[..., 1::2] * c + norm[..., 0::2] * s
    return result.view_as(value).bfloat16()


rq = rope_reference(q, qw)
rk = rope_reference(k, kw)
ops.fused_qk_norm_rope_3d_paired(q, k, 8, 4, 1e-6, qw, kw, cos, sin)
torch.testing.assert_close(q, rq, rtol=0.02, atol=0.03125)
torch.testing.assert_close(k, rk, rtol=0.02, atol=0.03125)
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "joy_fp8_probe", root / "deploy/xvideo/models/dit/fp8_linear.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
linear = torch.nn.Linear(1024, 512, device=device, dtype=torch.bfloat16)
fp8 = module.Fp8Linear.from_linear(linear)
actual = fp8(x)
expected = linear(x)
relative_rmse = (
    (
        (actual.float() - expected.float()).square().mean()
        / expected.float().square().mean()
    )
    .sqrt()
    .item()
)
assert relative_rmse < 0.08, relative_rmse
assert torch.isfinite(actual).all()
torch.cuda.synchronize()
result = {
    "gpu": torch.cuda.get_device_name(),
    "torch": torch.__version__,
    "fused_rmsnorm": True,
    "fused_norm_modulation": True,
    "fused_qk_norm_rope_identity": True,
    "triton_fp8_relative_rmse": relative_rmse,
    "fused_qk_norm_rope_weighted_rotations": True,
    "cutlass_fp8_available": ops.has_fp8(),
    "note": "Windows light CUDA extension + upstream GeForce Triton FP8 path; not a model inference benchmark",
}
print(json.dumps(result, indent=2))
(root / "tulip/native-probe.json").write_text(json.dumps(result, indent=2))
