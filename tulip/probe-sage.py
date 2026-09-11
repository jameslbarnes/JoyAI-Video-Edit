"""Verify patched SageAttention in eager execution and CUDA-graph replay."""

import json
from pathlib import Path
import torch
from sageattention import sageattn

torch.manual_seed(42)
q = torch.randn(1, 256, 4, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)


def run():
    return sageattn(q, k, v, tensor_layout="NHD", is_causal=False)


def reference():
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()
    ).transpose(1, 2)


def error(actual):
    expected = reference()
    value = (
        ((actual.float() - expected).square().mean() / expected.square().mean())
        .sqrt()
        .item()
    )
    assert value < 0.08, value
    assert torch.isfinite(actual).all()
    return value


eager = error(run())
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(3):
        run()
torch.cuda.current_stream().wait_stream(stream)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    result = run()
graph.replay()
first = error(result)
q.copy_(torch.randn_like(q))
k.copy_(torch.randn_like(k))
v.copy_(torch.randn_like(v))
graph.replay()
changed = error(result)
report = {
    "gpu": torch.cuda.get_device_name(),
    "torch": torch.__version__,
    "sage_version": "2.2.0+JoyAI current-stream patch+Windows compiler flags",
    "relative_rmse_eager": eager,
    "relative_rmse_graph": first,
    "relative_rmse_graph_changed_inputs": changed,
    "status": "passed",
}
print(json.dumps(report, indent=2))
(Path(__file__).parent / "sage-probe.json").write_text(json.dumps(report, indent=2))
