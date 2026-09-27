"""Graph-safe SageAttention for a single sequence with a valid key prefix.

Uses SageAttention's variable-length Triton kernels with capacity-sized scale
buffers. Its public varlen wrapper sizes buffers from GPU scalars, which cannot
be captured. All live lengths here stay on the GPU; no padded key is attended.
"""
import torch


def prepare_prefix_mask(mask, query_tokens, *, text_capacity=None):
    """Attach persistent sequence metadata before capture (batch size one)."""
    from sageattention.triton.quant_per_block_varlen import quant_per_block_int8_kernel
    from sageattention.triton.attn_qk_int8_block_varlen import forward
    nk = mask.shape[-1]
    mask._sage_prefix = (
        torch.tensor([0, query_tokens], device=mask.device, dtype=torch.int32),
        torch.tensor([0, nk], device=mask.device, dtype=torch.int32),
        torch.tensor([0, (query_tokens+127)//128], device=mask.device, dtype=torch.int32),
        torch.tensor([0, (nk+63)//64], device=mask.device, dtype=torch.int32),
        quant_per_block_int8_kernel, forward,
    )
    if text_capacity is not None:
        if not 0 < text_capacity <= query_tokens <= nk:
            raise ValueError('Invalid text/query capacities')
        mask._fp8_text_rows = torch.tensor([text_capacity],device=mask.device,dtype=torch.int32)
        mask._fp8_query_prefix = query_tokens - text_capacity
        mask._fp8_text_capacity = text_capacity


def protected_text_rows(query_prefix, text_tokens, capacity):
    """Keep whole 128-query quantization blocks, including their padded halo."""
    if query_prefix < 0 or not 0 < text_tokens <= capacity:
        raise ValueError('Invalid prefix or text length')
    return min(capacity, ((query_prefix + text_tokens + 127)//128)*128 - query_prefix)


def set_prefix_length(mask, length):
    if hasattr(mask, '_sage_prefix'):
        if hasattr(mask, '_fp8_text_rows'):
            capacity = mask._fp8_text_capacity
            tokens = length - (mask.shape[-1] - capacity)
            active = protected_text_rows(mask._fp8_query_prefix, tokens, capacity)
            mask._fp8_text_rows.fill_(active)
            mask._sage_prefix[0][1].fill_(mask._fp8_query_prefix + active)
        mask._sage_prefix[1][1].fill_(length)


def prefix_attention(query, key, value, mask):
    q, k, v = (x.squeeze(0) for x in (query, key, value))
    nq, heads, dim = q.shape
    nk = k.shape[0]
    if query.shape[0] != 1 or key.shape[0] != 1 or dim not in (64, 128):
        raise ValueError('Prefix attention requires batch one and head dimension 64 or 128')
    cq, ck, sq, sk, quantize, attend = mask._sage_prefix
    # Exclude padding from smoothing as well as attention/quantization. Padding
    # can contain large nonzero values after the transformer's text MLPs.
    mean = torch.where(mask.reshape(nk, 1, 1), k, 0).sum(dim=0, keepdim=True) / ck[1]
    k = k - mean
    qi, ki = torch.empty_like(q, dtype=torch.int8), torch.empty_like(k, dtype=torch.int8)
    qs = torch.empty(((nq+127)//128, heads), device=q.device, dtype=torch.float32)
    ks = torch.empty(((nk+63)//64, k.shape[1]), device=q.device, dtype=torch.float32)
    quantize[((nq+127)//128, heads, 1)](
        q, qi, qs, cq, sq, q.stride(1), q.stride(0), qi.stride(1), qi.stride(0),
        sm_scale=dim**-.5*1.44269504, H=heads, C=dim, BLK=128)
    quantize[((nk+63)//64, k.shape[1], 1)](
        k, ki, ks, ck, sk, k.stride(1), k.stride(0), ki.stride(1), ki.stride(0),
        sm_scale=1., H=k.shape[1], C=dim, BLK=64)
    output = attend(qi, ki, v.to(torch.float16), cq, ck, nq, qs, ks, sq, sk,
                    output_dtype=q.dtype)
    if hasattr(mask, '_fp8_text_rows'):
        from xvideo.models.dit.fp8_prefix import clear_query_tail
        output = clear_query_tail(output, cq)
    return output.unsqueeze(0)
