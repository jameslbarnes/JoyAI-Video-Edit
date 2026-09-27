"""Masked attention for fixed-capacity streaming prompt graphs.

The SM120 SageAttention dispatcher ignores arbitrary masks. Streaming graphs
with explicit prefix metadata use graph-safe varlen kernels; other masks use
cuDNN. Never pass a padded prompt to the unmasked dispatcher.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def masked_attention(query, key, value, mask):
    """NHD attention; True keys participate, False keys contribute nothing."""
    if query.is_cuda and hasattr(mask, '_sage_prefix'):
        from xvideo.models.dit.prefix_attention import prefix_attention
        return prefix_attention(query, key, value, mask)
    q, k, v = (x.transpose(1, 2) for x in (query, key, value))
    if query.is_cuda:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2)
