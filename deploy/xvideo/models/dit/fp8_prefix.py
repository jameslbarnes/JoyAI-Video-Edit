"""Graph-safe active-row wrappers around the existing FP8 arithmetic.

The row limit stays on the GPU. Static tensor shapes/addresses remain intact,
and unused output rows are initialized. No extra CUDA graph pools are needed.
"""
import torch
import triton
import triton.language as tl

from xvideo.models.dit.fp8_linear import _FAST_ACCUM, _FA_DIV
if _FAST_ACCUM:
    from xvideo.models.dit.fp8_linear import _fa_mm_kernel, _fa_quant_kernel, _fa_pick_config


@triton.jit
def _prefix_quant(X, XQ, XS, ROWS, M, K, sxm, sxk,
                  BLOCK: tl.constexpr, DIV: tl.constexpr):
    active = tl.load(ROWS)
    if tl.program_id(0) < active:
        _fa_quant_kernel(X, XQ, XS, active, K, sxm, sxk, BLOCK, DIV)


@triton.jit
def _prefix_mm(A, B, C, XS, WS, BIAS, ROWS, M, N, K,
               sam, sak, sbk, sbn, scm, scn,
               HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr, GROUP: tl.constexpr):
    active = tl.load(ROWS)
    pid = tl.program_id(0)
    columns = tl.cdiv(N, BN)
    # Clear only the inactive rows. This mapping is independent of the grouped
    # GEMM mapping; the store masks make the two sets of writes disjoint.
    clear_m = (pid // columns) * BM
    if clear_m + BM > active:
        rm = clear_m + tl.arange(0, BM)
        rn = (pid % columns) * BN + tl.arange(0, BN)
        tl.store(C + rm[:, None]*scm + rn[None, :]*scn, 0.,
                 mask=(rm[:, None] >= active) & (rm[:, None] < M) & (rn[None, :] < N))
    if pid < tl.cdiv(active, BM) * columns:
        _fa_mm_kernel(A, B, C, XS, WS, BIAS, active, N, K,
                      sam, sak, sbk, sbn, scm, scn, HAS_BIAS, BM, BN, BK, GROUP)


@triton.jit
def _zero_query_tail(OUT, CU_QUERY, NQ: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    active = tl.load(CU_QUERY + 1)
    rows = block * 128 + tl.arange(0, 128)
    if block * 128 + 128 > active:
        cols = tl.arange(0, DIM)
        tl.store(OUT + rows[:, None]*(HEADS*DIM) + head*DIM + cols[None, :], 0.,
                 mask=(rows[:, None] >= active) & (rows[:, None] < NQ))


def clear_query_tail(output, cu_query):
    nq, heads, dim = output.shape
    _zero_query_tail[(triton.cdiv(nq,128),heads)](output,cu_query,nq,heads,dim,num_warps=4)
    return output


def prefix_linear(module, x, active_rows):
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    m, k = x2.shape
    if x.ndim != 3 or x.shape[0] != 1 or k != module.in_features:
        raise ValueError('Active text rows require batch-one token features')
    xq = torch.empty((m,k), device=x.device, dtype=torch.float8_e4m3fn)
    xs = torch.empty((m,), device=x.device, dtype=torch.float32)
    _prefix_quant[(m,)](x2,xq,xs,active_rows,m,k,x2.stride(0),x2.stride(1),
                        BLOCK=1024,DIV=_FA_DIV,num_warps=4)
    n = module.out_features
    output = torch.empty((m,n),device=x.device,dtype=module.out_dtype)
    bm,bn,warps,stages = _fa_pick_config(m,n,k)
    _prefix_mm[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](
        xq,module.weight_fp8,output,xs,module.weight_scale,
        module.bias if module.bias is not None else output,active_rows,m,n,k,
        xq.stride(0),xq.stride(1),module.weight_fp8.stride(0),module.weight_fp8.stride(1),
        output.stride(0),output.stride(1),HAS_BIAS=module.bias is not None,
        BM=bm,BN=bn,BK=128,GROUP=8,num_warps=warps,num_stages=stages)
    return output.reshape(*shape[:-1],n)
