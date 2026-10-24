"""Fused RGB8 -> NDI UYVY on CUDA; NDI's resolution-selected YUV matrix.

The output is studio-range YCbCr 4:2:2, U0 Y0 V0 Y1. No alpha is sent.
NDI selects 601 for SD, 709 for HD and 2020 above 1920x1080 (either axis).
The GPU frame and destination must remain alive until their stream completes.
"""
import torch
import triton
import triton.language as tl


def ndi_coefficients(width, height):
    if width > 1920 or height > 1080:
        return .2627, .0593
    if width > 720 or height > 576:
        return .2126, .0722
    return .299, .114


@triton.jit
def _pack_rgb_uyvy(rgb, output, pairs: tl.constexpr, kr: tl.constexpr,
                   kb: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < pairs
    r0 = tl.load(rgb + i*6, valid, 0).to(tl.float32)
    g0 = tl.load(rgb + i*6+1, valid, 0).to(tl.float32)
    b0 = tl.load(rgb + i*6+2, valid, 0).to(tl.float32)
    r1 = tl.load(rgb + i*6+3, valid, 0).to(tl.float32)
    g1 = tl.load(rgb + i*6+4, valid, 0).to(tl.float32)
    b1 = tl.load(rgb + i*6+5, valid, 0).to(tl.float32)
    y0 = kr*r0 + (1-kr-kb)*g0 + kb*b0
    y1 = kr*r1 + (1-kr-kb)*g1 + kb*b1
    # Average the two chroma samples before quantizing, avoiding a hue bias.
    u = 128 + ((b0+b1)-(y0+y1)) * (56 / (255*(1-kb)))
    v = 128 + ((r0+r1)-(y0+y1)) * (56 / (255*(1-kr)))
    y0 = 16 + y0*(219/255)
    y1 = 16 + y1*(219/255)
    tl.store(output+i*4, tl.minimum(240, tl.maximum(16, u+.5)).to(tl.uint8), valid)
    tl.store(output+i*4+1, tl.minimum(235, tl.maximum(16, y0+.5)).to(tl.uint8), valid)
    tl.store(output+i*4+2, tl.minimum(240, tl.maximum(16, v+.5)).to(tl.uint8), valid)
    tl.store(output+i*4+3, tl.minimum(235, tl.maximum(16, y1+.5)).to(tl.uint8), valid)


def pack_uyvy(rgb, output=None):
    if (rgb.device.type != 'cuda' or rgb.dtype != torch.uint8 or
            rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.is_contiguous() or
            rgb.shape[1] % 2):
        raise ValueError('NDI packing requires contiguous CUDA RGB8 with even width')
    h, w, _ = rgb.shape
    if output is None:
        output = torch.empty((h, w, 2), device=rgb.device, dtype=torch.uint8)
    if (output.shape != (h, w, 2) or output.dtype != torch.uint8 or
            output.device != rgb.device or not output.is_contiguous()):
        raise ValueError('NDI packing destination does not match the source')
    kr, kb = ndi_coefficients(w, h)
    with torch.cuda.device(rgb.device):
        _pack_rgb_uyvy[(triton.cdiv(h*w//2, 256),)](
            rgb, output, h*w//2, kr, kb, BLOCK=256, enable_fp_fusion=False)
    return output
