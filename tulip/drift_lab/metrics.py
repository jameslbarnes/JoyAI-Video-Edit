"""Reference-free artifact metrics for generated frames, judged against their input.

All metrics work on small grayscale frames (default 480x270) split into a
grid, because the junk we chase is local (a glassy chest, chrome ladders at
an edge) and a whole-frame average hides it.

  detail      mean |Laplacian| per cell: fine high-contrast structure.
  excess      output detail minus input detail per cell; junk = high
              percentile of excess (detail the input does not explain).
  staleness   output motion / input motion per cell over the same interval
              (low = the region ignores the moving input).
  imprint     correlation of output and input Laplacian maps per cell at the
              best input lag (high = the input's texture shows through).
"""
from __future__ import annotations

import numpy as np

GRID = (4, 6)  # rows, cols


def gray(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] * .299 + rgb[..., 1] * .587 + rgb[..., 2] * .114).astype(np.float32) / 255.0


def laplacian(g: np.ndarray) -> np.ndarray:
    out = np.zeros_like(g)
    out[1:-1, 1:-1] = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1])
    return out


def cells(a: np.ndarray, grid=GRID):
    rows, cols = grid
    h, w = a.shape[:2]
    for r in range(rows):
        for c in range(cols):
            yield r, c, a[r * h // rows:(r + 1) * h // rows, c * w // cols:(c + 1) * w // cols]


def cell_mean(a: np.ndarray, grid=GRID) -> np.ndarray:
    m = np.zeros(grid, np.float32)
    for r, c, block in cells(a, grid):
        m[r, c] = float(block.mean())
    return m


def cell_corr(a: np.ndarray, b: np.ndarray, grid=GRID) -> np.ndarray:
    m = np.zeros(grid, np.float32)
    blocks_b = {(r, c): blk for r, c, blk in cells(b, grid)}
    for r, c, blk in cells(a, grid):
        x, y = blk.ravel(), blocks_b[(r, c)].ravel()
        x, y = x - x.mean(), y - y.mean()
        den = float(np.sqrt((x * x).sum() * (y * y).sum()))
        m[r, c] = float((x * y).sum() / den) if den > 1e-9 else 0.0
    return m


def frame_metrics(out_rgb, in_rgb, prev_out_rgb=None, prev_in_rgb=None, lagged_inputs=()):
    """Metrics for one output frame. ``lagged_inputs`` are earlier input frames
    (any order) searched for the best texture alignment."""
    go, gi = gray(out_rgb), gray(in_rgb)
    lo, li = np.abs(laplacian(go)), np.abs(laplacian(gi))
    d_out, d_in = cell_mean(lo), cell_mean(li)
    excess = d_out - d_in
    result = {
        "detail": float(d_out.mean()),
        "detailMax": float(d_out.max()),
        "inputDetail": float(d_in.mean()),
        "junk": float(np.percentile(excess, 90)),
        "junkCells": excess.round(4).tolist(),
        "brightness": float(go.mean()),
        "inputBrightness": float(gi.mean()),
    }
    if prev_out_rgb is not None and prev_in_rgb is not None:
        mo = cell_mean(np.abs(go - gray(prev_out_rgb)))
        mi = cell_mean(np.abs(gi - gray(prev_in_rgb)))
        ratio = mo / (mi + 0.01)
        result.update(motion=float(mo.mean()), inputMotion=float(mi.mean()),
                      staleness=float(np.median(ratio)), stalenessMin=float(ratio.min()),
                      staleCells=ratio.round(3).tolist())
    best = None
    for lag, rgb in enumerate([in_rgb, *lagged_inputs]):
        corr = cell_corr(laplacian(go), laplacian(gray(rgb)))
        score = float(corr.mean())
        if best is None or score > best[0]:
            best = (score, lag, corr)
    if best:
        result.update(imprint=best[0], imprintLag=best[1], imprintMax=float(best[2].max()))
    return result
