"""Reduce an RGBA image to a <=256-entry palette (index 0 = transparent)."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _median_cut(colors: np.ndarray, counts: np.ndarray, n: int) -> np.ndarray:
    """Weighted median cut; returns (n_boxes, 4) mean colours."""

    def make(b: np.ndarray) -> list:
        c = colors[b]
        rng = c.max(axis=0) - c.min(axis=0)
        ch = int(np.argmax(rng))
        score = float(rng[ch]) * float(counts[b].sum()) if len(b) > 1 else 0.0
        return [b, score, ch]

    boxes = [make(np.arange(len(colors)))]
    while len(boxes) < n:
        i = max(range(len(boxes)), key=lambda k: boxes[k][1])
        if boxes[i][1] <= 0:
            break
        b, _, ch = boxes.pop(i)
        b = b[np.argsort(colors[b, ch], kind="stable")]
        cum = np.cumsum(counts[b])
        cut = int(np.searchsorted(cum, cum[-1] / 2.0))
        cut = min(max(cut, 1), len(b) - 1)
        boxes.append(make(b[:cut]))
        boxes.append(make(b[cut:]))
    return np.array([np.average(colors[b], axis=0, weights=counts[b]) for b, _, _ in boxes])


def _weighted_means(colors: np.ndarray, counts: np.ndarray, mapping: np.ndarray,
                    old: np.ndarray) -> np.ndarray:
    k = len(old)
    wsum = np.bincount(mapping, weights=counts, minlength=k)
    out = old.copy()
    nz = wsum > 0
    for c in range(colors.shape[1]):
        s = np.bincount(mapping, weights=colors[:, c] * counts, minlength=k)
        out[nz, c] = s[nz] / wsum[nz]
    return out


def _nearest(colors: np.ndarray, palette: np.ndarray, chunk: int = 8192) -> np.ndarray:
    out = np.empty(len(colors), dtype=np.int64)
    w = np.array([1.0, 1.0, 1.0, 1.5])  # alpha errors are the most visible
    for s in range(0, len(colors), chunk):
        c = colors[s:s + chunk]
        d = (((c[:, None, :] - palette[None, :, :]) * w) ** 2).sum(axis=2)
        out[s:s + chunk] = d.argmin(axis=1)
    return out


def quantize(pm: np.ndarray, max_colors: int = 255) -> Tuple[np.ndarray, np.ndarray]:
    """Premultiplied float RGBA (0..255) -> (index image uint8, straight RGBA palette).

    Palette entry 0 is fully transparent; at most ``max_colors`` further
    entries are used.
    """
    pm8 = np.clip(np.round(pm), 0, 255).astype(np.uint8)
    # premultiplied colour channels can never exceed alpha
    pm8[..., :3] = np.minimum(pm8[..., :3], pm8[..., 3:4])
    h, w = pm8.shape[:2]
    alpha = pm8[..., 3]
    vis = alpha > 0
    idx = np.zeros((h, w), dtype=np.uint8)
    if not vis.any():
        return idx, np.zeros((1, 4), dtype=np.uint8)
    flat = pm8[vis].astype(np.uint32)
    packed = (flat[:, 0] << 24) | (flat[:, 1] << 16) | (flat[:, 2] << 8) | flat[:, 3]
    uniq, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    ucol = np.stack([(uniq >> 24) & 255, (uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255],
                    axis=1).astype(np.float64)
    if len(uniq) <= max_colors:
        pal_pm = ucol
        mapping = np.arange(len(uniq))
    else:
        pal_pm = _median_cut(ucol, counts.astype(np.float64), max_colors)
        for _ in range(3):  # weighted k-means refinement
            mapping = _nearest(ucol, pal_pm)
            pal_pm = _weighted_means(ucol, counts.astype(np.float64), mapping, pal_pm)
        mapping = _nearest(ucol, pal_pm)
    idx[vis] = (mapping[inverse.ravel()] + 1).astype(np.uint8)
    # straight-alpha palette
    a = np.clip(np.round(pal_pm[:, 3]), 1, 255)
    rgb = np.clip(np.round(pal_pm[:, :3] * 255.0 / a[:, None]), 0, 255)
    pal = np.zeros((len(pal_pm) + 1, 4), dtype=np.uint8)
    pal[1:, :3] = rgb
    pal[1:, 3] = a
    return idx, pal


def trim_indexed(idx: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Crop an index image to its non-transparent bounding box."""
    rows = np.flatnonzero(idx.any(axis=1))
    if rows.size == 0:
        return idx[:0, :0], 0, 0
    cols = np.flatnonzero(idx.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    return idx[y0:y1, x0:x1], x0, y0
