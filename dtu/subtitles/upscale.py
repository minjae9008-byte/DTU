"""Subtitle bitmap upscaling algorithms (no neural networks).

``auto`` (default)
    Chooses per bitmap: ``contour`` when the subtitle contains anti-aliasing
    pixels, ``xbr`` for hard-edged few-colour bitmaps, ``lanczos`` otherwise.
``contour``
    Treats a DVD subtitle as nested layers (outline around fill, optional
    shadow) and reconstructs each layer boundary as a smooth curve: every
    cumulative layer mask is interpolated with a cubic kernel and re-thresholded
    at 0.5 with gradient-normalised, one-output-pixel-wide anti-aliasing (the
    signed-distance-field idea used for vector glyph magnification).  Pixels
    that the original renderer used for anti-aliasing (a colour half-way between
    two neighbouring layers) are interpreted as 50 % coverage, so edges end up
    sharper than on the disc instead of turning into blurry grey bands.
``xbr``
    Hyllian's xBR (level 2) pixel-art magnifier at 4x, then an area filter to
    the exact target size.  Robust for any few-colour bitmap.
``lanczos``
    Plain Lanczos-3 resampling of premultiplied RGBA (best for already
    anti-aliased, many-colour PGS bitmaps).
``nearest``
    Pixel replication (the "original DVD look").
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ALGORITHMS = ("auto", "contour", "xbr", "lanczos", "nearest")


# ---------------------------------------------------------------------------
# resampling matrices
# ---------------------------------------------------------------------------

def _cubic(x: np.ndarray, b: float, c: float) -> np.ndarray:
    x = np.abs(x)
    x2, x3 = x * x, x * x * x
    out = np.zeros_like(x)
    m1 = x < 1
    m2 = (x >= 1) & (x < 2)
    out[m1] = ((12 - 9 * b - 6 * c) * x3[m1] + (-18 + 12 * b + 6 * c) * x2[m1] + (6 - 2 * b)) / 6
    out[m2] = ((-b - 6 * c) * x3[m2] + (6 * b + 30 * c) * x2[m2] + (-12 * b - 48 * c) * x[m2]
               + (8 * b + 24 * c)) / 6
    return out


def _lanczos(x: np.ndarray, a: int = 3) -> np.ndarray:
    x = np.abs(x)
    out = np.sinc(x) * np.sinc(x / a)
    out[x >= a] = 0
    return out


KERNELS = {
    "catrom": (lambda x: _cubic(x, 0.0, 0.5), 2),
    "mitchell": (lambda x: _cubic(x, 1 / 3, 1 / 3), 2),
    "bspline": (lambda x: _cubic(x, 1.0, 0.0), 2),
    "lanczos": (_lanczos, 3),
    "bilinear": (lambda x: np.maximum(0, 1 - np.abs(x)), 1),
}


def resample_matrix(n_in: int, n_out: int, scale: float, offset: float, kernel: str) -> np.ndarray:
    """Weights (n_out, n_in) mapping input samples to output pixel centres.

    Output pixel ``o`` covers input coordinate ``(o + 0.5) / scale + offset``
    (input pixel ``i`` has its centre at ``i + 0.5``).  Samples outside the
    input count as zero (transparent padding): weights are normalised over
    the infinite lattice, then out-of-range taps are dropped.
    """
    fn, support = KERNELS[kernel]
    stretch = max(1.0, 1.0 / scale)  # widen the kernel when downscaling
    centers = (np.arange(n_out) + 0.5) / scale + offset - 0.5
    radius = support * stretch
    taps = int(math.ceil(2 * radius)) + 1
    j = np.floor(centers - radius).astype(np.int64)[:, None] + 1 + np.arange(taps)[None, :]
    d = (j - centers[:, None]) / stretch
    w = fn(d) * (np.abs(d) < support)
    norm = w.sum(axis=1, keepdims=True)
    norm[norm == 0] = 1
    w = w / norm
    valid = (j >= 0) & (j < n_in)
    m = np.zeros((n_out, n_in))
    rows = np.broadcast_to(np.arange(n_out)[:, None], j.shape)
    m[rows[valid], j[valid]] = w[valid]
    return m


def _resample2d(img: np.ndarray, wy: np.ndarray, wx: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return wy @ img @ wx.T
    return np.einsum("oy,yxc,px->opc", wy, img, wx, optimize=True)


# ---------------------------------------------------------------------------
# layer analysis for palettised (DVD) bitmaps
# ---------------------------------------------------------------------------

def _premul(rgba: np.ndarray) -> np.ndarray:
    c = rgba.astype(np.float64)
    a = c[..., 3:4] / 255.0
    return np.concatenate([c[..., :3] * a, c[..., 3:4]], axis=-1)


def _neighbours(lab: np.ndarray) -> List[np.ndarray]:
    """8-neighbourhood shifted copies (edge-replicated)."""
    p = np.pad(lab, 1, mode="edge")
    h, w = lab.shape
    return [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def _fast_distance(bg: np.ndarray, max_d: int = 12) -> np.ndarray:
    """Approximate distance-to-background by iterative erosion (fast in numpy)."""
    d = np.zeros(bg.shape, dtype=np.float64)
    inside = ~bg
    cur = inside.copy()
    for k in range(1, max_d + 1):
        if not cur.any():
            break
        d[cur] = k
        p = np.pad(cur, 1, constant_values=False)
        h, w = cur.shape
        er = cur.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                er &= p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        cur = er
    return d


class LayerModel:
    """Nested-layer interpretation of a palettised subtitle bitmap."""

    def __init__(self, labels: np.ndarray, colors: np.ndarray):
        # labels: (h, w) class ids; colors: (n, 4) straight RGBA per class
        self.labels = labels
        self.colors = colors
        pm = _premul(colors.astype(np.float64))
        visible = colors[:, 3] > 0
        # merge classes with identical premultiplied colour
        canon: Dict[tuple, int] = {}
        remap = np.zeros(len(colors), dtype=np.int64)
        self.layer_colors: List[np.ndarray] = []
        for k in range(len(colors)):
            if not visible[k]:
                remap[k] = 0
                continue
            key = tuple(np.round(pm[k]).astype(int).tolist())
            if key not in canon:
                canon[key] = len(canon) + 1
                self.layer_colors.append(pm[k])
            remap[k] = canon[key]
        lab = remap[labels]  # 0 = transparent, 1..m = distinct visible colours
        m = len(self.layer_colors)
        self.m = m
        # depth of each merged class: mean distance to transparent pixels
        dist = _fast_distance(lab == 0)
        depth = np.zeros(m + 1)
        counts = np.bincount(lab.ravel(), minlength=m + 1)
        sums = np.bincount(lab.ravel(), weights=dist.ravel(), minlength=m + 1)
        for k in range(1, m + 1):
            depth[k] = sums[k] / counts[k] if counts[k] else 0
        # anti-aliasing classes: sit between two other classes and have their midpoint colour
        colors_all = [np.zeros(4)] + self.layer_colors
        aa: Dict[int, Tuple[int, int]] = {}
        if m >= 2:
            nb = _neighbours(lab)
            for k in range(1, m + 1):
                sel = lab == k
                n_k = int(sel.sum())
                if n_k == 0:
                    continue
                pair_votes: Dict[Tuple[int, int], int] = {}
                present = np.stack([np.stack([(q == j)[sel] for q in nb]).any(axis=0)
                                    for j in range(m + 1)])  # (m+1, n_k)
                # how many k pixels touch both classes a and b
                best = None
                for a in range(m + 1):
                    if a == k:
                        continue
                    for b in range(a + 1, m + 1):
                        if b == k:
                            continue
                        votes = int((present[a] & present[b]).sum())
                        if best is None or votes > best[0]:
                            best = (votes, a, b)
                if best is None or best[0] < 0.5 * n_k:
                    continue
                _, a, b = best
                ca, cb, ck = colors_all[a], colors_all[b], colors_all[k]
                mid = 0.5 * (ca + cb)
                span = np.linalg.norm(ca - cb)
                if span > 40 and np.linalg.norm(ck - mid) <= 0.25 * span:
                    # thin: few k pixels have k on both sides horizontally or vertically
                    aa[k] = (a, b)
        self.aa = aa
        # order remaining (non-AA) classes by depth: outermost first
        layers = [k for k in range(1, m + 1) if k not in aa and counts[k]]
        layers.sort(key=lambda k: depth[k])
        self.order = layers  # merged class ids, outer -> inner
        rank = {k: i + 1 for i, k in enumerate(layers)}
        rank[0] = 0
        self.rank = rank
        self.lab = lab

    def masks(self) -> List[np.ndarray]:
        """Cumulative masks M_i (i = 1..n): coverage of 'rank >= i'."""
        lab = self.lab
        n = len(self.order)
        rank_map = np.zeros(lab.shape, dtype=np.float64)
        for k, r in self.rank.items():
            if k == 0:
                continue
            rank_map[lab == k] = r
        out = []
        for i in range(1, n + 1):
            m = (rank_map >= i).astype(np.float64)
            for k, (a, b) in self.aa.items():
                ra, rb = self.rank.get(a, 0), self.rank.get(b, 0)
                lo, hi = min(ra, rb), max(ra, rb)
                sel = lab == k
                if i <= lo:
                    m[sel] = 1.0
                elif i <= hi:
                    m[sel] = 0.5
                else:
                    m[sel] = 0.0
            out.append(m)
        return out


def _aa_threshold(f: np.ndarray, width: float = 1.0) -> np.ndarray:
    """Coverage of {f >= 0.5} with ~1 output pixel of anti-aliasing."""
    gy, gx = np.gradient(f)
    g = np.sqrt(gx * gx + gy * gy)
    g = np.maximum(g, 1e-3)
    return np.clip(0.5 + (f - 0.5) / (g * width), 0.0, 1.0)


def _contour(labels: np.ndarray, colors: np.ndarray, wy: np.ndarray, wx: np.ndarray,
             kernel: str = "catrom") -> np.ndarray:
    model = LayerModel(labels, colors)
    masks = model.masks()
    oh, ow = wy.shape[0], wx.shape[0]
    if not masks:
        return np.zeros((oh, ow, 4))
    cov = []
    for m in masks:
        f = _resample2d(m, wy, wx)
        cov.append(_aa_threshold(f))
    # enforce nesting and build exclusive coverage
    for i in range(1, len(cov)):
        cov[i] = np.minimum(cov[i], cov[i - 1])
    out = np.zeros((oh, ow, 4))
    for i, k in enumerate(model.order):
        excl = cov[i] - (cov[i + 1] if i + 1 < len(cov) else 0)
        out += excl[..., None] * model.layer_colors[k - 1][None, None, :]
    return out


# ---------------------------------------------------------------------------
# xBR 4x - vectorised port of FFmpeg's vf_xbr.c (Hyllian's algorithm)
# ---------------------------------------------------------------------------

def _yuv(img: np.ndarray) -> np.ndarray:
    r, g, b, a = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.169 * r - 0.331 * g + 0.5 * b
    v = 0.5 * r - 0.419 * g - 0.081 * b
    return np.stack([y, u, v, a], axis=-1)


def _df(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).sum(axis=-1)


# 4x4 blend tables (fraction of the interpolated colour) for the bottom-right
# corner; the other corners are rotations.  Index layout:  0  1  2  3 / 4 ..
_DIA = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, .5], [0, 0, .5, 1]])
_LEFT = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, .25, .75], [.25, .75, 1, 1]])
_UP = np.array([[0, 0, 0, .25], [0, 0, 0, .75], [0, 0, .25, 1], [0, 0, .75, 1]])
_LEFT_UP = np.maximum(_LEFT, _UP)
_WEAK = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, .5]])


def xbr4(img: np.ndarray, eq_threshold: float = 155.0) -> np.ndarray:
    """4x xBR magnification of a premultiplied RGBA float image (0..255)."""
    h, w, _ = img.shape
    P = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode="edge")
    Y = _yuv(P)

    def at(arr, dy, dx):
        return arr[2 + dy:2 + dy + h, 2 + dx:2 + dx + w]

    names = {"A": (-1, -1), "B": (-1, 0), "C": (-1, 1), "D": (0, -1), "E": (0, 0), "F": (0, 1),
             "G": (1, -1), "H": (1, 0), "I": (1, 1), "A1": (-2, -1), "B1": (-2, 0), "C1": (-2, 1),
             "A0": (-1, -2), "D0": (0, -2), "G0": (1, -2), "C4": (-1, 2), "F4": (0, 2), "I4": (1, 2),
             "G5": (2, -1), "H5": (2, 0), "I5": (2, 1)}
    col = {k: at(P, *v) for k, v in names.items()}
    yuv = {k: at(Y, *v) for k, v in names.items()}
    E = col["E"]
    out = np.repeat(np.repeat(E[:, None, :, None, :], 4, axis=1), 4, axis=3)  # (h,4,w,4,4)

    # neighbour renaming per corner, in FFmpeg's processing order
    corners = [
        (0, dict(I="I", H="H", F="F", G="G", C="C", D="D", B="B", F4="F4", I4="I4", H5="H5", I5="I5")),
        (1, dict(I="C", H="F", F="B", G="I", C="A", D="H", B="D", F4="B1", I4="C1", H5="F4", I5="C4")),
        (2, dict(I="A", H="B", F="D", G="C", C="G", D="F", B="H", F4="D0", I4="A0", H5="B1", I5="A1")),
        (3, dict(I="G", H="D", F="H", G="A", C="I", D="B", B="F", F4="H5", I4="G5", H5="D0", I5="G0")),
    ]
    yE = yuv["E"]

    def ne(a, b):  # exact colour inequality
        return (col[a] != col[b]).any(axis=-1)

    for rot, m in corners:
        n = {k: m[k] for k in m}
        n["E"] = "E"
        c = lambda k: col[n[k]]
        y = lambda k: yuv[n[k]]
        df = lambda a, b: _df(y(a), y(b))
        eq = lambda a, b: df(a, b) < eq_threshold
        pre = (c("E") != c("H")).any(-1) & (c("E") != c("F")).any(-1)
        if not pre.any():
            continue
        e = df("E", "C") + df("E", "G") + df("I", "H5") + df("I", "F4") + 4 * df("H", "F")
        i = df("H", "D") + df("H", "I5") + df("F", "I4") + df("F", "B") + 4 * df("E", "I")
        le = pre & (e <= i)
        if not le.any():
            continue
        px = np.where((df("E", "F") <= df("E", "H"))[..., None], c("F"), c("H"))
        strong = le & (e < i) & (
            (~eq("F", "B") & ~eq("F", "C")) | (~eq("H", "D") & ~eq("H", "G"))
            | (eq("E", "I") & ((~eq("F", "F4") & ~eq("F", "I4")) | (~eq("H", "H5") & ~eq("H", "I5"))))
            | eq("E", "G") | eq("E", "C"))
        weak = le & ~strong
        ke = df("F", "G")
        ki = df("H", "C")
        left = strong & (2 * ke <= ki) & (c("E") != c("G")).any(-1) & (c("D") != c("G")).any(-1)
        up = strong & (ke >= 2 * ki) & (c("E") != c("C")).any(-1) & (c("B") != c("C")).any(-1)
        case = np.zeros((h, w), dtype=np.int8)
        case[weak] = 5
        case[strong & ~left & ~up] = 4
        case[strong & up & ~left] = 3
        case[strong & left & ~up] = 2
        case[strong & left & up] = 1
        tables = np.stack([np.zeros((4, 4)), _LEFT_UP, _LEFT, _UP, _DIA, _WEAK])
        tables = np.rot90(tables, k=rot, axes=(1, 2))
        wt = tables[case].transpose(0, 2, 1, 3)[..., None]  # (h, 4, w, 4, 1)
        out += (px[:, None, :, None, :] - out) * wt
    return out.reshape(h * 4, w * 4, 4)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def choose_algorithm(classes: Optional[np.ndarray], class_rgba: Optional[np.ndarray]) -> str:
    """Pick the best algorithm for one bitmap (used by ``auto``).

    Measured on synthetic DVD renders of Korean and Latin text against a true
    1080p rendering: xBR reconstructs hard-edged (3-colour) subtitles best,
    while bitmaps that contain anti-aliasing pixels are reproduced far better
    by the layered contour method, which turns those pixels into sub-pixel
    edge positions instead of grey halos.
    """
    if classes is None or class_rgba is None or len(class_rgba) > 16:
        return "lanczos"
    model = LayerModel(classes, class_rgba)
    return "contour" if model.aa else "xbr"


def upscale_picture(rgba: np.ndarray, classes: Optional[np.ndarray], class_rgba: Optional[np.ndarray],
                    sx: float, sy: float, fx: float, fy: float, algorithm: str = "auto"
                    ) -> Tuple[np.ndarray, int, int]:
    """Upscale one subtitle bitmap.

    ``(fx, fy)`` is the bitmap's top-left corner in *output* pixel units
    (fractional).  Returns (premultiplied float RGBA image, x0, y0) where
    (x0, y0) is the integer output position of the returned image.
    """
    h, w = rgba.shape[:2]
    x0 = int(math.floor(fx))
    y0 = int(math.floor(fy))
    x1 = int(math.ceil(fx + w * sx))
    y1 = int(math.ceil(fy + h * sy))
    ow, oh = max(x1 - x0, 1), max(y1 - y0, 1)
    # input coordinate of output pixel o: (o + x0 - fx) / s  => offset (x0 - fx)/s
    offx = (x0 - fx) / sx
    offy = (y0 - fy) / sy
    palettised = classes is not None and class_rgba is not None and len(class_rgba) <= 16
    if algorithm == "auto":
        algorithm = choose_algorithm(classes, class_rgba)
    if algorithm == "contour" and not palettised:
        algorithm = "lanczos"

    if algorithm == "nearest":
        cx = np.floor((np.arange(ow) + 0.5) / sx + offx).astype(int)
        cy = np.floor((np.arange(oh) + 0.5) / sy + offy).astype(int)
        vx = (cx >= 0) & (cx < w)
        vy = (cy >= 0) & (cy < h)
        out = np.zeros((oh, ow, 4))
        pm = _premul(rgba)
        out[np.ix_(vy, vx)] = pm[np.ix_(cy[vy], cx[vx])]
        return out, x0, y0

    if algorithm == "contour":
        wy = resample_matrix(h, oh, sy, offy, "bspline")
        wx = resample_matrix(w, ow, sx, offx, "bspline")
        return _contour(classes, class_rgba, wy, wx), x0, y0

    if algorithm == "xbr":
        big = xbr4(_premul(rgba))
        wy = resample_matrix(h * 4, oh, sy / 4, offy * 4, "mitchell")
        wx = resample_matrix(w * 4, ow, sx / 4, offx * 4, "mitchell")
        out = _resample2d(big, wy, wx)
        out[..., 3] = np.clip(out[..., 3], 0, 255)
        out[..., :3] = np.clip(out[..., :3], 0, out[..., 3:4])
        return out, x0, y0

    # lanczos
    wy = resample_matrix(h, oh, sy, offy, "lanczos")
    wx = resample_matrix(w, ow, sx, offx, "lanczos")
    out = _resample2d(_premul(rgba), wy, wx)
    out[..., 3] = np.clip(out[..., 3], 0, 255)
    out[..., :3] = np.clip(out[..., :3], 0, out[..., 3:4])
    return out, x0, y0


def unpremultiply(pm: np.ndarray) -> np.ndarray:
    """Premultiplied float RGBA -> straight uint8 RGBA."""
    a = np.clip(pm[..., 3], 0, 255)
    safe = np.where(a > 0, a, 1)[..., None]
    rgb = np.clip(pm[..., :3] * 255.0 / safe, 0, 255)
    out = np.concatenate([rgb, a[..., None]], axis=-1)
    return np.round(out).astype(np.uint8)
