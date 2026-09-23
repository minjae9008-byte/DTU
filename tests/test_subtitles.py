import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import encode_spu, glyph_classes  # noqa: E402

from dtu.subtitles import pgs, spu
from dtu.subtitles.imageio import encode_png, read_png, write_png
from dtu.subtitles.pipeline import Geometry, _convert_one, convert_pictures
from dtu.subtitles.quantize import quantize, trim_indexed
from dtu.subtitles.source import SubPicture, dvd_pictures, trim_picture
from dtu.subtitles.upscale import (LayerModel, choose_algorithm, resample_matrix, upscale_picture,
                                   unpremultiply, xbr4)


# ---------------------------------------------------------------------------
# SPU
# ---------------------------------------------------------------------------

def test_spu_roundtrip_exact():
    rng = np.random.default_rng(1)
    cls = rng.integers(0, 4, size=(37, 91)).astype(np.uint8)
    cls[:, 60:] = 1  # long runs too
    data = encode_spu(cls, 100, 380, colors=(0, 5, 9, 12), alpha=(0, 15, 11, 7), duration_ticks=300)
    imgs = spu.decode_spu(data)
    assert len(imgs) == 1
    im = imgs[0]
    assert (im.x, im.y, im.width, im.height) == (100, 380, 91, 37)
    assert np.array_equal(im.classes, cls)
    assert im.color_idx == (0, 5, 9, 12)
    assert im.alpha == (0, 15, 11, 7)
    assert im.start == 0.0
    assert abs(im.end - 300 * 1024 / 90000) < 1e-9
    assert not im.forced


def test_spu_forced_and_truncated():
    cls, _ = glyph_classes()
    data = encode_spu(cls, 10, 20, forced=True)
    assert spu.decode_spu(data)[0].forced
    with pytest.raises(spu.SpuError):
        spu.decode_spu(data[: len(data) // 2])


def test_parse_idx_header():
    text = ("# VobSub index file, v7\nsize: 720x576\npalette: 000000, ffffff, 808080, ff0000, "
            "00ff00, 0000ff, 111111, 222222, 333333, 444444, 555555, 666666, 777777, 888888, 999999, aaaaaa\n"
            "custom colors: OFF, tridx: 0000, colors: 000000, 000000, 000000, 000000\nid: ko, index: 0\n")
    info = spu.parse_idx_header(text)
    assert info["size"] == (720, 576)
    assert info["palette"][1] == (255, 255, 255)
    assert info["palette"][3] == (255, 0, 0)
    assert info["custom"] is None
    assert info["langs"] == ["ko"]


def test_dvd_pictures_timing_and_reassembly():
    cls, _ = glyph_classes()
    pal = [(0, 0, 0), (235, 235, 235), (16, 16, 16), (126, 126, 126)] + [(0, 0, 0)] * 12
    a = encode_spu(cls, 50, 400, duration_ticks=176)  # ~2 s
    b = encode_spu(cls, 60, 400, duration_ticks=0)
    # second unit split over two packets, with a 0.5 s stream start offset
    packets = [(10.5, None, a), (13.5, None, b[:30]), (13.6, None, b[30:])]
    pics = dvd_pictures(packets, pal, time_offset=0.5)
    assert len(pics) == 2
    assert abs(pics[0].start - 10.0) < 1e-9 and abs(pics[0].end - (10.0 + 176 * 1024 / 90000)) < 1e-6
    assert abs(pics[1].start - 13.0) < 1e-9
    # trimmed to visible pixels + 2 px margin
    assert pics[0].rgba.shape[0] <= cls.shape[0] + 4


# ---------------------------------------------------------------------------
# PGS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(1, 1), (7, 300), (120, 1900)])
def test_pgs_rle_roundtrip(shape):
    rng = np.random.default_rng(2)
    idx = rng.integers(0, 6, size=shape).astype(np.uint8)
    idx[:, : shape[1] // 3] = 0
    if shape[1] > 100:
        idx[:, 50:90] = 200  # long coloured run
    assert np.array_equal(pgs.rle_decode(pgs.rle_encode(idx), shape[1], shape[0]), idx)


def test_sup_roundtrip(tmp_path):
    cls, pal = glyph_classes()
    ev1 = pgs.PgsEvent(1.0, 3.0, 100, 900, cls, pal, forced=False)
    ev2 = pgs.PgsEvent(2.5, 6.0, 200, 950, cls, pal, forced=True)  # overlaps ev1
    path = str(tmp_path / "t.sup")
    n = pgs.write_sup(path, [ev2, ev1], 1920, 1080)
    assert n == 2
    w, h, events = pgs.read_sup(path)
    assert (w, h) == (1920, 1080)
    assert len(events) == 2
    (s1, e1, x1, y1, rgba1, f1), (s2, e2, x2, y2, rgba2, f2) = events
    assert abs(s1 - 1.0) < 1e-4 and abs(e1 - 2.5) < 1e-4  # clipped at the next start
    assert (x1, y1) == (100, 900) and not f1 and f2
    # colours survive the YCbCr palette within rounding
    ref = pal[cls]
    assert np.abs(rgba1.astype(int) - ref.astype(int)).max() <= 3


def test_sup_is_anchored_at_zero(tmp_path):
    cls, pal = glyph_classes()
    path = str(tmp_path / "a.sup")
    pgs.write_sup(path, [pgs.PgsEvent(42.0, 44.0, 0, 0, cls, pal)], 1280, 720)
    first_pts = next(pgs.iter_segments(open(path, "rb").read()))[0]
    assert first_pts == 0  # FFmpeg shifts every input by its first timestamp


def test_ods_fragmentation(tmp_path):
    rng = np.random.default_rng(3)
    idx = rng.integers(1, 255, size=(400, 1900)).astype(np.uint8)  # incompressible: > 64 KiB RLE
    pal = np.concatenate([np.zeros((1, 4), np.uint8), rng.integers(0, 256, size=(255, 4)).astype(np.uint8)])
    pal[1:, 3] = 255
    path = str(tmp_path / "big.sup")
    pgs.write_sup(path, [pgs.PgsEvent(0.0, 1.0, 0, 0, idx, pal)], 1920, 1080)
    _, _, events = pgs.read_sup(path)
    assert events and events[0][4].shape[:2] == (400, 1900)


# ---------------------------------------------------------------------------
# quantiser / png
# ---------------------------------------------------------------------------

def test_quantize_lossless_when_few_colours():
    cls, pal = glyph_classes()
    pm = pal[cls].astype(np.float64)
    pm[..., :3] *= pm[..., 3:4] / 255
    idx, qpal = quantize(pm)
    back = qpal[idx]
    assert np.array_equal(back[..., 3], pal[cls][..., 3])
    assert np.abs(back[..., :3].astype(int) - pal[cls][..., :3].astype(int))[pal[cls][..., 3] > 0].max() <= 1


def test_quantize_many_colours():
    rng = np.random.default_rng(4)
    pm = rng.uniform(0, 255, size=(64, 64, 4))
    pm[..., :3] = np.minimum(pm[..., :3], pm[..., 3:4])
    idx, pal = quantize(pm)
    assert len(pal) <= 256 and pal[0, 3] == 0
    err = np.abs(pal[idx][..., 3].astype(float) - np.round(pm[..., 3]))
    assert err.mean() < 12


def test_trim_indexed():
    a = np.zeros((10, 10), np.uint8)
    a[3:5, 4:8] = 2
    t, x, y = trim_indexed(a)
    assert t.shape == (2, 4) and (x, y) == (4, 3)


def test_png_roundtrip(tmp_path):
    img = (np.random.default_rng(5).integers(0, 255, size=(17, 23, 4))).astype(np.uint8)
    p = str(tmp_path / "x.png")
    write_png(p, img)
    assert np.array_equal(read_png(p), img)
    assert encode_png(img[..., :3])[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# upscaling
# ---------------------------------------------------------------------------

def test_resample_matrix_partition_of_unity():
    m = resample_matrix(50, 133, 133 / 50, 0.0, "catrom")
    rows = m.sum(axis=1)
    assert np.allclose(rows[10:-10], 1.0, atol=1e-9)


def test_xbr_keeps_flat_areas_and_scales_by_4():
    img = np.zeros((6, 7, 4))
    img[..., :] = (40, 50, 60, 255)
    out = xbr4(img)
    assert out.shape == (24, 28, 4)
    assert np.allclose(out, img[0, 0])


def test_xbr_smooths_diagonal():
    img = np.zeros((8, 8, 4))
    for i in range(8):
        img[i, : i + 1] = (255, 255, 255, 255)
    out = xbr4(img)
    # a pure staircase would only contain 0/255; xBR blends along the diagonal
    vals = np.unique(np.round(out[..., 3]))
    assert len(vals) > 2


def test_layer_model_detects_antialiasing():
    cls, pal = glyph_classes(aa=True)
    model = LayerModel(cls, pal)
    assert model.aa, "grey ring between white fill and black outline is anti-aliasing"
    cls2, pal2 = glyph_classes(aa=False)
    assert not LayerModel(cls2, pal2).aa


def test_choose_algorithm():
    cls, pal = glyph_classes(aa=True)
    assert choose_algorithm(cls, pal) == "contour"
    cls2, pal2 = glyph_classes(aa=False)
    assert choose_algorithm(cls2, pal2) == "xbr"
    assert choose_algorithm(None, None) == "lanczos"


@pytest.mark.parametrize("algo", ["auto", "contour", "xbr", "lanczos", "nearest"])
def test_upscale_picture_geometry_and_colours(algo):
    cls, pal = glyph_classes(aa=True)
    rgba = pal[cls]
    sx, sy = 1920 / 720, 1080 / 480
    pm, x0, y0 = upscale_picture(rgba, cls, pal, sx, sy, 100 * sx, 400 * sy, algo)
    h, w = cls.shape
    assert abs(pm.shape[1] - w * sx) <= 2 and abs(pm.shape[0] - h * sy) <= 2
    assert (x0, y0) == (int(100 * sx), int(400 * sy))
    straight = unpremultiply(pm)
    opaque = straight[..., 3] > 250
    assert opaque.any()
    # every opaque pixel is (close to) a mix of the white fill and black outline
    rgb = straight[opaque][:, :3].astype(int)
    assert np.all(np.abs(rgb[:, 0] - rgb[:, 1]) <= 12)


def test_convert_one_moves_letterbox_subtitles_inside_frame():
    cls, pal = glyph_classes(aa=False)
    # subtitle in the bottom black bar of a letterboxed frame (picture rows 60..420)
    pic = SubPicture(0.0, 1.0, 300, 440, pal[cls], cls, pal)
    geom = Geometry((0, 60, 720, 360), 1920, 800)
    r = _convert_one((pic, geom, "auto"))
    assert r is not None
    _s, _e, x0, y0, idx, _pal, _f = r
    assert 0 <= y0 and y0 + idx.shape[0] <= 800 - int(800 * 0.045) + 1


def test_convert_pictures_sequential():
    cls, pal = glyph_classes(aa=True)
    pics = [SubPicture(float(i), float(i) + 0.9, 100, 400, pal[cls], cls, pal) for i in range(3)]
    events = convert_pictures(pics, Geometry((0, 0, 720, 480), 1920, 1080), workers=1)
    assert len(events) == 3 and all(e.indices.size for e in events)
