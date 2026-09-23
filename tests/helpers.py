"""Test helpers: a minimal DVD SPU encoder and synthetic subtitle bitmaps."""
from __future__ import annotations

import struct
from typing import List, Tuple

import numpy as np


def _rle_line(row: np.ndarray) -> List[int]:
    nib: List[int] = []
    x, w = 0, len(row)
    while x < w:
        c = int(row[x])
        run = 1
        while x + run < w and row[x + run] == c and run < 255:
            run += 1
        v = (run << 2) | c
        if run < 4:
            nib += [v]
        elif run < 16:
            nib += [v >> 4, v & 15]
        elif run < 64:
            nib += [v >> 8, (v >> 4) & 15, v & 15]
        else:
            nib += [v >> 12, (v >> 8) & 15, (v >> 4) & 15, v & 15]
        x += run
    if len(nib) % 2:
        nib.append(0)
    return nib


def _pack(nibbles: List[int]) -> bytes:
    return bytes((nibbles[i] << 4) | nibbles[i + 1] for i in range(0, len(nibbles), 2))


def encode_spu(classes: np.ndarray, x: int, y: int, colors=(0, 1, 2, 3), alpha=(0, 15, 15, 15),
               duration_ticks: int = 200, forced: bool = False) -> bytes:
    """Encode a 2-bit class image as a DVD sub-picture unit."""
    h, w = classes.shape
    top = b"".join(_pack(_rle_line(classes[r])) for r in range(0, h, 2))
    bottom = b"".join(_pack(_rle_line(classes[r])) for r in range(1, h, 2))
    top_off = 4
    bot_off = top_off + len(top)
    ctrl = bot_off + len(bottom)
    x2, y2 = x + w - 1, y + h - 1
    area = bytes([(x >> 4) & 0xFF, ((x & 15) << 4) | (x2 >> 8), x2 & 0xFF,
                  (y >> 4) & 0xFF, ((y & 15) << 4) | (y2 >> 8), y2 & 0xFF])
    b, p, e1, e2 = colors
    ab, ap, ae1, ae2 = alpha
    seq1_cmds = (b"\x06" + struct.pack(">HH", top_off, bot_off)
                 + b"\x03" + bytes([(e2 << 4) | e1, (p << 4) | b])
                 + b"\x04" + bytes([(ae2 << 4) | ae1, (ap << 4) | ab])
                 + b"\x05" + area + (b"\x00" if forced else b"\x01") + b"\xff")
    seq2_off = ctrl + 4 + len(seq1_cmds)
    seq1 = struct.pack(">HH", 0, seq2_off) + seq1_cmds
    seq2 = struct.pack(">HH", duration_ticks, seq2_off) + b"\x02\xff"
    body = top + bottom + seq1 + seq2
    size = 4 + len(body)
    return struct.pack(">HH", size, ctrl) + body


def glyph_classes(aa: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """A small 'O'-like glyph: white fill, black 1-px outline, optional grey AA ring."""
    h, w = 24, 40
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2 + 0.5) / 1.3) ** 2 + (yy - h / 2 + 0.5) ** 2)
    cls = np.zeros((h, w), np.uint8)
    cls[(r <= 9.5)] = 2          # outline disc
    cls[(r <= 8.0) & (r > 4.0)] = 1  # fill ring
    cls[(r <= 4.0)] = 2          # inner outline
    cls[(r <= 3.0)] = 0          # hole
    if aa:
        ring = (np.abs(r - 8.0) < 0.5)
        cls[ring] = 3
    pal = np.array([[0, 0, 0, 0], [235, 235, 235, 255], [16, 16, 16, 255], [126, 126, 126, 255]], np.uint8)
    return cls, pal
