"""Minimal dependency-free PNG writer/reader for RGBA/RGB/gray numpy arrays."""
from __future__ import annotations

import struct
import zlib

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def encode_png(img: np.ndarray, level: int = 6) -> bytes:
    img = np.ascontiguousarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        color_type, ch = 0, 1
        img = img[:, :, None]
    else:
        ch = img.shape[2]
        color_type = {1: 0, 2: 4, 3: 2, 4: 6}[ch]
    h, w = img.shape[:2]
    rows = np.concatenate([np.zeros((h, 1), dtype=np.uint8), img.reshape(h, w * ch)], axis=1)
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(rows.tobytes(), level))
            + _chunk(b"IEND", b""))


def write_png(path: str, img: np.ndarray, level: int = 6) -> None:
    with open(path, "wb") as f:
        f.write(encode_png(img, level))


def _unfilter(raw: bytes, h: int, stride: int, bpp: int) -> np.ndarray:
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    pos = 0
    for y in range(h):
        ft = raw[pos]
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=pos + 1).astype(np.int32)
        pos += stride + 1
        if ft == 0:
            cur = line
        elif ft == 2:
            cur = (line + prev) & 0xFF
        else:
            cur = np.zeros(stride, dtype=np.int32)
            for x in range(stride):
                a = cur[x - bpp] if x >= bpp else 0
                b = prev[x]
                c = prev[x - bpp] if x >= bpp else 0
                if ft == 1:
                    v = a
                elif ft == 3:
                    v = (a + b) >> 1
                else:  # paeth
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    v = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[x] = (line[x] + v) & 0xFF
        out[y] = cur
        prev = cur
    return out


def read_png(path: str) -> np.ndarray:
    """Read 8-bit non-interlaced PNGs (enough for FFmpeg-written images)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos = 8
    idat = bytearray()
    w = h = ct = 0
    plte = None
    while pos < len(data):
        (ln,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + ln]
        pos += 12 + ln
        if tag == b"IHDR":
            w, h, depth, ct, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8 or interlace:
                raise ValueError("unsupported PNG layout")
        elif tag == b"PLTE":
            plte = np.frombuffer(body, dtype=np.uint8).reshape(-1, 3)
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
    ch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ct]
    raw = zlib.decompress(bytes(idat))
    img = _unfilter(raw, h, w * ch, ch).reshape(h, w, ch)
    if ct == 3 and plte is not None:
        img = plte[img[:, :, 0]]
    return img
