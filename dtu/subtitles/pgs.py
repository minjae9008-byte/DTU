"""HDMV Presentation Graphic Stream (PGS, ``.sup``) writer and reader.

PGS is the bitmap subtitle format of Blu-ray.  Unlike DVD subtitles it allows
256 palette entries with 8-bit alpha and any size up to the video canvas, so
it is the natural target for upscaled DVD subtitles: Matroska supports it and
practically every player (mpv, VLC, MPC-HC, Kodi, Plex, Jellyfin, most TVs)
renders it.

FFmpeg can decode PGS but has no encoder, hence this module.
"""
from __future__ import annotations

import struct
import zlib  # noqa: F401  (kept for symmetry with other readers)
from dataclasses import dataclass
from typing import BinaryIO, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SEG_PDS = 0x14
SEG_ODS = 0x15
SEG_PCS = 0x16
SEG_WDS = 0x17
SEG_END = 0x80

MAX_SEGMENT = 0xFFFF


@dataclass
class PgsEvent:
    """One subtitle picture in output-canvas coordinates."""

    start: float
    end: float
    x: int
    y: int
    indices: np.ndarray  # (h, w) uint8 palette indices, 0 = transparent
    palette: np.ndarray  # (n, 4) uint8 straight RGBA, n <= 256
    forced: bool = False


# ---------------------------------------------------------------------------
# colour conversion
# ---------------------------------------------------------------------------

_KR_KB = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114)}


def rgb_to_ycbcr(rgb: np.ndarray, matrix: str = "bt709") -> np.ndarray:
    """(n, 3) uint8 full-range RGB -> (n, 3) uint8 limited-range Y, Cb, Cr."""
    kr, kb = _KR_KB[matrix]
    kg = 1.0 - kr - kb
    c = rgb.astype(np.float64) / 255.0
    y = kr * c[:, 0] + kg * c[:, 1] + kb * c[:, 2]
    cb = (c[:, 2] - y) / (2 * (1 - kb))
    cr = (c[:, 0] - y) / (2 * (1 - kr))
    out = np.stack([16 + 219 * y, 128 + 224 * cb, 128 + 224 * cr], axis=1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def ycbcr_to_rgb(ycc: np.ndarray, matrix: str = "bt709") -> np.ndarray:
    kr, kb = _KR_KB[matrix]
    kg = 1.0 - kr - kb
    y = (ycc[:, 0].astype(np.float64) - 16) / 219
    cb = (ycc[:, 1].astype(np.float64) - 128) / 224
    cr = (ycc[:, 2].astype(np.float64) - 128) / 224
    r = y + 2 * (1 - kr) * cr
    b = y + 2 * (1 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    out = np.stack([r, g, b], axis=1) * 255
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def matrix_for_height(height: int) -> str:
    """Blu-ray/FFmpeg convention: SD canvases use BT.601, HD uses BT.709."""
    return "bt601" if height <= 576 else "bt709"


# ---------------------------------------------------------------------------
# RLE
# ---------------------------------------------------------------------------

def rle_encode(indices: np.ndarray) -> bytes:
    """PGS run-length encoding of an index image."""
    h, w = indices.shape
    out = bytearray()
    for row in indices:
        # run boundaries
        change = np.flatnonzero(np.diff(row)) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [w]))
        for s, e in zip(starts.tolist(), ends.tolist()):
            color = int(row[s])
            run = e - s
            while run > 0:
                n = min(run, 16383)
                if color == 0:
                    if n < 64:
                        out += bytes((0, n))
                    else:
                        out += bytes((0, 0x40 | (n >> 8), n & 0xFF))
                else:
                    if n < 3:
                        out += bytes((color,)) * n
                    elif n < 64:
                        out += bytes((0, 0x80 | n, color))
                    else:
                        out += bytes((0, 0xC0 | (n >> 8), n & 0xFF, color))
                run -= n
        out += b"\x00\x00"
    return bytes(out)


def rle_decode(data: bytes, width: int, height: int) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.uint8)
    x = y = 0
    i = 0
    n = len(data)
    while i < n and y < height:
        b = data[i]
        i += 1
        if b != 0:
            if x < width:
                out[y, x] = b
            x += 1
            continue
        if i >= n:
            break
        f = data[i]
        i += 1
        if f == 0:  # end of line
            x = 0
            y += 1
            continue
        flag = f >> 6
        run = f & 0x3F
        if flag & 1:  # long run
            run = (run << 8) | data[i]
            i += 1
        color = 0
        if flag & 2:
            color = data[i]
            i += 1
        if color:
            out[y, x:x + run] = color
        x += run
    return out


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------

def _segment(seg_type: int, pts: int, payload: bytes, dts: int = 0) -> bytes:
    if len(payload) > MAX_SEGMENT:
        raise ValueError("PGS segment too large")
    return (b"PG" + struct.pack(">IIBH", pts & 0xFFFFFFFF, dts & 0xFFFFFFFF, seg_type, len(payload))
            + payload)


class PgsWriter:
    """Serialise :class:`PgsEvent` objects to a ``.sup`` stream."""

    def __init__(self, fp: BinaryIO, width: int, height: int, fps_code: int = 0x10):
        self.fp = fp
        self.width = width
        self.height = height
        self.fps_code = fps_code
        self.comp_num = 0
        self.matrix = matrix_for_height(height)

    def _pcs(self, pts: int, state: int, objects: Sequence[Tuple[int, int, int, bool]],
             palette_update: bool = False) -> bytes:
        body = struct.pack(">HHBHBBBB", self.width, self.height, self.fps_code,
                           self.comp_num & 0xFFFF, state, 0x80 if palette_update else 0, 0,
                           len(objects))
        for obj_id, x, y, forced in objects:
            body += struct.pack(">HBBHH", obj_id, 0, 0x40 if forced else 0, x, y)
        self.comp_num += 1
        return _segment(SEG_PCS, pts, body)

    @staticmethod
    def _wds(pts: int, x: int, y: int, w: int, h: int) -> bytes:
        return _segment(SEG_WDS, pts, struct.pack(">BBHHHH", 1, 0, x, y, w, h))

    def _pds(self, pts: int, palette: np.ndarray) -> bytes:
        ycc = rgb_to_ycbcr(palette[:, :3], self.matrix)
        body = bytearray(struct.pack(">BB", 0, 0))
        for i in range(len(palette)):
            y, cb, cr = (int(v) for v in ycc[i])
            body += bytes((i, y, cr, cb, int(palette[i, 3])))
        return _segment(SEG_PDS, pts, bytes(body))

    @staticmethod
    def _ods(pts: int, obj_id: int, w: int, h: int, rle: bytes) -> bytes:
        out = bytearray()
        total = len(rle) + 4
        first_cap = MAX_SEGMENT - 11
        chunks = [rle[:first_cap]]
        rest = rle[first_cap:]
        cap = MAX_SEGMENT - 4
        while rest:
            chunks.append(rest[:cap])
            rest = rest[cap:]
        for i, chunk in enumerate(chunks):
            flag = (0x80 if i == 0 else 0) | (0x40 if i == len(chunks) - 1 else 0)
            if i == 0:
                body = struct.pack(">HBB", obj_id, 0, flag) + total.to_bytes(3, "big") + struct.pack(">HH", w, h)
            else:
                body = struct.pack(">HBB", obj_id, 0, flag)
            out += _segment(SEG_ODS, pts, body + chunk)
        return bytes(out)

    def write_event(self, ev: PgsEvent, next_start: Optional[float] = None):
        h, w = ev.indices.shape
        if w == 0 or h == 0:
            return
        # clamp into the canvas (PGS objects must lie inside it)
        x = int(min(max(ev.x, 0), max(self.width - w, 0)))
        y = int(min(max(ev.y, 0), max(self.height - h, 0)))
        w = min(w, self.width)
        h = min(h, self.height)
        indices = ev.indices[:h, :w]
        start = int(round(ev.start * 90000))
        end = int(round(ev.end * 90000))
        if end <= start:
            end = start + 1
        out = bytearray()
        out += self._pcs(start, 0x80, [(0, x, y, ev.forced)])
        out += self._wds(start, x, y, w, h)
        out += self._pds(start, ev.palette)
        out += self._ods(start, 0, w, h, rle_encode(indices))
        out += _segment(SEG_END, start, b"")
        # clear the screen unless the next subtitle replaces it anyway
        if next_start is None or next_start * 90000 > end:
            out += self._pcs(end, 0x00, [])
            out += self._wds(end, x, y, w, h)
            out += _segment(SEG_END, end, b"")
        self.fp.write(bytes(out))

    def write_blank(self, t: float = 0.0):
        """Empty epoch (used to anchor the stream at t=0)."""
        pts = int(round(t * 90000))
        out = self._pcs(pts, 0x80, [])
        out += self._wds(pts, 0, 0, 1, 1)
        out += _segment(SEG_END, pts, b"")
        self.fp.write(out)


def write_sup(path: str, events: Iterable[PgsEvent], width: int, height: int,
              anchor_zero: bool = True) -> int:
    """Write events (sorted by start) to ``path``; returns number written."""
    evs = sorted(events, key=lambda e: e.start)
    n = 0
    with open(path, "wb") as fp:
        wr = PgsWriter(fp, width, height)
        if anchor_zero and (not evs or evs[0].start > 0.001):
            wr.write_blank(0.0)
        for i, ev in enumerate(evs):
            nxt = evs[i + 1].start if i + 1 < len(evs) else None
            if nxt is not None and ev.end > nxt:
                ev = PgsEvent(ev.start, nxt, ev.x, ev.y, ev.indices, ev.palette, ev.forced)
            if ev.end - ev.start < 0.001:
                continue
            wr.write_event(ev, nxt)
            n += 1
    return n


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------

@dataclass
class _Obj:
    width: int = 0
    height: int = 0
    data: bytearray = None  # type: ignore


def iter_segments(buf: bytes):
    """Yield (pts, seg_type, payload) from a .sup byte string."""
    pos = 0
    n = len(buf)
    while pos + 13 <= n:
        if buf[pos:pos + 2] != b"PG":
            # resync
            nxt = buf.find(b"PG", pos + 1)
            if nxt < 0:
                break
            pos = nxt
            continue
        pts, _dts, seg_type, size = struct.unpack(">IIBH", buf[pos + 2:pos + 13])
        payload = buf[pos + 13:pos + 13 + size]
        pos += 13 + size
        yield pts, seg_type, payload


class PgsDecoder:
    """Stateful PGS decoder producing composited RGBA events."""

    def __init__(self):
        self.width = 0
        self.height = 0
        self.palettes: dict = {}
        self.objects: dict = {}
        self.events: List[Tuple[float, Optional[np.ndarray], int, int, bool]] = []
        self._pending = None  # composition waiting for END

    def feed(self, pts: int, seg_type: int, p: bytes):
        if seg_type == SEG_PCS and len(p) >= 11:
            w, h, _fps, _num, state, _pal_upd, pal_id, nobj = struct.unpack(">HHBHBBBB", p[:11])
            self.width, self.height = w, h
            if state & 0x80:  # epoch start: reset
                self.objects = {}
                self.palettes = {}
            objs = []
            off = 11
            for _ in range(nobj):
                if off + 8 > len(p):
                    break
                oid, _wid, flags, x, y = struct.unpack(">HBBHH", p[off:off + 8])
                off += 8
                crop = None
                if flags & 0x80 and off + 8 <= len(p):
                    crop = struct.unpack(">HHHH", p[off:off + 8])
                    off += 8
                objs.append((oid, x, y, bool(flags & 0x40), crop))
            self._pending = (pts, pal_id, objs)
        elif seg_type == SEG_PDS and len(p) >= 2:
            pid = p[0]
            pal = self.palettes.setdefault(pid, np.zeros((256, 4), dtype=np.uint8))
            ycc = []
            idx = []
            alphas = []
            for off in range(2, len(p) - 4, 5):
                i, y, cr, cb, a = p[off:off + 5]
                idx.append(i)
                ycc.append((y, cb, cr))
                alphas.append(a)
            if idx:
                matrix = matrix_for_height(self.height or 1080)
                rgb = ycbcr_to_rgb(np.array(ycc, dtype=np.uint8), matrix)
                pal[idx, :3] = rgb
                pal[idx, 3] = alphas
        elif seg_type == SEG_ODS and len(p) >= 4:
            oid = struct.unpack(">H", p[:2])[0]
            flag = p[3]
            if flag & 0x80:
                if len(p) < 11:
                    return
                w, h = struct.unpack(">HH", p[7:11])
                self.objects[oid] = _Obj(w, h, bytearray(p[11:]))
            else:
                o = self.objects.get(oid)
                if o is not None:
                    o.data += p[4:]
        elif seg_type == SEG_END and self._pending is not None:
            pts0, pal_id, objs = self._pending
            self._pending = None
            self.events.append(self._compose(pts0, pal_id, objs))

    def _compose(self, pts, pal_id, objs):
        if not objs:
            return (pts / 90000.0, None, 0, 0, False)
        pal = self.palettes.get(pal_id)
        if pal is None:
            pal = np.zeros((256, 4), dtype=np.uint8)
        layers = []
        forced = False
        for oid, x, y, f, crop in objs:
            o = self.objects.get(oid)
            if o is None or o.width == 0:
                continue
            idx = rle_decode(bytes(o.data), o.width, o.height)
            if crop:
                cx, cy, cw, ch = crop
                idx = idx[cy:cy + ch, cx:cx + cw]
            layers.append((x, y, idx))
            forced |= f
        if not layers:
            return (pts / 90000.0, None, 0, 0, False)
        x0 = min(l[0] for l in layers)
        y0 = min(l[1] for l in layers)
        x1 = max(l[0] + l[2].shape[1] for l in layers)
        y1 = max(l[1] + l[2].shape[0] for l in layers)
        rgba = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
        for x, y, idx in layers:
            h, w = idx.shape
            rgba[y - y0:y - y0 + h, x - x0:x - x0 + w] = pal[idx]
        return (pts / 90000.0, rgba, x0, y0, forced)


def read_sup(path_or_bytes) -> Tuple[int, int, List[Tuple[float, float, int, int, np.ndarray, bool]]]:
    """Decode a .sup file into (width, height, [(start, end, x, y, rgba, forced)])."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        buf = bytes(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            buf = f.read()
    dec = PgsDecoder()
    for pts, t, p in iter_segments(buf):
        dec.feed(pts, t, p)
    return dec.width, dec.height, events_from_compositions(dec.events)


def events_from_compositions(comps) -> List[Tuple[float, float, int, int, np.ndarray, bool]]:
    """Turn a list of (time, rgba|None, x, y, forced) display states into events."""
    out = []
    for i, (t, rgba, x, y, forced) in enumerate(comps):
        if rgba is None:
            continue
        end = comps[i + 1][0] if i + 1 < len(comps) else t + 5.0
        if end > t:
            out.append((t, end, x, y, rgba, forced))
    return out
