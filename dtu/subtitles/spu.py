"""DVD sub-picture unit (SPU, codec ``dvd_subtitle``) decoder.

An SPU is the bitmap subtitle format used on DVD-Video (and in VobSub
``.idx/.sub`` files and Matroska ``S_VOBSUB`` tracks).  Each unit carries a
2-bit-per-pixel run-length encoded, field-interlaced bitmap plus a table of
display-control command sequences (start/stop time, 4 palette indices, 4
alpha values, display rectangle).

Only the syntax needed for real-world discs is implemented; the rarely used
``CHG_COLCON`` (0x07) command is parsed but ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# One SP_DCSQ "date" tick is 1024 periods of the 90 kHz clock.
SPU_TICK = 1024.0 / 90000.0


@dataclass
class SpuImage:
    """A decoded sub-picture: a 2-bit class map and its colour mapping."""

    start: float  # seconds, relative to the SPU's own timestamp
    end: Optional[float]  # seconds, relative; None = no stop command
    x: int
    y: int
    classes: np.ndarray  # (h, w) uint8 with values 0..3
    color_idx: Tuple[int, int, int, int]  # palette index per class
    alpha: Tuple[int, int, int, int]  # 0..15 per class
    forced: bool = False

    @property
    def width(self) -> int:
        return int(self.classes.shape[1])

    @property
    def height(self) -> int:
        return int(self.classes.shape[0])


class SpuError(ValueError):
    pass


def _decode_field(data: bytes, offset: int, end: int, width: int, rows: int) -> np.ndarray:
    """Decode one field (every other line) of 2-bit RLE data."""
    out = np.zeros((rows, width), dtype=np.uint8)
    nib = offset * 2  # nibble position
    nib_end = end * 2

    def get(pos: int) -> int:
        b = data[pos >> 1]
        return (b >> 4) if (pos & 1) == 0 else (b & 0x0F)

    for row in range(rows):
        x = 0
        line = out[row]
        while x < width:
            if nib >= nib_end:
                return out  # truncated data: leave the rest transparent
            v = get(nib)
            nib += 1
            if v < 0x4:
                v = (v << 4) | get(nib)
                nib += 1
                if v < 0x10:
                    v = (v << 4) | get(nib)
                    nib += 1
                    if v < 0x40:
                        v = (v << 4) | get(nib)
                        nib += 1
            run = v >> 2
            color = v & 3
            if run == 0 or x + run > width:
                run = width - x
            if color:
                line[x:x + run] = color
            x += run
        if nib & 1:  # each line starts on a byte boundary
            nib += 1
    return out


def decode_spu(data: bytes) -> List[SpuImage]:
    """Decode a complete SPU packet into zero or more display images.

    Most SPUs yield exactly one image.  A unit may re-define the bitmap or the
    colours in later control sequences; each STA_DSP after a change starts a
    new image.
    """
    if len(data) < 4:
        raise SpuError("SPU too short")
    size = (data[0] << 8) | data[1]
    ctrl = (data[2] << 8) | data[3]
    if size > len(data):
        raise SpuError(f"incomplete SPU ({len(data)} of {size} bytes)")
    if ctrl >= size or ctrl < 4:
        raise SpuError("bad control sequence offset")

    color_idx = [0, 1, 2, 3]
    alpha = [0, 15, 15, 15]
    rect = None  # (x1, x2, y1, y2)
    offsets = None  # (top, bottom)
    forced = False
    images: List[SpuImage] = []
    shown: Optional[dict] = None

    def snapshot(start: float) -> Optional[dict]:
        if rect is None or offsets is None:
            return None
        return dict(start=start, rect=rect, offsets=offsets,
                    color_idx=tuple(color_idx), alpha=tuple(alpha), forced=forced)

    def finish(state: dict, end: Optional[float]):
        x1, x2, y1, y2 = state["rect"]
        w, h = x2 - x1 + 1, y2 - y1 + 1
        if w <= 0 or h <= 0 or w > 4096 or h > 4096:
            return
        top, bottom = state["offsets"]
        img = np.zeros((h, w), dtype=np.uint8)
        top_rows = (h + 1) // 2
        bottom_rows = h // 2
        # The top field ends where the bottom field starts (or at the control
        # table), the bottom field ends at the control table.
        top_end = bottom if bottom > top else ctrl
        bottom_end = ctrl if ctrl > bottom else size
        img[0::2] = _decode_field(data, top, min(top_end, size), w, top_rows)
        if bottom_rows:
            img[1::2] = _decode_field(data, bottom, min(bottom_end, size), w, bottom_rows)
        images.append(SpuImage(start=state["start"], end=end, x=x1, y=y1, classes=img,
                               color_idx=state["color_idx"], alpha=state["alpha"],
                               forced=state["forced"]))

    pos = ctrl
    seen = set()
    while pos + 4 <= size and pos not in seen:
        seen.add(pos)
        date = ((data[pos] << 8) | data[pos + 1]) * SPU_TICK
        next_pos = (data[pos + 2] << 8) | data[pos + 3]
        p = pos + 4
        start_cmd = False
        stop_cmd = False
        changed = False
        while p < size:
            cmd = data[p]
            p += 1
            if cmd == 0x00:  # FSTA_DSP: forced start
                forced = True
                start_cmd = True
            elif cmd == 0x01:  # STA_DSP
                start_cmd = True
            elif cmd == 0x02:  # STP_DSP
                stop_cmd = True
            elif cmd == 0x03:  # SET_COLOR: e2 e1 p b
                if p + 2 > size:
                    break
                color_idx = [data[p + 1] & 0x0F, data[p + 1] >> 4, data[p] & 0x0F, data[p] >> 4]
                p += 2
                changed = True
            elif cmd == 0x04:  # SET_CONTR
                if p + 2 > size:
                    break
                alpha = [data[p + 1] & 0x0F, data[p + 1] >> 4, data[p] & 0x0F, data[p] >> 4]
                p += 2
                changed = True
            elif cmd == 0x05:  # SET_DAAREA
                if p + 6 > size:
                    break
                x1 = (data[p] << 4) | (data[p + 1] >> 4)
                x2 = ((data[p + 1] & 0x0F) << 8) | data[p + 2]
                y1 = (data[p + 3] << 4) | (data[p + 4] >> 4)
                y2 = ((data[p + 4] & 0x0F) << 8) | data[p + 5]
                rect = (x1, x2, y1, y2)
                p += 6
                changed = True
            elif cmd == 0x06:  # SET_DSPXA
                if p + 4 > size:
                    break
                offsets = ((data[p] << 8) | data[p + 1], (data[p + 2] << 8) | data[p + 3])
                p += 4
                changed = True
            elif cmd == 0x07:  # CHG_COLCON: skip its parameter block
                if p + 2 > size:
                    break
                p += (data[p] << 8) | data[p + 1]
            elif cmd == 0xFF:  # CMD_END
                break
            else:  # unknown command: give up on this sequence
                break
        if stop_cmd and shown is not None:
            finish(shown, date)
            shown = None
        if start_cmd or (changed and shown is not None):
            if shown is not None:
                finish(shown, date)
            shown = snapshot(date)
        if next_pos == pos or next_pos < ctrl:
            break
        pos = next_pos
    if shown is not None:
        finish(shown, None)
    return images


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

def parse_idx_header(text: str) -> dict:
    """Parse the header of a VobSub .idx (or the S_VOBSUB CodecPrivate).

    Returns dict with ``size`` (w, h) or None, ``palette`` (16 RGB tuples) or
    None, ``custom`` (4 RGB tuples) or None and ``langs`` list.
    """
    info = {"size": None, "palette": None, "custom": None, "langs": []}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "size":
            try:
                w, h = value.lower().split("x")
                info["size"] = (int(w), int(h))
            except ValueError:
                pass
        elif key == "palette":
            cols = [c.strip() for c in value.split(",") if c.strip()]
            pal = []
            for c in cols[:16]:
                try:
                    v = int(c, 16)
                except ValueError:
                    continue
                pal.append(((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF))
            if pal:
                pal += [(0, 0, 0)] * (16 - len(pal))
                info["palette"] = pal
        elif key == "custom colors":
            # custom colors: ON, tridx: 1000, colors: 000000, ffffff, ...
            if value.upper().startswith("ON") and "colors:" in value:
                cols = value.split("colors:", 1)[1].split(",")
                try:
                    info["custom"] = [((int(c, 16) >> 16) & 0xFF, (int(c, 16) >> 8) & 0xFF,
                                       int(c, 16) & 0xFF) for c in cols[:4]]
                except ValueError:
                    pass
        elif key == "id":
            lang = value.split(",")[0].strip()
            info["langs"].append(lang)
    return info


def ycrcb_to_rgb(y: int, cr: int, cb: int) -> Tuple[int, int, int]:
    """BT.601 limited-range YCbCr -> RGB (used by DVD IFO palettes)."""
    yy = 1.164383 * (y - 16)
    r = yy + 1.596027 * (cr - 128)
    g = yy - 0.391762 * (cb - 128) - 0.812968 * (cr - 128)
    b = yy + 2.017232 * (cb - 128)
    return tuple(int(max(0, min(255, round(v)))) for v in (r, g, b))  # type: ignore


def guess_palette() -> List[Tuple[int, int, int]]:
    """Fallback palette when a stream has none (e.g. bare VOB files).

    Real discs almost always use class 1 = text fill, class 2 = outline and
    class 3 = anti-alias.  We build a palette in which *indices* 0..15 are
    varied so that each class resolves to a sensible grey level no matter
    which index the SPU picks: even indices light, odd indices dark.
    """
    pal = []
    for i in range(16):
        if i % 4 == 0:
            pal.append((16, 16, 16))
        elif i % 4 == 1:
            pal.append((235, 235, 235))
        elif i % 4 == 2:
            pal.append((16, 16, 16))
        else:
            pal.append((128, 128, 128))
    return pal


def image_rgba_palette(img: SpuImage, palette: Sequence[Tuple[int, int, int]],
                       custom: Optional[Sequence[Tuple[int, int, int]]] = None,
                       guessed: bool = False) -> np.ndarray:
    """Return the (4, 4) uint8 RGBA colour of each SPU class."""
    out = np.zeros((4, 4), dtype=np.uint8)
    for k in range(4):
        if custom is not None:
            rgb = custom[k]
        elif guessed:
            # classes: 0 background, 1 fill (white), 2 outline (black), 3 AA (grey)
            rgb = [(16, 16, 16), (235, 235, 235), (16, 16, 16), (128, 128, 128)][k]
        else:
            rgb = palette[img.color_idx[k] & 15]
        a = img.alpha[k] * 17
        out[k] = (rgb[0], rgb[1], rgb[2], a)
    return out
