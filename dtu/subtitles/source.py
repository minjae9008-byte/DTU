"""Reading bitmap subtitle streams (DVD SPU / PGS) from any FFmpeg input."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..ffmpeg import FFmpeg, FFmpegError, hexdump_to_bytes, run
from . import pgs, spu


@dataclass
class SubPicture:
    """A subtitle bitmap in *source* canvas coordinates.

    ``classes``/``class_rgba`` describe palettised images (DVD: 4 classes,
    PGS: up to 256); ``rgba`` is always filled in.
    """

    start: float
    end: float
    x: int
    y: int
    rgba: np.ndarray  # (h, w, 4) uint8 straight alpha
    classes: Optional[np.ndarray] = None  # (h, w) uint8
    class_rgba: Optional[np.ndarray] = None  # (n, 4) uint8
    forced: bool = False


@dataclass
class SubTrack:
    codec: str
    canvas: Tuple[int, int]
    pictures: List[SubPicture] = field(default_factory=list)
    language: Optional[str] = None
    title: Optional[str] = None
    default: bool = False
    forced_flag: bool = False


def read_packets(ff: FFmpeg, input_args: Sequence[str], stream_index: int,
                 timeout: Optional[float] = 1800) -> Tuple[List[Tuple[float, Optional[float], bytes]], bytes, dict]:
    """Return ([(pts, duration, data)], extradata, stream_json) for one stream."""
    sel = ["-select_streams", str(stream_index)]
    st = ff.probe(input_args, ["-show_streams", "-show_data", *sel])
    streams = st.get("streams") or []
    if not streams:
        raise FFmpegError(f"stream {stream_index} not found")
    sjson = streams[0]
    extradata = hexdump_to_bytes(sjson.get("extradata", "")) if sjson.get("extradata") else b""
    args = [ff.ffprobe, "-hide_banner", "-v", "error", "-of", "json", "-show_packets", "-show_data",
            *sel, *input_args]
    p = run(args, timeout=timeout)
    if p.returncode != 0:
        raise FFmpegError("ffprobe failed reading subtitle packets",
                          p.stderr.decode("utf-8", "replace"))
    pk = json.loads(p.stdout.decode("utf-8", "replace") or "{}").get("packets", [])
    out = []
    for q in pk:
        if "pts_time" not in q or q.get("pts_time") in (None, "N/A"):
            continue
        dur = q.get("duration_time")
        dur_f = float(dur) if dur not in (None, "N/A") else None
        out.append((float(q["pts_time"]), dur_f, hexdump_to_bytes(q.get("data", ""))))
    return out, extradata, sjson


def read_packets_multi(ff: FFmpeg, input_args: Sequence[str], stream_indices: Sequence[int],
                       timeout: Optional[float] = 3600):
    """Read several subtitle streams in ONE pass over the file.

    Returns {index: (packets, extradata, stream_json)}.  A DVD movie often
    carries 4-8 subtitle languages; reading them together avoids demuxing a
    multi-gigabyte VOB set once per track.
    """
    wanted = set(stream_indices)
    st = ff.probe(input_args, ["-show_streams", "-show_data", "-select_streams", "s"])
    streams = {s_["index"]: s_ for s_ in st.get("streams") or [] if s_.get("index") in wanted}
    args = [ff.ffprobe, "-hide_banner", "-v", "error", "-of", "json", "-show_packets", "-show_data",
            "-select_streams", "s", *input_args]
    p = run(args, timeout=timeout)
    if p.returncode != 0:
        raise FFmpegError("ffprobe failed reading subtitle packets", p.stderr.decode("utf-8", "replace"))
    out = {i: ([], hexdump_to_bytes(streams[i].get("extradata", "")) if streams.get(i, {}).get("extradata")
               else b"", streams.get(i, {})) for i in wanted}
    for q in json.loads(p.stdout.decode("utf-8", "replace") or "{}").get("packets", []):
        i = q.get("stream_index")
        if i not in wanted or q.get("pts_time") in (None, "N/A"):
            continue
        dur = q.get("duration_time")
        out[i][0].append((float(q["pts_time"]), float(dur) if dur not in (None, "N/A") else None,
                          hexdump_to_bytes(q.get("data", ""))))
    return out


# ---------------------------------------------------------------------------
# DVD
# ---------------------------------------------------------------------------

def dvd_pictures(packets: List[Tuple[float, Optional[float], bytes]], palette, custom=None,
                 guessed: bool = False, time_offset: float = 0.0) -> List[SubPicture]:
    """Decode SPU packets (reassembling split units) into SubPictures."""
    pics: List[SubPicture] = []
    pending: List[SubPicture] = []  # pictures without a stop time
    buf = b""
    t0 = 0.0
    d0: Optional[float] = None
    for pts, dur, data in packets:
        if buf:
            buf += data
        else:
            buf, t0, d0 = data, pts, dur
        if len(buf) < 2:
            buf = b""
            continue
        size = (buf[0] << 8) | buf[1]
        if len(buf) < size:
            continue
        unit, buf = buf[:size], b""
        try:
            images = spu.decode_spu(unit)
        except spu.SpuError:
            continue
        base = t0 - time_offset
        if d0 is not None and not (0 < d0 < 600):
            d0 = None  # "unknown end" is often stored as 2^32 ms
        # a new unit ends every still-open picture
        for p in pending:
            p.end = max(p.start, base)
        pending = []
        for im in images:
            rgba_pal = spu.image_rgba_palette(im, palette, custom, guessed)
            start = base + im.start
            if im.end is not None and im.end - im.start < 600:  # 745 s = "no stop" sentinel
                end = base + im.end
            elif d0:
                end = base + d0
            else:
                end = start + 10.0  # provisional; clipped by the next unit
            pic = SubPicture(start=start, end=end, x=im.x, y=im.y, rgba=rgba_pal[im.classes],
                             classes=im.classes, class_rgba=rgba_pal, forced=im.forced)
            pic = trim_picture(pic)
            if pic is None:
                continue
            pics.append(pic)
            if (im.end is None or im.end - im.start >= 600) and not d0:
                pending.append(pic)
    # make sure nothing overlaps the next picture's start more than needed
    pics.sort(key=lambda p: p.start)
    return pics


def trim_picture(pic: SubPicture, margin: int = 2) -> Optional[SubPicture]:
    """Crop a picture to its visible (alpha > 0) bounding box plus a margin."""
    alpha = pic.rgba[:, :, 3]
    rows = np.flatnonzero(alpha.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(alpha.any(axis=0))
    h, w = alpha.shape
    y0 = max(int(rows[0]) - margin, 0)
    y1 = min(int(rows[-1]) + 1 + margin, h)
    x0 = max(int(cols[0]) - margin, 0)
    x1 = min(int(cols[-1]) + 1 + margin, w)
    if (y0, x0, y1, x1) == (0, 0, h, w):
        return pic
    return SubPicture(start=pic.start, end=pic.end, x=pic.x + x0, y=pic.y + y0,
                      rgba=pic.rgba[y0:y1, x0:x1],
                      classes=None if pic.classes is None else pic.classes[y0:y1, x0:x1],
                      class_rgba=pic.class_rgba, forced=pic.forced)


# ---------------------------------------------------------------------------
# PGS
# ---------------------------------------------------------------------------

def pgs_pictures(packets: List[Tuple[float, Optional[float], bytes]], time_offset: float = 0.0,
                 canvas_height: int = 1080) -> List[SubPicture]:
    dec = pgs.PgsDecoder()
    dec.height = canvas_height
    for pts, _dur, data in packets:
        # packets are one or more whole segments without the "PG" header
        pos = 0
        n = len(data)
        while pos + 3 <= n:
            seg_type = data[pos]
            size = (data[pos + 1] << 8) | data[pos + 2]
            payload = data[pos + 3:pos + 3 + size]
            pos += 3 + size
            dec.feed(int(round((pts - time_offset) * 90000)), seg_type, payload)
    pics = []
    for start, end, x, y, rgba, forced in pgs.events_from_compositions(dec.events):
        pic = trim_picture(SubPicture(start=start, end=end, x=x, y=y, rgba=rgba, forced=forced))
        if pic is not None:
            pics.append(pic)
    return pics


def load_track(ff: FFmpeg, input_args: Sequence[str], stream_index: int, time_offset: float = 0.0,
               ifo_palette=None, canvas: Optional[Tuple[int, int]] = None, preread=None) -> SubTrack:
    """Load a bitmap subtitle stream from an FFmpeg input.

    ``preread`` may hold ``(packets, extradata, stream_json)`` from
    :func:`read_packets_multi`.
    """
    packets, extradata, sj = preread if preread is not None else read_packets(ff, input_args, stream_index)
    codec = sj.get("codec_name", "")
    tags = sj.get("tags") or {}
    disp = sj.get("disposition") or {}
    lang = tags.get("language")
    title = tags.get("title")
    w = int(sj.get("width") or 0)
    h = int(sj.get("height") or 0)
    if codec == "dvd_subtitle":
        info = spu.parse_idx_header(extradata.decode("latin-1", "replace")) if extradata else \
            {"size": None, "palette": None, "custom": None}
        palette = info.get("palette") or ifo_palette
        if info.get("size"):
            w, h = info["size"]
        if not (w and h):
            w, h = canvas or (720, 480)
        pics = dvd_pictures(packets, palette or [(0, 0, 0)] * 16, info.get("custom"),
                            guessed=palette is None, time_offset=time_offset)
    elif codec == "hdmv_pgs_subtitle":
        if not (w and h):
            w, h = canvas or (1920, 1080)
        pics = pgs_pictures(packets, time_offset, h)
    else:
        raise ValueError(f"unsupported bitmap subtitle codec: {codec}")
    return SubTrack(codec=codec, canvas=(w, h), pictures=pics, language=lang, title=title,
                    default=bool(disp.get("default")), forced_flag=bool(disp.get("forced")))
