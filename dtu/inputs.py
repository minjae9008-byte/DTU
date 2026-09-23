"""Input resolution: single files, VOB sets, VIDEO_TS folders and ISO images.

Also parses DVD ``.IFO`` files for the sub-picture palette and the audio /
subtitle language codes, which are not stored inside VOB files.
"""
from __future__ import annotations

import glob
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .subtitles.spu import ycrcb_to_rgb

VIDEO_EXTS = {".mkv", ".mk3d", ".vob", ".mpg", ".mpeg", ".m2v", ".mp4", ".m4v", ".mov", ".avi",
              ".ts", ".m2ts", ".mts", ".wmv", ".webm", ".ogm", ".divx", ".iso", ".evo"}


@dataclass
class InputSpec:
    """How to open a source with FFmpeg."""

    display: str
    paths: List[str]
    fmt: Optional[str] = None  # forced demuxer, e.g. "dvdvideo"
    options: Dict[str, str] = field(default_factory=dict)
    ifo: Optional[str] = None  # VTS_xx_0.IFO with palette / languages
    external_subs: List[str] = field(default_factory=list)  # .idx/.sup next to the file

    @property
    def url(self) -> str:
        if len(self.paths) == 1:
            return self.paths[0]
        return "concat:" + "|".join(self.paths)

    def args(self) -> List[str]:
        """Input arguments for ffmpeg/ffprobe (ending with -i URL).

        A deep probe is used everywhere so that every tool sees the same
        streams (VOB files can reveal subtitle streams late) and therefore
        the same stream indices.
        """
        out: List[str] = ["-probesize", "100M", "-analyzeduration", "100M"]
        if self.fmt:
            out += ["-f", self.fmt]
        for k, v in self.options.items():
            out += [f"-{k}", str(v)]
        out += ["-i", self.url]
        return out

    @property
    def stem(self) -> str:
        base = os.path.basename(self.display.rstrip("/\\")) or "output"
        if base.upper() == "VIDEO_TS":
            base = os.path.basename(os.path.dirname(self.display.rstrip("/\\"))) or base
        return os.path.splitext(base)[0] if not os.path.isdir(self.display) else base

    @property
    def size_bytes(self) -> int:
        total = 0
        for p in self.paths:
            try:
                total += os.path.getsize(p)
            except OSError:
                pass
        return total


# ---------------------------------------------------------------------------
# VOB / VIDEO_TS discovery
# ---------------------------------------------------------------------------

_VTS_RE = re.compile(r"^VTS_(\d\d)_(\d)\.VOB$", re.I)


def _find_ci(folder: str, name: str) -> Optional[str]:
    """Case-insensitive file lookup."""
    try:
        for f in os.listdir(folder):
            if f.lower() == name.lower():
                return os.path.join(folder, f)
    except OSError:
        pass
    return None


def vob_set(path: str) -> Tuple[List[str], Optional[str]]:
    """For VTS_xx_N.VOB return all title VOBs of that set (N>=1) and the IFO."""
    folder, name = os.path.split(path)
    m = _VTS_RE.match(name)
    if not m:
        return [path], None
    vts = m.group(1)
    parts = []
    for f in os.listdir(folder or "."):
        mm = _VTS_RE.match(f)
        if mm and mm.group(1) == vts and mm.group(2) != "0":
            parts.append((int(mm.group(2)), os.path.join(folder, f)))
    parts.sort()
    ifo = _find_ci(folder, f"VTS_{vts}_0.IFO") or _find_ci(folder, f"VTS_{vts}_0.BUP")
    return [p for _, p in parts] or [path], ifo


def largest_title_set(video_ts: str) -> Tuple[List[str], Optional[str]]:
    """Pick the VTS with the most data (almost always the main movie)."""
    sets: Dict[str, int] = {}
    for f in os.listdir(video_ts):
        m = _VTS_RE.match(f)
        if m and m.group(2) != "0":
            sets[m.group(1)] = sets.get(m.group(1), 0) + os.path.getsize(os.path.join(video_ts, f))
    if not sets:
        raise FileNotFoundError("VIDEO_TS 폴더에서 VTS_xx_1.VOB 파일을 찾을 수 없습니다.")
    vts = max(sets, key=lambda k: sets[k])
    first = _find_ci(video_ts, f"VTS_{vts}_1.VOB")
    return vob_set(first)  # type: ignore[arg-type]


def find_external_subs(path: str) -> List[str]:
    base = os.path.splitext(path)[0]
    out = []
    for ext in (".idx", ".sup"):
        for cand in glob.glob(glob.escape(base) + "*" + ext):
            out.append(cand)
    return sorted(set(out))


def resolve_input(path: str, prefer_dvdvideo: bool = True, has_dvdvideo: bool = False,
                  title: int = 0) -> InputSpec:
    """Turn a user selection into an :class:`InputSpec`."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        video_ts = path if os.path.basename(path).upper() == "VIDEO_TS" else \
            (_find_ci(path, "VIDEO_TS") or path)
        root = os.path.dirname(video_ts) if os.path.basename(video_ts).upper() == "VIDEO_TS" else path
        if prefer_dvdvideo and has_dvdvideo and _find_ci(video_ts, "VIDEO_TS.IFO"):
            opts = {"title": str(title)} if title else {}
            return InputSpec(display=root, paths=[root], fmt="dvdvideo", options=opts)
        vobs, ifo = largest_title_set(video_ts)
        return InputSpec(display=path, paths=vobs, ifo=ifo)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".iso" and has_dvdvideo:
        opts = {"title": str(title)} if title else {}
        return InputSpec(display=path, paths=[path], fmt="dvdvideo", options=opts)
    if ext == ".ifo":
        folder = os.path.dirname(path)
        m = re.match(r"VTS_(\d\d)_0\.IFO", os.path.basename(path), re.I)
        if m:
            first = _find_ci(folder, f"VTS_{m.group(1)}_1.VOB")
            if first:
                vobs, ifo = vob_set(first)
                return InputSpec(display=path, paths=vobs, ifo=ifo)
        return resolve_input(folder, prefer_dvdvideo, has_dvdvideo, title)
    if ext == ".vob":
        vobs, ifo = vob_set(path)
        return InputSpec(display=path, paths=vobs, ifo=ifo, external_subs=find_external_subs(path))
    return InputSpec(display=path, paths=[path], external_subs=find_external_subs(path))


# ---------------------------------------------------------------------------
# IFO parsing
# ---------------------------------------------------------------------------

def _lang(code: bytes) -> Optional[str]:
    try:
        s = code.decode("ascii").strip("\x00 ").lower()
    except UnicodeDecodeError:
        return None
    return s if re.fullmatch(r"[a-z]{2}", s) else None


@dataclass
class IfoInfo:
    palette: Optional[List[Tuple[int, int, int]]] = None
    audio_langs: List[Optional[str]] = field(default_factory=list)  # logical streams
    sub_langs: List[Optional[str]] = field(default_factory=list)
    audio_by_phys: Dict[int, Optional[str]] = field(default_factory=dict)  # physical n -> lang
    sub_by_phys: Dict[int, Optional[str]] = field(default_factory=dict)  # 0x20+n -> lang

    def audio_lang_for_id(self, stream_id: int) -> Optional[str]:
        return self.audio_by_phys.get(stream_id & 0x07)

    def sub_lang_for_id(self, stream_id: int) -> Optional[str]:
        return self.sub_by_phys.get(stream_id & 0x1F)


def parse_vts_ifo(path: str) -> IfoInfo:
    """Read palette (first PGC) and stream languages from a VTS_xx_0.IFO."""
    with open(path, "rb") as f:
        data = f.read()
    info = IfoInfo()
    if not data.startswith(b"DVDVIDEO-VTS") or len(data) < 0x400:
        return info
    # audio attributes: count at 0x202, 8 entries of 8 bytes at 0x204
    n_audio = struct.unpack(">H", data[0x202:0x204])[0]
    for i in range(min(n_audio, 8)):
        e = data[0x204 + 8 * i:0x204 + 8 * i + 8]
        info.audio_langs.append(_lang(e[2:4]) if (e[0] & 0x0C) == 0x04 else None)
    # sub-picture attributes: count at 0x254, 32 entries of 6 bytes at 0x256
    n_sub = struct.unpack(">H", data[0x254:0x256])[0]
    for i in range(min(n_sub, 32)):
        e = data[0x256 + 6 * i:0x256 + 6 * i + 6]
        info.sub_langs.append(_lang(e[2:4]) if (e[0] & 0x03) == 0x01 else None)
    # VTS_PGCITI sector pointer at 0xCC -> first program chain
    sector = struct.unpack(">I", data[0xCC:0xD0])[0]
    base = sector * 2048
    if base + 16 > len(data):
        return info
    n_pgc = struct.unpack(">H", data[base:base + 2])[0]
    if not n_pgc:
        return info
    pgc = base + struct.unpack(">I", data[base + 12:base + 16])[0]
    if pgc + 0xA4 + 64 > len(data):
        return info
    # audio stream control: 8 x 2 bytes at 0x0C (bit15 = present, bits 10..8 = physical number)
    for i in range(8):
        (v,) = struct.unpack(">H", data[pgc + 0x0C + 2 * i:pgc + 0x0E + 2 * i])
        if v & 0x8000 and i < len(info.audio_langs):
            info.audio_by_phys.setdefault((v >> 8) & 0x07, info.audio_langs[i])
    # sub-picture stream control: 32 x 4 bytes at 0x1C (4:3 / wide / letterbox / pan-scan)
    for i in range(32):
        (v,) = struct.unpack(">I", data[pgc + 0x1C + 4 * i:pgc + 0x20 + 4 * i])
        if v & 0x80000000 and i < len(info.sub_langs):
            for shift in (24, 16, 8, 0):
                info.sub_by_phys.setdefault((v >> shift) & 0x1F, info.sub_langs[i])
    pal = []
    for i in range(16):
        _, y, cr, cb = data[pgc + 0xA4 + 4 * i:pgc + 0xA4 + 4 * i + 4]
        pal.append(ycrcb_to_rgb(y, cr, cb))
    info.palette = pal
    return info
