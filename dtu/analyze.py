"""Source analysis: scan type (progressive / telecine / interlaced) and black borders.

Scan type detection uses a two-stage test that was calibrated on telecined,
interlaced and progressive MPEG-2 test clips:

1. ``idet`` on the decoded frames.  Hard-telecined film looks "interlaced"
   here, but it also shows ~40 % repeated fields (the 3:2 pulldown cadence).
2. ``fieldmatch`` followed by ``idet``.  Film content (telecine or PAL
   field-shift) becomes fully progressive after field matching, true
   interlaced video stays interlaced.

Soft telecine (progressive frames with repeat-field flags, the most common
NTSC film DVD case) is detected from the decoder's ``repeat_pict`` flags.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .ffmpeg import FFmpeg, run
from .probe import MediaInfo

_IDET_MULTI = re.compile(r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)")
_IDET_REP = re.compile(r"Repeated Fields:\s*Neither:\s*(\d+)\s*Top:\s*(\d+)\s*Bottom:\s*(\d+)")
_CROP = re.compile(r"crop=(-?\d+):(-?\d+):(-?\d+):(-?\d+)")


@dataclass
class ScanAnalysis:
    scan: str = "progressive"  # progressive | soft_telecine | telecine | field_shift | interlaced | mixed
    field_order: str = "tff"
    interlaced_ratio: float = 0.0
    repeated_ratio: float = 0.0
    matched_interlaced_ratio: float = 0.0
    soft_pulldown_ratio: float = 0.0
    frames: int = 0

    def describe(self) -> str:
        names = {
            "progressive": "프로그레시브 (디인터레이스 불필요)",
            "soft_telecine": "소프트 텔레시네 필름 (23.976p로 복원)",
            "telecine": "하드 텔레시네 필름 (역텔레시네 IVTC 적용)",
            "field_shift": "필드 어긋난 필름 (필드 매칭 적용)",
            "interlaced": "인터레이스 비디오 (BWDIF 디인터레이스)",
            "mixed": "혼합 (필드 매칭 + 디인터레이스)",
        }
        return (f"{names.get(self.scan, self.scan)} - 인터레이스 {self.interlaced_ratio:.0%}, "
                f"반복필드 {self.repeated_ratio:.0%}, 필드매칭 후 {self.matched_interlaced_ratio:.0%}, "
                f"필드순서 {self.field_order.upper()}")


@dataclass
class CropAnalysis:
    width: int
    height: int
    x: int
    y: int
    source_w: int
    source_h: int
    samples: int = 0

    @property
    def is_full(self) -> bool:
        return self.width == self.source_w and self.height == self.source_h

    def as_filter(self) -> str:
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"

    def describe(self) -> str:
        if self.is_full:
            return "검은 여백 없음"
        return (f"{self.source_w}x{self.source_h} → {self.width}x{self.height} "
                f"(좌 {self.x}, 위 {self.y}, 우 {self.source_w - self.width - self.x}, "
                f"아래 {self.source_h - self.height - self.y})")


def _sample_times(duration: float, n: int) -> List[float]:
    if duration <= 0:
        return [0.0]
    if duration < 60:
        return [0.0]
    lo, hi = 0.1 * duration, 0.8 * duration
    if n == 1:
        return [0.5 * duration]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _run_filter(ff: FFmpeg, info: MediaInfo, t: float, frames: int, vf: str, vindex: int) -> str:
    args = [ff.ffmpeg, "-hide_banner", "-nostdin", "-v", "info"]
    if t > 0:
        args += ["-ss", f"{t:.3f}"]
    args += info.spec.args()
    args += ["-map", f"0:{vindex}", "-frames:v", str(frames), "-vf", vf, "-an", "-sn", "-dn",
             "-f", "null", "-"]
    p = run(args, timeout=600)
    return p.stderr.decode("utf-8", "replace")


def _last(regex, text) -> Optional[Tuple[int, ...]]:
    m = regex.findall(text)
    if not m:
        return None
    return tuple(int(v) for v in m[-1])


def analyze_scan(ff: FFmpeg, info: MediaInfo, samples: int = 4, frames: int = 300,
                 progress: Optional[Callable[[str], None]] = None) -> ScanAnalysis:
    v = info.main_video
    res = ScanAnalysis()
    if v is None:
        return res
    tff = bff = prog = und = 0
    neither = top = bottom = 0
    times = _sample_times(info.duration, samples)
    for i, t in enumerate(times):
        if progress:
            progress(f"인터레이스 분석 {i + 1}/{len(times)}")
        text = _run_filter(ff, info, t, frames, "idet", v.index)
        m = _last(_IDET_MULTI, text)
        r = _last(_IDET_REP, text)
        if m:
            tff += m[0]; bff += m[1]; prog += m[2]; und += m[3]
        if r:
            neither += r[0]; top += r[1]; bottom += r[2]
    decided = tff + bff + prog
    res.frames = decided + und
    res.field_order = "bff" if bff > tff else "tff"
    res.interlaced_ratio = (tff + bff) / decided if decided else 0.0
    total_rep = neither + top + bottom
    res.repeated_ratio = (top + bottom) / total_rep if total_rep else 0.0
    ntsc = abs(float(v.fps) - 29.97) < 0.05 or abs(float(v.fps) - 59.94) < 0.1

    # soft pulldown: repeat_pict flags on decoded frames
    if ntsc and v.codec in ("mpeg2video", "mpeg1video"):
        try:
            t0 = times[len(times) // 2]
            p = run([ff.ffprobe, "-v", "error", "-select_streams", f"{v.index}", "-read_intervals",
                     f"{t0:.3f}%+#{frames}", "-show_entries", "frame=repeat_pict", "-of", "csv=p=0",
                     *info.spec.args()], timeout=300)
            vals = [l.strip() for l in p.stdout.decode("utf-8", "replace").splitlines() if l.strip()]
            if vals:
                res.soft_pulldown_ratio = sum(1 for x in vals if x not in ("0", "")) / len(vals)
        except Exception:  # noqa: BLE001 - optional refinement only
            pass

    if res.interlaced_ratio < 0.10:
        res.scan = "soft_telecine" if res.soft_pulldown_ratio > 0.2 else "progressive"
        return res

    # stage 2: does field matching produce progressive frames?
    tff2 = bff2 = prog2 = 0
    for i, t in enumerate(times):
        if progress:
            progress(f"필드 매칭 분석 {i + 1}/{len(times)}")
        text = _run_filter(ff, info, t, frames, "fieldmatch=order=auto:combmatch=none,idet", v.index)
        m = _last(_IDET_MULTI, text)
        if m:
            tff2 += m[0]; bff2 += m[1]; prog2 += m[2]
    d2 = tff2 + bff2 + prog2
    res.matched_interlaced_ratio = (tff2 + bff2) / d2 if d2 else 1.0
    if res.matched_interlaced_ratio < 0.10:
        if ntsc and (res.repeated_ratio > 0.10 or res.soft_pulldown_ratio > 0.2):
            res.scan = "telecine"
        else:
            res.scan = "field_shift"
    elif res.matched_interlaced_ratio < 0.5 and ntsc and res.repeated_ratio > 0.08:
        res.scan = "mixed"
    else:
        res.scan = "interlaced"
    return res


def analyze_crop(ff: FFmpeg, info: MediaInfo, samples: int = 8, frames: int = 30,
                 pre_filter: str = "", limit: int = 24,
                 progress: Optional[Callable[[str], None]] = None) -> Optional[CropAnalysis]:
    """Detect black borders; the union of all samples is kept (never over-crops)."""
    v = info.main_video
    if v is None:
        return None
    boxes = []
    times = _sample_times(info.duration, samples)
    src_w, src_h = v.width, v.height
    for i, t in enumerate(times):
        if progress:
            progress(f"검은 여백 분석 {i + 1}/{len(times)}")
        vf = (pre_filter + "," if pre_filter else "") + f"cropdetect=limit={limit}:round=2:reset=0"
        text = _run_filter(ff, info, t, frames, vf, v.index)
        m = _last(_CROP, text)
        if not m:
            continue
        w, h, x, y = m
        if w < 0.3 * src_w or h < 0.3 * src_h or w <= 0 or h <= 0:
            continue  # black / fade-out sample
        boxes.append((x, y, x + w, y + h))
    if not boxes:
        return CropAnalysis(src_w, src_h, 0, 0, src_w, src_h, 0)
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    # snap to even numbers (4:2:0) and ignore tiny (<= 2 px) borders
    x0 = x0 - (x0 % 2)
    y0 = y0 - (y0 % 2)
    x1 = min(src_w, x1 + (x1 % 2))
    y1 = min(src_h, y1 + (y1 % 2))
    if x0 <= 2:
        x0 = 0
    if y0 <= 2:
        y0 = 0
    if src_w - x1 <= 2:
        x1 = src_w
    if src_h - y1 <= 2:
        y1 = src_h
    return CropAnalysis(x1 - x0, y1 - y0, x0, y0, src_w, src_h, len(boxes))
