"""Media information from ffprobe, with DVD-aware defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional

from .ffmpeg import FFmpeg
from .inputs import InputSpec, IfoInfo, parse_vts_ifo

BITMAP_SUB_CODECS = {"dvd_subtitle", "hdmv_pgs_subtitle", "dvb_subtitle", "xsub"}
TEXT_SUB_CODECS = {"subrip", "ass", "ssa", "webvtt", "mov_text", "text", "srt"}

LANG_NAMES_KO = {
    "ko": "한국어", "kor": "한국어", "en": "영어", "eng": "영어", "ja": "일본어", "jpn": "일본어",
    "zh": "중국어", "chi": "중국어", "zho": "중국어", "fr": "프랑스어", "fre": "프랑스어", "fra": "프랑스어",
    "de": "독일어", "ger": "독일어", "deu": "독일어", "es": "스페인어", "spa": "스페인어",
    "it": "이탈리아어", "ita": "이탈리아어", "ru": "러시아어", "rus": "러시아어", "th": "태국어", "tha": "태국어",
    "pt": "포르투갈어", "por": "포르투갈어", "nl": "네덜란드어", "dut": "네덜란드어", "nld": "네덜란드어",
}

# ISO 639-1 -> 639-2/B (Matroska language tags)
ISO1_TO_2 = {"ko": "kor", "en": "eng", "ja": "jpn", "zh": "chi", "fr": "fre", "de": "ger", "es": "spa",
             "it": "ita", "ru": "rus", "th": "tha", "pt": "por", "nl": "dut", "sv": "swe", "da": "dan",
             "no": "nor", "fi": "fin", "pl": "pol", "cs": "cze", "hu": "hun", "el": "gre", "tr": "tur",
             "he": "heb", "ar": "ara", "hi": "hin", "vi": "vie", "id": "ind", "ms": "may"}


def lang3(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    code = code.lower()
    if code in ("und", "unk", ""):
        return None
    return ISO1_TO_2.get(code, code)


def _frac(s: Optional[str], default: Fraction = Fraction(0)) -> Fraction:
    if not s or s in ("0/0", "N/A"):
        return default
    try:
        return Fraction(s.replace(":", "/"))
    except (ValueError, ZeroDivisionError):
        return default


def _float(s, default=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


@dataclass
class VideoStream:
    index: int
    codec: str
    width: int
    height: int
    sar: Fraction
    fps: Fraction  # r_frame_rate
    avg_fps: Fraction
    field_order: str
    pix_fmt: str
    color_space: Optional[str]
    color_primaries: Optional[str]
    color_transfer: Optional[str]
    color_range: Optional[str]
    chroma_location: Optional[str]
    bits: int = 8

    @property
    def dar(self) -> Fraction:
        return Fraction(self.width, self.height) * (self.sar or 1)

    @property
    def is_ntsc(self) -> bool:
        return self.height in (480, 486) or abs(float(self.fps) - 29.97) < 0.05 or abs(float(self.fps) - 23.976) < 0.01

    @property
    def is_pal(self) -> bool:
        return self.height == 576 or abs(float(self.fps) - 25) < 0.01 or abs(float(self.fps) - 50) < 0.01

    @property
    def is_sd(self) -> bool:
        return self.height <= 576

    def source_colors(self):
        """(matrix, primaries, transfer, range) with SD defaults for untagged DVDs."""
        pal = self.height == 576
        matrix = self.color_space if self.color_space not in (None, "unknown", "reserved") else None
        prim = self.color_primaries if self.color_primaries not in (None, "unknown", "reserved") else None
        trc = self.color_transfer if self.color_transfer not in (None, "unknown", "reserved") else None
        if self.is_sd:
            matrix = matrix or ("bt470bg" if pal else "smpte170m")
            prim = prim or ("bt470bg" if pal else "smpte170m")
            trc = trc or ("bt470bg" if False else "smpte170m")
        else:
            matrix = matrix or "bt709"
            prim = prim or "bt709"
            trc = trc or "bt709"
        rng = self.color_range if self.color_range in ("tv", "pc") else "tv"
        return matrix, prim, trc, rng


@dataclass
class AudioStream:
    index: int
    codec: str
    channels: int
    layout: str
    sample_rate: int
    bit_rate: Optional[int]
    language: Optional[str]
    title: Optional[str]
    default: bool
    stream_id: Optional[int] = None

    def label(self) -> str:
        lang = LANG_NAMES_KO.get(self.language or "", self.language or "언어 미상")
        ch = {1: "모노", 2: "스테레오", 6: "5.1", 8: "7.1"}.get(self.channels, f"{self.channels}ch")
        br = f" {self.bit_rate // 1000}kbps" if self.bit_rate else ""
        return f"#{self.index} {lang} {self.codec.upper()} {ch}{br}"


@dataclass
class SubtitleStream:
    index: int
    codec: str
    language: Optional[str]
    title: Optional[str]
    default: bool
    forced: bool
    width: int = 0
    height: int = 0
    stream_id: Optional[int] = None
    external: Optional[str] = None  # path for external .idx/.sup

    @property
    def is_bitmap(self) -> bool:
        return self.codec in BITMAP_SUB_CODECS

    @property
    def is_text(self) -> bool:
        return self.codec in TEXT_SUB_CODECS

    def label(self) -> str:
        lang = LANG_NAMES_KO.get(self.language or "", self.language or "언어 미상")
        kind = {"dvd_subtitle": "DVD 자막(VobSub)", "hdmv_pgs_subtitle": "PGS"}.get(self.codec, self.codec)
        extra = " (외부 파일)" if self.external else ""
        return f"#{self.index} {lang} {kind}{extra}"


@dataclass
class MediaInfo:
    spec: InputSpec
    format_name: str
    duration: float
    start_time: float
    video: List[VideoStream] = field(default_factory=list)
    audio: List[AudioStream] = field(default_factory=list)
    subtitles: List[SubtitleStream] = field(default_factory=list)
    chapters: int = 0
    ifo: Optional[IfoInfo] = None

    @property
    def main_video(self) -> Optional[VideoStream]:
        return self.video[0] if self.video else None

    def summary(self) -> str:
        v = self.main_video
        lines = []
        if v:
            dar = v.dar
            dar_s = "16:9" if abs(float(dar) - 16 / 9) < 0.03 else ("4:3" if abs(float(dar) - 4 / 3) < 0.03 else f"{float(dar):.2f}:1")
            lines.append(f"영상: {v.codec} {v.width}x{v.height} (화면비 {dar_s}), {float(v.fps):.3f} fps, "
                         f"필드: {v.field_order}")
        for a in self.audio:
            lines.append("오디오: " + a.label())
        for s in self.subtitles:
            lines.append("자막: " + s.label())
        m, s = divmod(int(self.duration), 60)
        h, m = divmod(m, 60)
        lines.append(f"길이: {h}:{m:02d}:{s:02d}, 챕터 {self.chapters}개")
        return "\n".join(lines)


def _stream_id(s: dict) -> Optional[int]:
    sid = s.get("id")
    if not sid:
        return None
    try:
        return int(sid, 16) & 0xFF
    except ValueError:
        return None


def _tag_seconds(v: Optional[str]) -> Optional[float]:
    """Parse Matroska DURATION tags like 01:52:03.123000000."""
    if not v:
        return None
    try:
        h, m, s = v.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        return None


def _media_duration(fmt: dict, streams: list) -> float:
    """Container duration, unless a broken subtitle track inflated it.

    Subtitle packets with an "unknown" end time are sometimes stored with a
    duration of 2^32 ms, which then becomes the file duration.  Audio/video
    stream durations are trusted first.
    """
    av = []
    for s in streams:
        if s.get("codec_type") not in ("video", "audio"):
            continue
        d = _float(s.get("duration"))
        if d is None:
            d = _tag_seconds({k.upper(): v for k, v in (s.get("tags") or {}).items()}.get("DURATION"))
        if d and d > 0:
            av.append(d)
    f = _float(fmt.get("duration"), 0.0) or 0.0
    if av:
        best = max(av)
        if f <= 0 or f > best * 1.5 + 5:
            return best
    return f


def probe_media(ff: FFmpeg, spec: InputSpec) -> MediaInfo:
    data = ff.probe(spec.args(), ["-show_format", "-show_streams", "-show_chapters"])
    fmt = data.get("format", {})
    ifo = None
    if spec.ifo:
        try:
            ifo = parse_vts_ifo(spec.ifo)
        except OSError:
            ifo = None
    info = MediaInfo(spec=spec, format_name=fmt.get("format_name", ""),
                     duration=_media_duration(fmt, data.get("streams", [])),
                     start_time=_float(fmt.get("start_time"), 0.0) or 0.0,
                     chapters=len(data.get("chapters", [])), ifo=ifo)
    for s in data.get("streams", []):
        t = s.get("codec_type")
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        disp = s.get("disposition") or {}
        lang = lang3(tags.get("language"))
        if t == "video":
            if disp.get("attached_pic"):
                continue
            pix = s.get("pix_fmt", "")
            bits = 10 if "10" in pix else (12 if "12" in pix else 8)
            info.video.append(VideoStream(
                index=s["index"], codec=s.get("codec_name", ""), width=int(s.get("width") or 0),
                height=int(s.get("height") or 0), sar=_frac(s.get("sample_aspect_ratio"), Fraction(1)),
                fps=_frac(s.get("r_frame_rate")), avg_fps=_frac(s.get("avg_frame_rate")),
                field_order=s.get("field_order", "unknown"), pix_fmt=pix,
                color_space=s.get("color_space"), color_primaries=s.get("color_primaries"),
                color_transfer=s.get("color_transfer"), color_range=s.get("color_range"),
                chroma_location=s.get("chroma_location"), bits=bits))
        elif t == "audio":
            sid = _stream_id(s)
            if not lang and ifo and sid is not None:
                lang = lang3(ifo.audio_lang_for_id(sid))
            br = s.get("bit_rate")
            info.audio.append(AudioStream(
                index=s["index"], codec=s.get("codec_name", ""), channels=int(s.get("channels") or 0),
                layout=s.get("channel_layout", ""), sample_rate=int(s.get("sample_rate") or 0),
                bit_rate=int(br) if br and br.isdigit() else None, language=lang,
                title=tags.get("title"), default=bool(disp.get("default")), stream_id=sid))
        elif t == "subtitle":
            sid = _stream_id(s)
            if not lang and ifo and sid is not None:
                lang = lang3(ifo.sub_lang_for_id(sid))
            info.subtitles.append(SubtitleStream(
                index=s["index"], codec=s.get("codec_name", ""), language=lang, title=tags.get("title"),
                default=bool(disp.get("default")), forced=bool(disp.get("forced")),
                width=int(s.get("width") or 0), height=int(s.get("height") or 0), stream_id=sid))
    # external subtitle files (movie.idx / movie.sup next to the video)
    for ext_path in spec.external_subs:
        try:
            d = ff.probe(["-i", ext_path], ["-show_streams"])
        except Exception:  # noqa: BLE001 - unreadable side files are ignored
            continue
        for s in d.get("streams", []):
            if s.get("codec_type") != "subtitle":
                continue
            tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
            info.subtitles.append(SubtitleStream(
                index=s["index"], codec=s.get("codec_name", ""), language=lang3(tags.get("language")),
                title=tags.get("title"), default=False, forced=False, width=int(s.get("width") or 0),
                height=int(s.get("height") or 0), external=ext_path))
    return info
