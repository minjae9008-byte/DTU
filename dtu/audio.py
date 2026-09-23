"""Audio track planning and (non-AI) enhancement chains.

Enhancements, in processing order:

* ``adeclick`` / ``adeclip``  - impulse-noise and clipping repair (old films, vinyl-sourced tracks)
* ``afftdn``                  - FFT spectral-subtraction denoiser with adaptive noise-floor tracking
* dialogue boost             - centre channel gain for 5.1; ``dialoguenhance`` voice extraction for stereo
* downmix                    - ITU-R BS.775 stereo downmix or a dialogue-first "night" downmix
* ``dynaudnorm``             - dynamic range control (evens out quiet dialogue and loud effects)
* ``aexciter``               - harmonic exciter restoring high-frequency "air" lost to lossy coding
* loudness                   - EBU R128 integrated loudness measured in a first pass with ``ebur128``,
                               corrected with a constant gain and a look-ahead peak limiter
                               (keeps the original dynamics, unlike single-pass dynamic normalisation)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .ffmpeg import FFmpeg, popen
from .probe import AudioStream, MediaInfo, LANG_NAMES_KO
from .settings import AudioSettings

LAYOUT_CHANNELS: Dict[str, List[str]] = {
    "mono": ["FC"], "stereo": ["FL", "FR"], "2.1": ["FL", "FR", "LFE"], "3.0": ["FL", "FR", "FC"],
    "3.1": ["FL", "FR", "FC", "LFE"], "4.0": ["FL", "FR", "FC", "BC"], "quad": ["FL", "FR", "BL", "BR"],
    "quad(side)": ["FL", "FR", "SL", "SR"], "5.0": ["FL", "FR", "FC", "BL", "BR"],
    "5.0(side)": ["FL", "FR", "FC", "SL", "SR"], "5.1": ["FL", "FR", "FC", "LFE", "BL", "BR"],
    "5.1(side)": ["FL", "FR", "FC", "LFE", "SL", "SR"],
    "6.1": ["FL", "FR", "FC", "LFE", "BC", "SL", "SR"],
    "7.1": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
}

CODEC_ENCODER = {"opus": "libopus", "aac": "aac", "flac": "flac", "ac3": "ac3", "eac3": "eac3"}
_I_RE = re.compile(r"^\s*I:\s*(-?[\d.]+|-inf)\s*LUFS", re.M)


def default_bitrate(codec: str, channels: int) -> int:
    ch = max(1, channels)
    table = {
        "opus": {1: 96, 2: 160, 6: 320, 8: 448},
        "aac": {1: 128, 2: 256, 6: 448, 8: 640},
        "ac3": {1: 128, 2: 224, 6: 640, 8: 640},
        "eac3": {1: 128, 2: 224, 6: 640, 8: 768},
    }.get(codec, {})
    if not table:
        return 0
    key = min(table, key=lambda k: (abs(k - ch), -k))
    return table[key]


@dataclass
class AudioTrackPlan:
    stream: AudioStream
    action: str  # copy | encode
    filters: List[str] = field(default_factory=list)  # before the loudness stage
    codec: str = "copy"
    bitrate: int = 0
    out_channels: int = 0
    out_layout: str = ""
    normalize: bool = False
    target_lufs: float = -23.0
    measured_lufs: Optional[float] = None
    gain_db: float = 0.0
    title: Optional[str] = None
    default: bool = False
    notes: List[str] = field(default_factory=list)

    def full_filters(self) -> List[str]:
        f = list(self.filters)
        if self.normalize and self.measured_lufs is not None:
            if abs(self.gain_db) > 0.05:
                f.append(f"volume={self.gain_db:+.2f}dB")
        if self.action == "encode":
            # look-ahead limiter at -1 dBFS keeps peaks legal after gain changes
            f.append("alimiter=limit=0.891:attack=5:release=50:level=0:latency=1")
            if self.codec == "opus":
                f.append("aresample=48000")
                if self.out_layout == "5.1(side)":
                    f.append("channelmap=channel_layout=5.1")
                elif self.out_layout == "7.1(wide)":
                    f.append("channelmap=channel_layout=7.1")
        return f

    def codec_args(self, out_index: int) -> List[str]:
        s = f":a:{out_index}"
        if self.action == "copy":
            return [f"-c{s}", "copy"]
        enc = CODEC_ENCODER.get(self.codec, "libopus")
        a = [f"-c{s}", enc]
        if self.codec == "flac":
            a += [f"-compression_level{s}", "8", f"-sample_fmt{s}", "s32", f"-bits_per_raw_sample{s}", "24"]
        else:
            a += [f"-b{s}", f"{self.bitrate}k"]
        if self.codec == "opus":
            a += [f"-vbr{s}", "on", f"-compression_level{s}", "10", f"-application{s}", "audio"]
        return a


def _channels_of(layout: str, n: int) -> List[str]:
    if layout in LAYOUT_CHANNELS:
        return LAYOUT_CHANNELS[layout]
    return {1: LAYOUT_CHANNELS["mono"], 2: LAYOUT_CHANNELS["stereo"],
            6: LAYOUT_CHANNELS["5.1(side)"], 8: LAYOUT_CHANNELS["7.1"]}.get(n, [])


def _db(g: float) -> float:
    return 10 ** (g / 20.0)


def build_enhance_filters(st: AudioStream, s: AudioSettings) -> (List[str], str, List[str]):
    """Return (filters, output layout, human notes) for one track."""
    f: List[str] = []
    notes: List[str] = []
    layout = st.layout or {1: "mono", 2: "stereo", 6: "5.1(side)"}.get(st.channels, "")
    chans = _channels_of(layout, st.channels)
    if s.declick:
        f.append("adeclick=w=55:o=75")
        notes.append("클릭 잡음 제거")
    if s.declip:
        f.append("adeclip")
        notes.append("클리핑 복원")
    if s.denoise in ("light", "strong"):
        nr, nf = (10, -50) if s.denoise == "light" else (18, -42)
        f.append(f"afftdn=nr={nr}:nf={nf}:tn=1")
        notes.append("배경 잡음 제거")
    has_fc = "FC" in chans and len(chans) > 1
    surround = [c for c in chans if c in ("SL", "BL")], [c for c in chans if c in ("SR", "BR")]

    # dialogue boost on multichannel sources: raise the centre channel
    if s.dialog_boost > 0.05 and has_fc and len(chans) >= 3:
        g = _db(s.dialog_boost)
        spec = "|".join(f"{c}={g:.4f}*{c}" if c == "FC" else f"{c}={c}" for c in chans)
        f.append(f"pan={layout}|{spec}")
        notes.append(f"대사(센터) +{s.dialog_boost:.1f}dB")
    elif s.dialog_boost > 0.05 and len(chans) == 2:
        # extract the voice-like centre component and mix it back louder
        enh = min(3.0, max(0.5, s.dialog_boost / 3.0))
        f.append(f"dialoguenhance=original=1:enhance={enh:.2f}:voice=2")
        f.append("pan=stereo|FL<FL+0.707*FC|FR<FR+0.707*FC")
        notes.append(f"대사 강조 +{s.dialog_boost:.1f}dB")
    out_layout = layout

    if s.channels in ("stereo", "stereo_dialog") and len(chans) > 2:
        ls = "+".join(f"0.707*{c}" for c in surround[0])
        rs = "+".join(f"0.707*{c}" for c in surround[1])
        if s.channels == "stereo_dialog" and has_fc:
            ls = "+".join(f"0.30*{c}" for c in surround[0])
            rs = "+".join(f"0.30*{c}" for c in surround[1])
            left = "FC+0.30*FL" + (("+" + ls) if ls else "")
            right = "FC+0.30*FR" + (("+" + rs) if rs else "")
            notes.append("대사 우선 스테레오 다운믹스")
        else:
            left = "FL" + ("+0.707*FC" if has_fc else "") + (("+" + ls) if ls else "")
            right = "FR" + ("+0.707*FC" if has_fc else "") + (("+" + rs) if rs else "")
            notes.append("스테레오 다운믹스 (ITU-R BS.775)")
        f.append(f"pan=stereo|FL<{left}|FR<{right}")
        out_layout = "stereo"

    if s.drc == "light":
        f.append("dynaudnorm=f=500:g=31:p=0.95:m=5:r=0:s=0")
        notes.append("다이내믹 레인지 완만 압축")
    elif s.drc == "night":
        f.append("dynaudnorm=f=250:g=15:p=0.9:m=15:s=10")
        notes.append("야간 모드 (큰 소리 억제, 작은 소리 증폭)")
    if s.exciter:
        f.append("aexciter=amount=0.6:drive=6:blend=0:freq=6000:ceil=16000")
        notes.append("고음역 보강 (익사이터)")
    return f, out_layout, notes


def select_tracks(info: MediaInfo, spec: str) -> List[AudioStream]:
    tracks = list(info.audio)
    spec = (spec or "all").strip().lower()
    if spec == "all" or not tracks:
        return tracks
    if spec == "first":
        return tracks[:1]
    wanted = [w.strip() for w in spec.split(",") if w.strip()]
    sel = [t for t in tracks if (t.language or "") in wanted]
    return sel or tracks[:1]


def plan_audio(info: MediaInfo, s: AudioSettings, container: str = "mkv") -> List[AudioTrackPlan]:
    plans: List[AudioTrackPlan] = []
    tracks = select_tracks(info, s.tracks)
    codec = s.codec if s.codec in CODEC_ENCODER else "opus"
    if container == "mp4" and codec == "flac":
        codec = "aac"
    for i, st in enumerate(tracks):
        lang = LANG_NAMES_KO.get(st.language or "", st.language or "")
        if s.mode == "copy" or (container == "mp4" and st.codec in ("pcm_dvd", "dts") and s.mode == "copy"):
            plans.append(AudioTrackPlan(stream=st, action="copy", title=st.title, default=(i == 0),
                                        out_channels=st.channels))
            continue
        filters, layout, notes = ([], st.layout, []) if s.mode == "encode" else build_enhance_filters(st, s)
        out_ch = len(_channels_of(layout, st.channels)) or st.channels
        if codec in ("ac3", "eac3") and out_ch > 6:
            filters.append("pan=5.1(side)|FL=FL|FR=FR|FC=FC|LFE=LFE|SL=SL+BL|SR=SR+BR")
            layout, out_ch = "5.1(side)", 6
        br = s.bitrate_kbps or default_bitrate(codec, out_ch)
        normalize = s.normalize and s.mode == "enhance"
        if normalize:
            notes.append(f"음량 정규화 EBU R128 ({s.target_lufs:g} LUFS)")
        ch_name = {1: "모노", 2: "스테레오", 6: "5.1", 8: "7.1"}.get(out_ch, f"{out_ch}ch")
        title = f"{lang} {ch_name} {codec.upper()}".strip() if lang else f"{ch_name} {codec.upper()}"
        if notes:
            title += " - " + ", ".join(n.split(" (")[0] for n in notes[:3])
        plans.append(AudioTrackPlan(stream=st, action="encode", filters=filters, codec=codec, bitrate=br,
                                    out_channels=out_ch, out_layout=layout, normalize=normalize,
                                    target_lufs=s.target_lufs, title=title, default=(i == 0), notes=notes))
        if s.keep_original:
            plans.append(AudioTrackPlan(stream=st, action="copy", title=(st.title or f"{lang} 원본").strip(),
                                        default=False, out_channels=st.channels))
    return plans


def measure_loudness(ff: FFmpeg, info: MediaInfo, plans: Sequence[AudioTrackPlan],
                     progress: Optional[Callable[[float], None]] = None,
                     cancel: Optional[Callable[[], bool]] = None,
                     input_args: Optional[Sequence[str]] = None) -> None:
    """First pass: integrated loudness of every track that will be normalised."""
    todo = [p for p in plans if p.normalize and p.action == "encode"]
    if not todo:
        return
    parts = []
    maps: List[str] = []
    for k, p in enumerate(todo):
        chain = ",".join(p.filters + [f"ebur128@m{k}=framelog=quiet"]) if p.filters else f"ebur128@m{k}=framelog=quiet"
        parts.append(f"[0:{p.stream.index}]{chain}[m{k}]")
        maps += ["-map", f"[m{k}]"]
    args = [ff.ffmpeg, "-hide_banner", "-nostdin", "-v", "info", "-nostats", "-progress", "pipe:1",
            *(input_args or info.spec.args()), "-vn", "-sn", "-dn", "-filter_complex", ";".join(parts), *maps, "-f", "null", "-"]
    import subprocess
    proc = popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err_chunks: List[bytes] = []
    import threading

    def drain():
        for chunk in iter(lambda: proc.stderr.read(65536), b""):
            err_chunks.append(chunk)

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    dur = info.duration or 0
    for line in iter(proc.stdout.readline, b""):
        if cancel and cancel():
            proc.kill()
            raise KeyboardInterrupt
        m = re.match(rb"out_time_us=(\d+)", line)
        if m and dur > 0 and progress:
            progress(min(1.0, int(m.group(1)) / 1e6 / dur))
    proc.wait()
    t.join(timeout=5)
    text = b"".join(err_chunks).decode("utf-8", "replace")
    for k, p in enumerate(todo):
        idx = text.find(f"[ebur128@m{k} @")
        if idx < 0:
            continue
        m = _I_RE.search(text, idx)
        if not m or m.group(1) == "-inf":
            continue
        val = float(m.group(1))
        if val < -70 or not math.isfinite(val):
            continue
        p.measured_lufs = val
        p.gain_db = max(-20.0, min(20.0, p.target_lufs - val))
