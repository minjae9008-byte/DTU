"""Video encoder definitions (software and hardware)."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .settings import EncodeSettings

SPEEDS = ("fastest", "fast", "balanced", "slow", "slowest")


@dataclass(frozen=True)
class CodecDef:
    key: str
    label: str
    encoder: str
    family: str  # av1 | hevc | h264 | vvc
    hardware: bool
    default_quality: int
    quality_range: Tuple[int, int]
    quality_label: str
    ten_bit: bool = True
    eight_bit: bool = True
    note: str = ""


CODECS: Dict[str, CodecDef] = {c.key: c for c in [
    CodecDef("av1", "AV1 (SVT-AV1) - 최신·최고 압축, 추천", "libsvtav1", "av1", False, 27, (10, 50), "CRF",
             note="같은 화질에서 H.264 대비 약 50%, HEVC 대비 약 20~30% 작은 파일"),
    CodecDef("hevc", "HEVC/H.265 (x265) - 호환성 좋음", "libx265", "hevc", False, 20, (12, 32), "CRF"),
    CodecDef("h264", "H.264 (x264) - 최대 호환성", "libx264", "h264", False, 18, (12, 30), "CRF",
             ten_bit=False, note="구형 TV/기기 재생용 (8비트)"),
    CodecDef("vvc", "VVC/H.266 (VVenC) - 차세대, 실험적", "libvvenc", "vvc", False, 30, (18, 45), "QP",
             eight_bit=False, note="압축률은 가장 높지만 재생 가능한 플레이어가 적고 매우 느림"),
    CodecDef("av1_nvenc", "AV1 (NVIDIA NVENC, RTX 40 이상)", "av1_nvenc", "av1", True, 30, (15, 45), "CQ"),
    CodecDef("hevc_nvenc", "HEVC (NVIDIA NVENC)", "hevc_nvenc", "hevc", True, 24, (15, 38), "CQ"),
    CodecDef("h264_nvenc", "H.264 (NVIDIA NVENC)", "h264_nvenc", "h264", True, 21, (15, 35), "CQ",
             ten_bit=False),
    CodecDef("av1_qsv", "AV1 (Intel Quick Sync, Arc/13세대 이상)", "av1_qsv", "av1", True, 26, (10, 45), "ICQ"),
    CodecDef("hevc_qsv", "HEVC (Intel Quick Sync)", "hevc_qsv", "hevc", True, 23, (10, 40), "ICQ"),
    CodecDef("av1_amf", "AV1 (AMD AMF, RX 7000 이상)", "av1_amf", "av1", True, 30, (15, 45), "QP"),
    CodecDef("hevc_amf", "HEVC (AMD AMF)", "hevc_amf", "hevc", True, 24, (15, 38), "QP"),
    CodecDef("hevc_videotoolbox", "HEVC (Apple VideoToolbox)", "hevc_videotoolbox", "hevc", True, 65, (30, 90),
             "품질(높을수록 좋음)"),
]}

_SVT_PRESET = {"fastest": 10, "fast": 8, "balanced": 6, "slow": 4, "slowest": 2}
_X26X_PRESET = {"fastest": "veryfast", "fast": "fast", "balanced": "medium", "slow": "slow", "slowest": "slower"}
_VVENC_PRESET = {"fastest": "faster", "fast": "fast", "balanced": "medium", "slow": "slow", "slowest": "slower"}
_NVENC_PRESET = {"fastest": "p3", "fast": "p5", "balanced": "p6", "slow": "p7", "slowest": "p7"}
_QSV_PRESET = {"fastest": "veryfast", "fast": "faster", "balanced": "medium", "slow": "slow", "slowest": "veryslow"}
_AMF_QUALITY = {"fastest": "speed", "fast": "balanced", "balanced": "quality", "slow": "quality",
                "slowest": "quality"}


def output_pix_fmt(cd: CodecDef, bit_depth: int) -> str:
    ten = bit_depth >= 10 and cd.ten_bit or not cd.eight_bit
    if cd.hardware:
        if cd.encoder.endswith("_nvenc"):
            return "p010le" if ten else "yuv420p"
        return "p010le" if ten else "nv12"
    return "yuv420p10le" if ten else "yuv420p"


def software_pix_fmt(cd: CodecDef, bit_depth: int) -> str:
    """Planar format to produce at the end of the software filter chain."""
    ten = bit_depth >= 10 and cd.ten_bit or not cd.eight_bit
    return "yuv420p10le" if ten else "yuv420p"


def encoder_args(es: EncodeSettings, fps: Optional[Fraction] = None) -> List[str]:
    cd = CODECS[es.codec]
    q = es.quality if es.quality >= 0 else cd.default_quality
    speed = es.speed if es.speed in SPEEDS else "balanced"
    pix = output_pix_fmt(cd, es.bit_depth)
    gop = int(round(float(fps) * 10)) if fps else 240  # ~10 s keyframe interval for seeking
    a: List[str] = ["-c:v", cd.encoder]
    enc = cd.encoder
    if enc == "libsvtav1":
        params = ["tune=0", "enable-variance-boost=1", "enable-qm=1", "qm-min=0"]
        if es.film_grain > 0:
            params += [f"film-grain={int(es.film_grain)}", "film-grain-denoise=0"]
        a += ["-preset", str(_SVT_PRESET[speed]), "-crf", str(q), "-g", str(gop),
              "-svtav1-params", ":".join(params)]
    elif enc in ("libx265", "libx264"):
        a += ["-preset", _X26X_PRESET[speed], "-crf", str(q)]
        tune = es.tune
        if tune in ("film", "animation", "grain"):
            if enc == "libx265" and tune == "film":
                tune = ""  # x265 has no 'film' tune
            if tune:
                a += ["-tune", tune]
        if enc == "libx265":
            xp = ["aq-mode=3", f"keyint={gop}"]
            if es.tune != "grain":
                xp.append("no-sao=1")
            a += ["-x265-params", ":".join(xp), "-tag:v", "hvc1"]
        else:
            a += ["-g", str(gop)]
    elif enc == "libvvenc":
        # intra period 5 s (VVenC emits no frames for clips shorter than a
        # longer period in current FFmpeg builds)
        a += ["-preset", _VVENC_PRESET[speed], "-qp", str(q), "-period", "5"]
    elif enc.endswith("_nvenc"):
        a += ["-preset", _NVENC_PRESET[speed], "-tune", "hq", "-rc", "vbr", "-cq", str(q), "-b:v", "0",
              "-spatial-aq", "1", "-temporal-aq", "1", "-rc-lookahead", "32", "-g", str(gop)]
        if enc == "hevc_nvenc" and pix == "p010le":
            a += ["-profile:v", "main10"]
        if enc != "av1_nvenc":
            a += ["-bf", "3", "-b_ref_mode", "middle"]
        if speed in ("slow", "slowest"):
            a += ["-multipass", "fullres"]
    elif enc.endswith("_qsv"):
        a += ["-preset", _QSV_PRESET[speed], "-global_quality", str(q), "-look_ahead_depth", "40",
              "-g", str(gop)]
        if enc == "hevc_qsv" and pix == "p010le":
            a += ["-profile:v", "main10"]
    elif enc.endswith("_amf"):
        a += ["-quality", _AMF_QUALITY[speed], "-rc", "cqp", "-qp_i", str(q), "-qp_p", str(q + 2),
              "-g", str(gop)]
        if enc == "hevc_amf":
            a += ["-qp_b", str(q + 4)]
            if pix == "p010le":
                a += ["-profile:v", "main10"]
    elif enc == "hevc_videotoolbox":
        a += ["-q:v", str(q), "-tag:v", "hvc1", "-g", str(gop)]
        if pix == "p010le":
            a += ["-profile:v", "main10"]
    a += ["-pix_fmt", pix]
    if es.extra_args.strip():
        import shlex
        a += shlex.split(es.extra_args)
    return a


def available_codecs(ff) -> List[str]:
    """Codec keys whose encoder is present (hardware ones are test-encoded)."""
    out = []
    for key, cd in CODECS.items():
        if not ff.has_encoder(cd.encoder):
            continue
        if cd.hardware and not ff.encoder_works(cd.encoder, "nv12" if not cd.encoder.endswith("videotoolbox") else "nv12"):
            continue
        out.append(key)
    return out
