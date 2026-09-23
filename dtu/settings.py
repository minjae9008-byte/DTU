"""All user-facing options, presets and their (de)serialisation."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict


@dataclass
class VideoSettings:
    deinterlace: str = "auto"  # auto | off | bwdif | bwdif_double | ivtc | fieldmatch
    field_order: str = "auto"  # auto | tff | bff
    crop: str = "auto"  # auto | off | manual
    crop_manual: str = ""  # "w:h:x:y" in source pixels
    deblock: str = "off"  # off | light | medium | strong
    denoise: str = "light"  # off | light | medium | strong | very_strong
    upscale: str = "1080p"  # off | 720p | 1080p | 1440p | 2160p | custom
    upscale_custom: str = ""  # "WxH" box to fit into
    scaler: str = "auto"  # auto | lanczos | spline36 | spline64 | bicubic | ewa_lanczossharp | ewa_lanczos
    gpu_scaler: str = "auto"  # auto | on | off  (libplacebo/Vulkan)
    sharpen: float = 0.4  # CAS strength (0 = off)
    deband: str = "light"  # off | light | strong
    interpolate: str = "off"  # off | 50 | 60 | 2x
    interpolate_quality: str = "mci"  # mci (motion compensated) | blend
    color_convert: bool = True  # BT.601 -> BT.709 when upscaling to HD


@dataclass
class EncodeSettings:
    codec: str = "av1"  # av1 | hevc | h264 | vvc | av1_nvenc | hevc_nvenc | av1_qsv | hevc_qsv | av1_amf | hevc_amf | hevc_videotoolbox
    quality: int = -1  # CRF/CQ; -1 = codec default
    speed: str = "balanced"  # fastest | fast | balanced | slow | slowest
    bit_depth: int = 10
    film_grain: int = 0  # AV1 film grain synthesis 0..50
    tune: str = "auto"  # auto | film | animation | grain
    container: str = "mkv"  # mkv | mp4
    extra_args: str = ""


@dataclass
class AudioSettings:
    mode: str = "enhance"  # copy | encode | enhance
    codec: str = "opus"  # opus | aac | flac | ac3 | eac3
    bitrate_kbps: int = 0  # 0 = automatic by channel count
    channels: str = "keep"  # keep | stereo | stereo_dialog | headphones
    normalize: bool = True  # EBU R128 loudness normalisation
    target_lufs: float = -23.0
    dialog_boost: float = 0.0  # dB (centre channel / mid signal)
    drc: str = "off"  # off | light | night
    denoise: str = "off"  # off | light | strong
    declick: bool = False
    declip: bool = False
    exciter: bool = False
    keep_original: bool = False  # also keep the untouched source track
    tracks: str = "all"  # all | first | comma separated language codes


@dataclass
class SubtitleSettings:
    mode: str = "soft"  # soft | burn | copy | none
    algorithm: str = "auto"  # auto | contour | xbr | lanczos | nearest
    tracks: str = "all"  # all | first | comma separated language codes
    burn_index: int = 0  # position among selected tracks when burning
    forced_track: bool = True  # add a separate forced-only track when present


@dataclass
class Settings:
    video: VideoSettings = field(default_factory=VideoSettings)
    encode: EncodeSettings = field(default_factory=EncodeSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    subtitles: SubtitleSettings = field(default_factory=SubtitleSettings)
    output_dir: str = ""  # empty = next to the source
    name_suffix: str = "_DTU"
    ffmpeg_dir: str = ""
    workers: int = 0  # subtitle upscaling processes (0 = auto)
    keep_temp: bool = False
    preset: str = "balanced"  # last chosen preset (GUI)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Settings":
        s = cls()
        apply_overrides(s, d)
        return s

    @classmethod
    def from_json(cls, text: str) -> "Settings":
        return cls.from_dict(json.loads(text))

    def copy(self) -> "Settings":
        return copy.deepcopy(self)


def apply_overrides(obj: Any, d: Dict[str, Any]) -> None:
    """Recursively set dataclass fields from a (partial) dict; unknown keys are ignored."""
    names = {f.name: f for f in fields(obj)}
    for k, v in d.items():
        if k not in names:
            continue
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            apply_overrides(cur, v)
        else:
            if isinstance(cur, bool):
                v = bool(v)
            elif isinstance(cur, int) and not isinstance(cur, bool):
                v = int(v)
            elif isinstance(cur, float):
                v = float(v)
            setattr(obj, k, v)


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "label": "빠름 - 가벼운 보정, 빠른 인코딩",
        "video": {"deblock": "off", "denoise": "light", "scaler": "auto", "sharpen": 0.3, "deband": "off",
                  "interpolate": "off"},
        "encode": {"speed": "fast", "quality": -1},
        "audio": {"mode": "enhance", "normalize": True, "drc": "off", "dialog_boost": 0.0},
    },
    "balanced": {
        "label": "균형 (추천) - 화질과 속도의 균형",
        "video": {"deblock": "off", "denoise": "light", "scaler": "auto", "sharpen": 0.4, "deband": "light",
                  "interpolate": "off"},
        "encode": {"speed": "balanced", "quality": -1},
        "audio": {"mode": "enhance", "normalize": True, "drc": "off", "dialog_boost": 0.0},
    },
    "quality": {
        "label": "고품질 - 잡음 제거 강화, 느린 고효율 인코딩",
        "video": {"deblock": "off", "denoise": "medium", "scaler": "auto", "sharpen": 0.4, "deband": "light",
                  "interpolate": "off"},
        "encode": {"speed": "slow", "quality": -1},
        "audio": {"mode": "enhance", "normalize": True, "drc": "off", "dialog_boost": 0.0},
    },
    "old_film": {
        "label": "오래된 영화 복원 - 최강 잡음 제거 + 음성 잡음 제거",
        "video": {"deblock": "light", "denoise": "very_strong", "scaler": "auto", "sharpen": 0.5,
                  "deband": "strong", "interpolate": "off"},
        "encode": {"speed": "slow", "quality": -1, "film_grain": 8},
        "audio": {"mode": "enhance", "normalize": True, "denoise": "light", "declick": True},
    },
    "animation": {
        "label": "애니메이션 - 선명한 선, 밴딩 제거",
        "video": {"deblock": "light", "denoise": "medium", "scaler": "auto", "sharpen": 0.5, "deband": "strong",
                  "interpolate": "off"},
        "encode": {"speed": "balanced", "quality": -1, "tune": "animation"},
        "audio": {"mode": "enhance", "normalize": True},
    },
    "smooth": {
        "label": "부드러운 움직임 - 60fps 프레임 보간 (매우 느림)",
        "video": {"deblock": "off", "denoise": "light", "scaler": "auto", "sharpen": 0.4, "deband": "light",
                  "interpolate": "60", "interpolate_quality": "mci"},
        "encode": {"speed": "balanced", "quality": -1},
        "audio": {"mode": "enhance", "normalize": True},
    },
    "night": {
        "label": "야간 시청 - 대사 강조 + 다이내믹 레인지 압축",
        "video": {"deblock": "off", "denoise": "light", "scaler": "auto", "sharpen": 0.4, "deband": "light"},
        "encode": {"speed": "balanced", "quality": -1},
        "audio": {"mode": "enhance", "normalize": True, "drc": "night", "dialog_boost": 4.0,
                  "channels": "stereo_dialog"},
    },
    "preserve": {
        "label": "원본 보존 - 업스케일만, 음성/자막 원본 유지",
        "video": {"deblock": "off", "denoise": "off", "scaler": "auto", "sharpen": 0.0, "deband": "off",
                  "interpolate": "off"},
        "encode": {"speed": "balanced", "quality": -1},
        "audio": {"mode": "copy"},
        "subtitles": {"mode": "soft"},
    },
}


# fields a preset never changes (user environment and format choices)
_KEEP = {
    "encode": ("codec", "container", "bit_depth", "extra_args"),
    "audio": ("codec", "tracks", "keep_original", "bitrate_kbps", "target_lufs"),
    "subtitles": ("tracks", "burn_index", "forced_track", "algorithm"),
    "video": ("upscale", "upscale_custom", "gpu_scaler", "crop", "crop_manual", "field_order",
              "color_convert"),
}


def preset_settings(name: str, base: Settings | None = None) -> Settings:
    """Defaults + preset values, keeping the user's format/environment choices from ``base``."""
    s = Settings()
    if base is not None:
        for attr in ("output_dir", "name_suffix", "ffmpeg_dir", "workers", "keep_temp"):
            setattr(s, attr, getattr(base, attr))
        for group, keys in _KEEP.items():
            for k in keys:
                setattr(getattr(s, group), k, getattr(getattr(base, group), k))
        if name != "preserve":
            s.subtitles.mode = base.subtitles.mode
    p = PRESETS.get(name)
    if p:
        s.preset = name
        apply_overrides(s, {k: v for k, v in p.items() if k != "label"})
        if base is not None and "encode" in p and "codec" in p["encode"]:
            s.encode.codec = base.encode.codec  # presets never switch the codec
    return s
