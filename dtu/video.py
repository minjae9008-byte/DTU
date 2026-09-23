"""Build the FFmpeg video filter chain from settings and source analysis.

Processing order (all "repair" steps run at SD resolution, where they are
cheap and where the artefacts actually live):

    field order fix -> deinterlace / inverse telecine -> crop
    -> MPEG-2 deblocking -> denoising -> frame interpolation
    -> upscale + BT.601->BT.709 colour conversion (one zimg/libplacebo pass)
    -> debanding (high bit depth) -> CAS sharpening -> output pixel format
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Tuple

from .analyze import CropAnalysis, ScanAnalysis
from .codecs import CODECS, software_pix_fmt
from .probe import MediaInfo, VideoStream
from .settings import EncodeSettings, VideoSettings

TARGET_BOXES = {"720p": (1280, 720), "1080p": (1920, 1080), "1440p": (2560, 1440), "2160p": (3840, 2160)}

# zimg names for colour properties reported by ffprobe
_Z_MATRIX = {"smpte170m": "170m", "bt470bg": "470bg", "bt709": "709", "bt2020nc": "2020_ncl",
             "fcc": "fcc", "smpte240m": "240m"}
_Z_PRIM = {"smpte170m": "170m", "bt470bg": "470bg", "bt709": "709", "bt2020": "2020", "smpte240m": "240m",
           "bt470m": "470m"}
_Z_TRC = {"smpte170m": "601", "bt470bg": "601", "bt470m": "601", "bt709": "709", "bt2020-10": "2020_10",
          "smpte240m": "240m", "gamma22": "601", "gamma28": "601"}

# Denoiser tiers.  Chosen from the benchmark in tools/benchmark: on grainy
# 6 Mbit/s MPEG-2 DVD encodes scored against the original HD master, the
# temporal hqdn3d filter gave the best PSNR/SSIM gain at 600+ fps, while
# non-local means (3-10 fps) and FFmpeg's BM3D (3 fps) did not improve any
# metric.  The strongest tier adds a temporal 3D-FFT denoiser for heavy grain.
DENOISE = {
    "light": "hqdn3d=2:1.5:3:2.25",
    "medium": "hqdn3d=3:2.25:6:4.5",
    "strong": "hqdn3d=4:3:9:6.75",
    "very_strong": "hqdn3d=4:3:6:4.5,fftdnoiz=sigma=6:prev=1:next=1",
}
# MPEG-2 deblocking.  At normal DVD bitrates every deblocker measurably lowered
# fidelity (spp/pp7/fspp even with the real quantiser tables), so it is off by
# default and meant for visibly blocky discs.  FFmpeg's H.264-style ``deblock``
# was the gentlest.
DEBLOCK = {
    "light": "deblock=filter=weak:block=8",
    "medium": "deblock=filter=strong:block=8",
    "strong": "deblock=filter=strong:block=8:alpha=0.12:beta=0.07:gamma=0.06:delta=0.05",
}
DEBAND = {
    "light": "deband=1thr=0.012:2thr=0.012:3thr=0.012:4thr=0.012:range=20:blur=1",
    "strong": "deband=1thr=0.02:2thr=0.02:3thr=0.02:4thr=0.02:range=24:blur=1",
}
CPU_SCALERS = {  # zimg filter names / params
    "lanczos": "filter=lanczos:param_a=3",
    "spline36": "filter=spline36",
    "spline64": "filter=spline64",
    "bicubic": "filter=bicubic:param_a=0:param_b=0.5",
}
GPU_SCALERS = ("ewa_lanczossharp", "ewa_lanczos", "ewa_lanczos4sharpest", "ewa_robidouxsharp", "lanczos",
               "spline36")


@dataclass
class VideoPlan:
    chain: List[str] = field(default_factory=list)
    out_w: int = 0
    out_h: int = 0
    out_fps: Optional[Fraction] = None
    crop: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h in source pixels
    uses_vulkan: bool = False
    notes: List[str] = field(default_factory=list)
    deinterlace_mode: str = "off"
    upscaled: bool = False

    @property
    def filtergraph(self) -> str:
        return ",".join(self.chain) if self.chain else "null"


def fit_box(dar: float, box_w: int, box_h: int) -> Tuple[int, int]:
    """Largest even WxH with the given display aspect that fits the box."""
    if dar >= box_w / box_h:
        w = box_w
        h = int(round(box_w / dar / 2)) * 2
    else:
        h = box_h
        w = int(round(box_h * dar / 2)) * 2
    return max(w, 2), max(h, 2)


def resolve_deinterlace(vs: VideoSettings, scan: Optional[ScanAnalysis]) -> str:
    mode = vs.deinterlace
    if mode != "auto":
        return mode
    if scan is None:
        return "off"
    return {"progressive": "off", "soft_telecine": "soft_telecine", "telecine": "ivtc", "mixed": "ivtc",
            "field_shift": "fieldmatch", "interlaced": "bwdif_double"}.get(scan.scan, "off")


def deinterlace_filters(mode: str, v: VideoStream, parity: str) -> Tuple[List[str], Fraction]:
    fps = v.fps if v.fps else Fraction(30000, 1001)
    par = {"tff": "tff", "bff": "bff"}.get(parity, "auto")
    fm_order = {"tff": "tff", "bff": "bff"}.get(parity, "auto")
    if mode == "off":
        return [], fps
    if mode == "soft_telecine":
        return ["fps=24000/1001"], Fraction(24000, 1001)
    if mode == "bwdif":
        return [f"bwdif=mode=send_frame:parity={par}:deint=all"], fps
    if mode == "bwdif_double":
        return [f"bwdif=mode=send_field:parity={par}:deint=all"], fps * 2
    if mode == "fieldmatch":
        return [f"fieldmatch=order={fm_order}:combmatch=full",
                f"bwdif=mode=send_frame:parity={par}:deint=interlaced"], fps
    if mode == "ivtc":
        # fps= first turns soft-pulldown (repeat-field) frames into real
        # duplicates, so hybrid soft/hard telecine is handled by decimate too
        return ["fps=30000/1001", f"fieldmatch=order={fm_order}:combmatch=full",
                f"bwdif=mode=send_frame:parity={par}:deint=interlaced", "decimate"], Fraction(24000, 1001)
    raise ValueError(f"unknown deinterlace mode {mode}")


def parse_manual_crop(text: str, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    parts = [p for p in text.replace("x", ":").replace(",", ":").split(":") if p.strip()]
    if len(parts) != 4:
        return None
    cw, ch, cx, cy = (int(float(p)) for p in parts)
    cw, ch = cw - cw % 2, ch - ch % 2
    if cw <= 0 or ch <= 0 or cx + cw > w or cy + ch > h:
        return None
    return cx, cy, cw, ch


def target_size(vs: VideoSettings, dar: float, cw: int, ch: int) -> Optional[Tuple[int, int]]:
    if vs.upscale == "off":
        return None
    if vs.upscale == "custom":
        try:
            bw, bh = (int(x) for x in vs.upscale_custom.lower().split("x"))
        except ValueError:
            return None
    else:
        bw, bh = TARGET_BOXES.get(vs.upscale, (1920, 1080))
    return fit_box(dar, bw, bh)


def build_video_plan(info: MediaInfo, vs: VideoSettings, es: EncodeSettings,
                     scan: Optional[ScanAnalysis] = None, crop: Optional[CropAnalysis] = None,
                     caps=None) -> VideoPlan:
    """Assemble the filter chain.  ``caps`` is an FFmpeg instance (or None)."""
    v = info.main_video
    if v is None:
        raise ValueError("no video stream")
    has = (lambda f: caps.has_filter(f)) if caps is not None else (lambda f: True)
    plan = VideoPlan()
    chain = plan.chain

    # 1. field order override
    parity = vs.field_order if vs.field_order in ("tff", "bff") else (scan.field_order if scan else "auto")
    if vs.field_order in ("tff", "bff"):
        chain.append(f"setfield={vs.field_order}")

    # 2. deinterlace / IVTC
    mode = resolve_deinterlace(vs, scan)
    f, fps = deinterlace_filters(mode, v, parity if vs.field_order in ("tff", "bff") else "auto")
    chain += f
    plan.deinterlace_mode = mode
    names = {"off": "", "soft_telecine": "소프트 텔레시네 → 23.976p", "bwdif": "BWDIF 디인터레이스",
             "bwdif_double": "BWDIF 디인터레이스 (2배 프레임, 부드러운 움직임)",
             "fieldmatch": "필드 매칭", "ivtc": "역텔레시네 (fieldmatch + decimate → 23.976p)"}
    if names.get(mode):
        plan.notes.append(names[mode])

    # 3. crop (after deinterlacing so field parity is irrelevant)
    cx, cy, cw, ch = 0, 0, v.width, v.height
    if vs.crop == "manual":
        c = parse_manual_crop(vs.crop_manual, v.width, v.height)
        if c:
            cx, cy, cw, ch = c
    elif vs.crop == "auto" and crop is not None and not crop.is_full:
        cx, cy, cw, ch = crop.x, crop.y, crop.width, crop.height
    if (cw, ch) != (v.width, v.height):
        chain.append(f"crop={cw}:{ch}:{cx}:{cy}")
        plan.notes.append(f"검은 여백 제거 {v.width}x{v.height} → {cw}x{ch}")
    plan.crop = (cx, cy, cw, ch)

    # 4. deblock (MPEG-2 8x8 block edges)
    if vs.deblock in DEBLOCK and has("deblock"):
        chain.append(DEBLOCK[vs.deblock])
        plan.notes.append(f"블록 노이즈 제거 ({vs.deblock})")

    # 5. denoise
    if vs.denoise in DENOISE:
        dn = DENOISE[vs.denoise]
        if "fftdnoiz" in dn and not has("fftdnoiz"):
            dn = dn.split(",")[0]
        chain += dn.split(",")
        plan.notes.append(f"잡음 제거 ({vs.denoise})")

    # 6. frame interpolation (at SD: much faster, same result)
    if vs.interpolate != "off" and float(fps) < 48:
        pal_family = abs(float(fps) - 25) < 0.5
        if vs.interpolate == "2x":
            target = fps * 2
        elif vs.interpolate == "50":
            target = Fraction(50)
        else:  # "60" = smooth display rate of the source family
            target = Fraction(50) if pal_family else Fraction(60000, 1001)
        if vs.interpolate_quality == "blend":
            chain.append(f"framerate=fps={target.numerator}/{target.denominator}")
        else:
            chain.append(f"minterpolate=fps={target.numerator}/{target.denominator}:mi_mode=mci:"
                         f"mc_mode=aobmc:me_mode=bidir:me=epzs:vsbmc=1:scd=fdiff:scd_threshold=10")
        plan.notes.append(f"프레임 보간 {float(fps):.3f} → {float(target):.3f} fps")
        fps = target
    plan.out_fps = fps

    # 7. upscale + colour conversion
    cd = CODECS.get(es.codec, CODECS["av1"])
    pix = software_pix_fmt(cd, es.bit_depth)
    dar = float(Fraction(cw, ch) * (v.sar or 1))
    size = target_size(vs, dar, cw, ch)
    matrix, prim, trc, rng = v.source_colors()
    if size is not None:
        W, H = size
        plan.out_w, plan.out_h = W, H
        plan.upscaled = True
        convert = vs.color_convert and v.is_sd and H > 576
        use_gpu = False
        scaler = vs.scaler
        if vs.gpu_scaler != "off" and caps is not None and caps.has_filter("libplacebo"):
            forced = vs.gpu_scaler == "on"
            if (forced or scaler in GPU_SCALERS or scaler == "auto") and \
                    caps.vulkan_works(allow_software=forced):
                use_gpu = forced or scaler in GPU_SCALERS or scaler == "auto"
        if use_gpu:
            up = scaler if scaler in GPU_SCALERS else "ewa_lanczossharp"
            src = (f"setparams=range={rng}:color_primaries={prim}:color_trc={trc}:colorspace={matrix}"
                   f":chroma_location={v.chroma_location or 'left'}")
            tgt = (":colorspace=bt709:color_primaries=bt709:color_trc=bt709" if convert else
                   f":colorspace={matrix}:color_primaries={prim}:color_trc={trc}")
            deb = ""
            if vs.deband in DEBAND:
                deb = ":deband=1:deband_iterations={}:deband_threshold={}:deband_radius=16:deband_grain=4".format(
                    2 if vs.deband == "strong" else 1, 5 if vs.deband == "strong" else 3)
            chain.append(src)
            chain.append(f"libplacebo=w={W}:h={H}:upscaler={up}:downscaler=mitchell{tgt}:range=tv"
                         f":format={pix}{deb}")
            plan.uses_vulkan = True
            plan.notes.append(f"업스케일 {cw}x{ch} → {W}x{H} (GPU {up})")
        else:
            sc = scaler if scaler in CPU_SCALERS else "lanczos"
            zin = (f"matrixin={_Z_MATRIX.get(matrix, '170m')}:transferin={_Z_TRC.get(trc, '601')}:"
                   f"primariesin={_Z_PRIM.get(prim, '170m')}:rangein={'full' if rng == 'pc' else 'limited'}:"
                   f"chromalin={v.chroma_location or 'left'}")
            if convert:
                zout = "matrix=709:transfer=709:primaries=709:range=limited"
            else:
                zout = (f"matrix={_Z_MATRIX.get(matrix, '709')}:transfer={_Z_TRC.get(trc, '709')}:"
                        f"primaries={_Z_PRIM.get(prim, '709')}:range=limited")
            chain.append(f"zscale=w={W}:h={H}:{CPU_SCALERS[sc]}:{zin}:{zout}:dither=error_diffusion")
            chain.append(f"format={pix}")
            plan.notes.append(f"업스케일 {cw}x{ch} → {W}x{H} (zimg {sc})")
            if vs.deband in DEBAND:
                chain.append(DEBAND[vs.deband])
        if convert:
            plan.notes.append("색 공간 변환 BT.601 → BT.709")
    else:
        plan.out_w, plan.out_h = cw, ch
        chain.append(f"format={pix}")
        if vs.deband in DEBAND:
            chain.append(DEBAND[vs.deband])

    # 8. sharpen (luma only, at output resolution)
    if vs.sharpen > 0.001:
        s = min(max(vs.sharpen, 0.0), 0.8)
        chain.append(f"cas=strength={s:.2f}:planes=1")
        plan.notes.append(f"CAS 선명화 {s:.2f}")
    if vs.deband in DEBAND:
        plan.notes.append(f"디밴딩 ({vs.deband})")

    # 9. square pixels after scaling
    if plan.upscaled:
        chain.append("setsar=1")
    return plan
