"""Command line interface.

Examples::

    python -m dtu info  movie.mkv
    python -m dtu encode movie.mkv --preset balanced
    python -m dtu encode VIDEO_TS --preset quality --codec hevc -o out.mkv
    python -m dtu subs movie.idx --size 1920x1080 -o movie.sup
    python -m dtu preview movie.mkv --time 600 -o preview.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

from . import __version__
from .codecs import CODECS, available_codecs
from .ffmpeg import FFmpeg, FFmpegError, FFmpegNotFound
from .settings import PRESETS, Settings, apply_overrides, preset_settings


def _progress_printer():
    state = {"stage": None, "t": 0.0}
    names = {"analyze": "분석", "subtitles": "자막", "loudness": "음량 측정", "encode": "인코딩"}

    def cb(stage: str, frac: float, detail: str):
        now = time.time()
        if stage != state["stage"]:
            if state["stage"] is not None:
                sys.stderr.write("\n")
            state["stage"] = stage
        elif now - state["t"] < 0.5 and frac < 1.0:
            return
        state["t"] = now
        bar = "#" * int(frac * 30)
        sys.stderr.write(f"\r[{names.get(stage, stage):6s}] [{bar:30s}] {detail[:90]:90s}")
        sys.stderr.flush()
    return cb


def _log(msg: str):
    sys.stderr.write("\n" + msg + "\n")
    sys.stderr.flush()


def build_settings(ns: argparse.Namespace) -> Settings:
    s = Settings()
    if ns.settings:
        with open(ns.settings, "r", encoding="utf-8") as f:
            s = Settings.from_json(f.read())
    if ns.preset:
        s = preset_settings(ns.preset, s)
    o = {}
    v, e, a, t = {}, {}, {}, {}
    if ns.upscale:
        if "x" in ns.upscale:
            v["upscale"], v["upscale_custom"] = "custom", ns.upscale
        else:
            v["upscale"] = ns.upscale
    for key, dest in (("deinterlace", v), ("denoise", v), ("deblock", v), ("deband", v), ("scaler", v),
                      ("interpolate", v), ("crop", v)):
        val = getattr(ns, key, None)
        if val is not None:
            dest[key] = val
    if ns.sharpen is not None:
        v["sharpen"] = ns.sharpen
    if ns.gpu is not None:
        v["gpu_scaler"] = ns.gpu
    if ns.codec:
        e["codec"] = ns.codec
    if ns.quality is not None:
        e["quality"] = ns.quality
    if ns.speed:
        e["speed"] = ns.speed
    if ns.grain is not None:
        e["film_grain"] = ns.grain
    if ns.container:
        e["container"] = ns.container
    if ns.bit_depth:
        e["bit_depth"] = ns.bit_depth
    if ns.audio:
        a["mode"] = ns.audio
    if ns.audio_codec:
        a["codec"] = ns.audio_codec
    if ns.loudness is not None:
        a["normalize"] = ns.loudness != "off"
        if ns.loudness != "off":
            a["target_lufs"] = float(ns.loudness)
    if ns.dialog is not None:
        a["dialog_boost"] = ns.dialog
    if ns.drc:
        a["drc"] = ns.drc
    if ns.downmix:
        a["channels"] = ns.downmix
    if ns.audio_denoise:
        a["denoise"] = ns.audio_denoise
    if ns.keep_original_audio:
        a["keep_original"] = True
    if ns.subs:
        t["mode"] = ns.subs
    if ns.sub_algo:
        t["algorithm"] = ns.sub_algo
    if ns.sub_tracks:
        t["tracks"] = ns.sub_tracks
    if ns.audio_tracks:
        a["tracks"] = ns.audio_tracks
    o.update({"video": v, "encode": e, "audio": a, "subtitles": t})
    if ns.ffmpeg:
        o["ffmpeg_dir"] = ns.ffmpeg
    apply_overrides(s, o)
    return s


def add_common(p: argparse.ArgumentParser):
    p.add_argument("--ffmpeg", help="FFmpeg 폴더(또는 ffmpeg 실행 파일) 경로")


def cmd_info(ns) -> int:
    from .analyze import analyze_crop, analyze_scan
    from .inputs import resolve_input
    from .probe import probe_media
    ff = FFmpeg(folder=ns.ffmpeg)
    spec = resolve_input(ns.input, has_dvdvideo=ff.has_demuxer("dvdvideo"))
    info = probe_media(ff, spec)
    print(info.summary())
    if not ns.quick:
        scan = analyze_scan(ff, info)
        print("스캔: " + scan.describe())
        crop = analyze_crop(ff, info)
        if crop:
            print("여백: " + crop.describe())
    return 0


def cmd_caps(ns) -> int:
    ff = FFmpeg(folder=ns.ffmpeg)
    print(ff.version_string)
    print("사용 가능한 비디오 코덱:")
    for k in available_codecs(ff):
        print(f"  {k:18s} {CODECS[k].label}")
    for f in ("zscale", "libplacebo", "bwdif", "fieldmatch", "nlmeans", "bm3d", "minterpolate", "cas",
              "deband", "dialoguenhance", "afftdn", "dynaudnorm"):
        print(f"  filter {f:15s} {'OK' if ff.has_filter(f) else '없음'}")
    print(f"  GPU(Vulkan/libplacebo) {'OK' if ff.vulkan_works() else '사용 불가'}")
    print(f"  DVD-Video 직접 읽기     {'OK' if ff.has_demuxer('dvdvideo') else '없음'}")
    return 0


def cmd_encode(ns) -> int:
    from .job import Cancelled, Job
    s = build_settings(ns)
    if ns.save_settings:
        with open(ns.save_settings, "w", encoding="utf-8") as f:
            f.write(s.to_json())
    rc = 0
    for src in ns.inputs:
        sample = None
        if ns.sample:
            parts = [float(x) for x in ns.sample.split(",")] if "," in ns.sample else [-1.0, float(ns.sample)]
            sample = (parts[0], parts[1])
        job = Job(src, s, output=ns.output if len(ns.inputs) == 1 else None, sample=sample)
        try:
            plan = job.analyze(_progress_printer(), _log)
            if ns.dry_run:
                print(plan.describe())
                from .ffmpeg import args_to_display
                job.temp_dir = "<temp>"
                print(args_to_display(job.build_command()))
                continue
            out = job.run(_progress_printer(), _log)
            print(out)
        except KeyboardInterrupt:
            job.cancel()
            _log("중단됨")
            return 130
        except Cancelled:
            _log("중단됨")
            return 130
        except (FFmpegError, OSError, ValueError) as e:
            _log(f"오류: {e}")
            rc = 1
    return rc


def cmd_subs(ns) -> int:
    from .subtitles.pipeline import Geometry, convert_track
    from .subtitles.source import load_track
    ff = FFmpeg(folder=ns.ffmpeg)
    track = load_track(ff, ["-i", ns.input], ns.stream)
    w, h = (int(x) for x in ns.size.lower().split("x"))
    cw, ch = track.canvas
    crop = (0, 0, cw, ch)
    if ns.crop:
        cw2, ch2, cx, cy = (int(x) for x in ns.crop.split(":"))
        crop = (cx, cy, cw2, ch2)
    out = ns.output or os.path.splitext(ns.input)[0] + f".{w}x{h}.sup"
    n = convert_track(track, Geometry(crop, w, h), out, ns.algorithm, progress=None)
    print(f"{n}개 자막 → {out}")
    return 0


def cmd_preview(ns) -> int:
    from .preview import make_preview_files
    s = build_settings(ns)
    before, after = make_preview_files(ns.input, s, ns.time, ns.output or "preview.png")
    print(before)
    print(after)
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dtu", description=f"DTU {__version__} - DVD 영상/자막 리마스터 (AI 미사용)")
    sub = p.add_subparsers(dest="cmd")

    pi = sub.add_parser("info", help="미디어 정보 및 인터레이스/여백 분석")
    pi.add_argument("input")
    pi.add_argument("--quick", action="store_true", help="분석 생략")
    add_common(pi)

    pc = sub.add_parser("caps", help="FFmpeg 기능/코덱 확인")
    add_common(pc)

    def add_options(q):
        q.add_argument("--preset", choices=sorted(PRESETS), help="프리셋")
        q.add_argument("--settings", help="설정 JSON 파일")
        q.add_argument("--save-settings", help="최종 설정을 JSON으로 저장")
        q.add_argument("--upscale", help="off|720p|1080p|1440p|2160p|WxH")
        q.add_argument("--scaler", help="auto|lanczos|spline36|spline64|bicubic|ewa_lanczossharp|...")
        q.add_argument("--gpu", choices=["auto", "on", "off"], help="GPU(libplacebo) 업스케일")
        q.add_argument("--deinterlace", choices=["auto", "off", "bwdif", "bwdif_double", "ivtc", "fieldmatch"])
        q.add_argument("--crop", choices=["auto", "off"])
        q.add_argument("--deblock", choices=["off", "light", "medium", "strong"])
        q.add_argument("--denoise", choices=["off", "light", "medium", "strong", "very_strong"])
        q.add_argument("--deband", choices=["off", "light", "strong"])
        q.add_argument("--sharpen", type=float, help="CAS 선명화 강도 0~0.8")
        q.add_argument("--interpolate", choices=["off", "60", "50", "2x"])
        q.add_argument("--codec", choices=sorted(CODECS))
        q.add_argument("--quality", type=int, help="CRF/QP (낮을수록 고화질)")
        q.add_argument("--speed", choices=["fastest", "fast", "balanced", "slow", "slowest"])
        q.add_argument("--grain", type=int, help="AV1 필름 그레인 합성 0~50")
        q.add_argument("--bit-depth", type=int, choices=[8, 10])
        q.add_argument("--container", choices=["mkv", "mp4"])
        q.add_argument("--audio", choices=["copy", "encode", "enhance"])
        q.add_argument("--audio-codec", choices=["opus", "aac", "flac", "ac3", "eac3"])
        q.add_argument("--audio-tracks", help="all|first|kor,eng")
        q.add_argument("--loudness", help="목표 LUFS (예: -23) 또는 off")
        q.add_argument("--dialog", type=float, help="대사 강조 dB (0=끔)")
        q.add_argument("--drc", choices=["off", "light", "night"])
        q.add_argument("--downmix", choices=["keep", "stereo", "stereo_dialog"])
        q.add_argument("--audio-denoise", choices=["off", "light", "strong"])
        q.add_argument("--keep-original-audio", action="store_true")
        q.add_argument("--subs", choices=["soft", "burn", "copy", "none"])
        q.add_argument("--sub-algo", choices=["auto", "contour", "xbr", "lanczos", "nearest"])
        q.add_argument("--sub-tracks", help="all|first|kor,eng")
        add_common(q)

    pe = sub.add_parser("encode", help="변환 실행")
    pe.add_argument("inputs", nargs="+")
    pe.add_argument("-o", "--output")
    pe.add_argument("--dry-run", action="store_true", help="명령만 출력")
    pe.add_argument("--sample", help="시험 변환: 'SECONDS' (영상 1/3 지점부터) 또는 'START,SECONDS'")
    add_options(pe)

    pp = sub.add_parser("preview", help="처리 전/후 비교 이미지")
    pp.add_argument("input")
    pp.add_argument("--time", type=float, default=0.0, help="초 (0=영상 중간)")
    pp.add_argument("-o", "--output")
    add_options(pp)

    ps = sub.add_parser("subs", help="이미지 자막(VobSub/PGS)만 업스케일해 .sup로 저장")
    ps.add_argument("input", help=".idx/.sup/.mkv/.vob")
    ps.add_argument("--stream", type=int, default=0, help="자막 스트림 번호 (절대 인덱스)")
    ps.add_argument("--size", default="1920x1080")
    ps.add_argument("--crop", help="원본 기준 영상 크롭 w:h:x:y")
    ps.add_argument("--algorithm", default="auto", choices=["auto", "contour", "xbr", "lanczos", "nearest"])
    ps.add_argument("-o", "--output")
    add_common(ps)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = make_parser()
    ns = parser.parse_args(argv)
    if not ns.cmd:
        parser.print_help()
        return 0
    try:
        return {"info": cmd_info, "caps": cmd_caps, "encode": cmd_encode, "subs": cmd_subs,
                "preview": cmd_preview}[ns.cmd](ns)
    except FFmpegNotFound as e:
        _log(str(e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
