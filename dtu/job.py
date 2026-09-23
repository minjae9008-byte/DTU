"""End-to-end conversion job: analysis, subtitle upscaling, loudness pass, encode."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from . import __version__
from .analyze import CropAnalysis, ScanAnalysis, analyze_crop, analyze_scan
from .audio import AudioTrackPlan, measure_loudness, plan_audio
from .codecs import CODECS, encoder_args, software_pix_fmt
from .ffmpeg import FFmpeg, FFmpegError, args_to_display, popen
from .inputs import InputSpec, resolve_input
from .probe import LANG_NAMES_KO, MediaInfo, SubtitleStream, lang3, probe_media
from .settings import Settings
from .subtitles.pipeline import Geometry, convert_track
from .subtitles.source import load_track
from .video import VideoPlan, build_video_plan, deinterlace_filters, resolve_deinterlace

ProgressCB = Callable[[str, float, str], None]  # (stage, fraction 0..1, detail)
LogCB = Callable[[str], None]


class Cancelled(Exception):
    pass


@dataclass
class SubPlan:
    stream: SubtitleStream
    action: str  # pgs | copy | burn
    forced_only: bool = False
    sup_path: Optional[str] = None
    events: int = 0
    title: Optional[str] = None
    default: bool = False


@dataclass
class JobPlan:
    info: MediaInfo
    settings: Settings
    scan: Optional[ScanAnalysis]
    crop: Optional[CropAnalysis]
    video: VideoPlan
    audio: List[AudioTrackPlan]
    subs: List[SubPlan]
    output: str

    def describe(self) -> str:
        v = self.video
        lines = [f"출력: {self.output}"]
        if self.scan:
            lines.append("스캔 분석: " + self.scan.describe())
        if self.crop:
            lines.append("여백 분석: " + self.crop.describe())
        lines.append(f"영상: {v.out_w}x{v.out_h} @ {float(v.out_fps or 0):.3f} fps, "
                     f"{CODECS[self.settings.encode.codec].label}")
        for n in v.notes:
            lines.append("  · " + n)
        for a in self.audio:
            if a.action == "copy":
                lines.append(f"오디오 #{a.stream.index}: 원본 그대로 복사 ({a.stream.codec})")
            else:
                lines.append(f"오디오 #{a.stream.index}: {a.codec.upper()} {a.bitrate or ''}"
                             f"{'kbps' if a.bitrate else ''} {a.out_channels}ch" +
                             (" - " + ", ".join(a.notes) if a.notes else ""))
        for s in self.subs:
            act = {"pgs": "업스케일 PGS 자막", "copy": "복사", "burn": "영상에 입히기(하드섭)"}[s.action]
            lines.append(f"자막 #{s.stream.index} ({s.stream.language or '미상'}): {act}"
                         + (" [강제 자막만]" if s.forced_only else ""))
        return "\n".join(lines)


def default_output_path(spec: InputSpec, s: Settings, sample: bool = False) -> str:
    ext = "mp4" if s.encode.container == "mp4" else "mkv"
    base_dir = s.output_dir or (spec.display if os.path.isdir(spec.display) else os.path.dirname(spec.display))
    suffix = s.name_suffix + ("_sample" if sample else "")
    out = os.path.join(base_dir, f"{spec.stem}{suffix}.{ext}")
    k = 2
    while os.path.abspath(out) in [os.path.abspath(p) for p in spec.paths] or \
            (os.path.exists(out) and not sample):
        out = os.path.join(base_dir, f"{spec.stem}{suffix}_{k}.{ext}")
        k += 1
    return out


def select_subs(info: MediaInfo, s: Settings) -> List[SubtitleStream]:
    subs = [x for x in info.subtitles if x.is_bitmap or x.is_text]
    spec = (s.subtitles.tracks or "all").strip().lower()
    if spec == "first":
        return subs[:1]
    if spec not in ("all", ""):
        wanted = {lang3(w.strip()) or w.strip() for w in spec.split(",") if w.strip()}
        subs = [x for x in subs if (x.language or "") in wanted]
    return subs


def _window_track(track, start: float, dur: float):
    """Copy of a subtitle track limited to [start, start+dur) and shifted to 0."""
    import copy
    t2 = copy.copy(track)
    pics = []
    for p in track.pictures:
        if p.end <= start or p.start >= start + dur:
            continue
        q = copy.copy(p)
        q.start = max(0.0, p.start - start)
        q.end = min(dur, p.end - start)
        pics.append(q)
    t2.pictures = pics
    return t2


class Job:
    def __init__(self, source, settings: Settings, output: Optional[str] = None,
                 ff: Optional[FFmpeg] = None, sample: Optional[Tuple[float, float]] = None):
        """``sample=(start, seconds)`` converts only that window (test encode)."""
        self.settings = settings
        self.sample = sample
        self.ff = ff or FFmpeg(folder=settings.ffmpeg_dir or None)
        if isinstance(source, InputSpec):
            self.spec = source
        else:
            self.spec = resolve_input(str(source), has_dvdvideo=self.ff.has_demuxer("dvdvideo"))
        self.output = output
        self.plan: Optional[JobPlan] = None
        self._proc: Optional[subprocess.Popen] = None
        self._cancel = threading.Event()
        self.temp_dir: Optional[str] = None
        self.last_command: List[str] = []

    def input_args(self) -> List[str]:
        """Main input arguments, with seek/limit for sample encodes."""
        a = self.spec.args()
        if self.sample is not None:
            start, dur = self.sample
            a = a[:-2] + ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}"] + a[-2:]
        return a

    # -- control ------------------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()
        p = self._proc
        if p and p.poll() is None:
            try:
                p.stdin.write(b"q")  # graceful stop keeps a playable file
                p.stdin.flush()
            except Exception:  # noqa: BLE001
                pass
            threading.Timer(5.0, lambda: p.poll() is None and p.kill()).start()

    def _check(self):
        if self._cancel.is_set():
            raise Cancelled()

    # -- analysis -----------------------------------------------------------
    def analyze(self, progress: Optional[ProgressCB] = None, log: Optional[LogCB] = None,
                scan: Optional[ScanAnalysis] = None, crop: Optional[CropAnalysis] = None) -> JobPlan:
        s = self.settings
        pr = (lambda msg: progress("analyze", 0.0, msg)) if progress else None
        if pr:
            pr("미디어 정보 읽는 중")
        info = probe_media(self.ff, self.spec)
        if info.main_video is None:
            raise FFmpegError("영상 스트림이 없습니다.")
        if log:
            log(info.summary())
        v = info.main_video
        if scan is None and s.video.deinterlace == "auto":
            scan = analyze_scan(self.ff, info, progress=pr)
            if log:
                log("스캔 분석: " + scan.describe())
        self._check()
        if crop is None and s.video.crop == "auto":
            mode = resolve_deinterlace(s.video, scan)
            pre, _ = deinterlace_filters(mode, v, "auto")
            crop = analyze_crop(self.ff, info, pre_filter=",".join(pre), progress=pr)
            if log and crop:
                log("여백 분석: " + crop.describe())
        vplan = build_video_plan(info, s.video, s.encode, scan, crop, self.ff)
        container = "mp4" if s.encode.container == "mp4" else "mkv"
        aplans = plan_audio(info, s.audio, container)
        subs: List[SubPlan] = []
        mode = s.subtitles.mode
        if mode != "none":
            chosen = select_subs(info, s)
            if mode == "burn":
                bitmap = [x for x in chosen if x.is_bitmap]
                if bitmap:
                    k = min(max(s.subtitles.burn_index, 0), len(bitmap) - 1)
                    subs.append(SubPlan(stream=bitmap[k], action="burn"))
            else:
                for x in chosen:
                    if x.is_bitmap and mode == "soft":
                        subs.append(SubPlan(stream=x, action="pgs", default=x.default))
                        if s.subtitles.forced_track:
                            subs.append(SubPlan(stream=x, action="pgs", forced_only=True))
                    elif x.is_bitmap and mode == "copy":
                        subs.append(SubPlan(stream=x, action="copy", default=x.default))
                    elif x.is_text and not x.external:
                        subs.append(SubPlan(stream=x, action="copy", default=x.default))
        out = self.output or default_output_path(self.spec, s, sample=self.sample is not None)
        if self.sample is not None:
            start, dur = self.sample
            if start < 0 or (info.duration and start >= info.duration):
                start = max(0.0, (info.duration or 0) / 3)
            self.sample = (start, max(1.0, min(dur, (info.duration or start + dur) - start)))
        self.plan = JobPlan(info=info, settings=s, scan=scan, crop=crop, video=vplan, audio=aplans,
                            subs=subs, output=out)
        return self.plan

    # -- subtitles ----------------------------------------------------------
    def _prepare_subtitles(self, progress: Optional[ProgressCB], log: Optional[LogCB]) -> None:
        plan = self.plan
        assert plan is not None and self.temp_dir
        info = plan.info
        vp = plan.video
        ifo_pal = info.ifo.palette if info.ifo else None
        cache = {}
        todo = [sp for sp in plan.subs if sp.action in ("pgs", "burn")]
        for n, sp in enumerate(todo):
            self._check()
            st = sp.stream
            key = (st.external, st.index)
            if key not in cache:
                if progress:
                    progress("subtitles", n / max(1, len(todo)), f"자막 #{st.index} 읽는 중")
                if st.external:
                    track = load_track(self.ff, ["-i", st.external], st.index)
                else:
                    track = load_track(self.ff, info.spec.args(), st.index, time_offset=info.start_time,
                                       ifo_palette=ifo_pal,
                                       canvas=(info.main_video.width, info.main_video.height))
                cache[key] = track
            track = cache[key]
            has_forced = any(p.forced for p in track.pictures)
            if sp.forced_only and (not has_forced or all(p.forced for p in track.pictures)):
                sp.events = 0  # no separate forced-only track needed
                continue
            # subtitle canvas -> video crop mapping (canvas usually equals the video frame)
            vw, vh = info.main_video.width, info.main_video.height
            cw_, ch_ = track.canvas
            fx, fy = cw_ / vw if vw else 1.0, ch_ / vh if vh else 1.0
            cx, cy, cw, ch = vp.crop or (0, 0, vw, vh)
            geom = Geometry(crop=(round(cx * fx), round(cy * fy), max(1, round(cw * fx)), max(1, round(ch * fy))),
                            out_w=vp.out_w, out_h=vp.out_h)
            sp.sup_path = os.path.join(self.temp_dir, f"sub_{st.index}{'_forced' if sp.forced_only else ''}.sup")

            def prog(i, total, _n=n):
                if progress:
                    progress("subtitles", (_n + i / max(1, total)) / max(1, len(todo)),
                             f"자막 #{st.index} 업스케일 {i}/{total}")

            if self.sample is not None:
                track = _window_track(track, *self.sample)
            sp.events = convert_track(track, geom, sp.sup_path, self.settings.subtitles.algorithm,
                                      forced_only=sp.forced_only, workers=self.settings.workers,
                                      progress=prog, cancel=self._cancel.is_set)
            lang = lang3(track.language) or st.language
            lang_ko = LANG_NAMES_KO.get(lang or "", lang or "") or "자막"
            sp.title = f"{lang_ko} 강제 자막" if sp.forced_only else f"{lang_ko} (업스케일)"
            if log:
                log(f"자막 #{st.index}: {sp.events}개 자막을 {vp.out_w}x{vp.out_h} PGS로 변환"
                    + (" (강제 자막 전용)" if sp.forced_only else ""))

    # -- command ------------------------------------------------------------
    def build_command(self) -> List[str]:
        """The single FFmpeg command that filters, encodes and muxes everything."""
        plan = self.plan
        assert plan is not None
        s = self.settings
        info = plan.info
        vp = plan.video
        ff = self.ff
        container = "mp4" if s.encode.container == "mp4" else "mkv"
        # no -nostdin: cancel() sends "q" on stdin for a clean stop
        args: List[str] = [ff.ffmpeg, "-hide_banner", "-y", "-progress", "pipe:1", "-nostats",
                           "-stats_period", "1"]
        if vp.uses_vulkan:
            args += ["-init_hw_device", "vulkan=vk", "-filter_hw_device", "vk"]
        args += self.input_args()
        extra_inputs: List[SubPlan] = []
        for sp in plan.subs:
            if sp.action in ("pgs", "burn") and sp.sup_path and sp.events > 0:
                args += ["-i", sp.sup_path]
                extra_inputs.append(sp)
        v = info.main_video
        graph = f"[0:{v.index}]{vp.filtergraph}"
        burn = [sp for sp in extra_inputs if sp.action == "burn"]
        if burn:
            k = extra_inputs.index(burn[0]) + 1
            pix = software_pix_fmt(CODECS[s.encode.codec], s.encode.bit_depth)
            spix = "yuva420p10le" if pix == "yuv420p10le" else "yuva420p"
            ofmt = "yuv420p10" if pix == "yuv420p10le" else "yuv420"
            mtx = "bt709" if vp.out_h > 576 else "bt601"
            graph += (f"[vmain];[{k}:0]scale=out_color_matrix={mtx}:out_range=tv,format={spix}[vsub];"
                      f"[vmain][vsub]overlay=eof_action=pass:format={ofmt},format={pix}")
        graph += "[vout]"
        args += ["-filter_complex", graph]

        # ---- video
        vargs = ["-map", "[vout]", "-disposition:v:0", "default"]
        venc = encoder_args(s.encode, vp.out_fps)
        while "-tag:v" in venc:  # added back below for MP4 only
            i = venc.index("-tag:v")
            del venc[i:i + 2]
        vargs += venc
        if container == "mp4" and CODECS[s.encode.codec].family == "hevc":
            vargs += ["-tag:v", "hvc1"]
        if vp.upscaled and s.video.color_convert and v.is_sd and vp.out_h > 576:
            vargs += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
        else:
            m, p, t, _ = v.source_colors()
            vargs += ["-color_primaries", p, "-color_trc", t, "-colorspace", m]
        vargs += ["-color_range", "tv"]

        # ---- audio
        aargs: List[str] = []
        for n, ap in enumerate(plan.audio):
            aargs += ["-map", f"0:{ap.stream.index}"]
            flt = ap.full_filters() if ap.action == "encode" else []
            if flt:
                aargs += [f"-filter:a:{n}", ",".join(flt)]
            aargs += ap.codec_args(n)
            lang = ap.stream.language
            if lang:
                aargs += [f"-metadata:s:a:{n}", f"language={lang}"]
            if ap.title:
                aargs += [f"-metadata:s:a:{n}", f"title={ap.title}"]
            aargs += [f"-disposition:a:{n}", "default" if ap.default else "0"]

        # ---- subtitles
        sargs: List[str] = []
        sn = 0
        for sp in plan.subs:
            if sp.action == "pgs":
                if not sp.sup_path or sp.events <= 0:
                    continue
                if container == "mp4":
                    continue  # PGS is not allowed in MP4: written next to the output instead
                k = extra_inputs.index(sp) + 1
                sargs += ["-map", f"{k}:0", f"-c:s:{sn}", "copy"]
            elif sp.action == "copy":
                if container == "mp4" and sp.stream.is_bitmap:
                    continue
                sargs += ["-map", f"0:{sp.stream.index}",
                          f"-c:s:{sn}", "mov_text" if container == "mp4" else "copy"]
            else:
                continue
            lang = sp.stream.language
            if lang:
                sargs += [f"-metadata:s:s:{sn}", f"language={lang}"]
            if sp.title:
                sargs += [f"-metadata:s:s:{sn}", f"title={sp.title}"]
            disp = "forced" if sp.forced_only else ("default" if sp.default else "0")
            sargs += [f"-disposition:s:{sn}", disp]
            sn += 1
        common = ["-map_chapters", "0", "-map_metadata", "0",
                  "-metadata", f"encoding_tool=DTU {__version__} (FFmpeg)", "-max_muxing_queue_size", "4096"]
        args += vargs + aargs + sargs + common
        if container == "mp4":
            args += ["-movflags", "+faststart"]
        args += [plan.output]
        return args

    # -- run ----------------------------------------------------------------
    def run(self, progress: Optional[ProgressCB] = None, log: Optional[LogCB] = None) -> str:
        t0 = time.time()
        if self.plan is None:
            self.analyze(progress, log)
        plan = self.plan
        assert plan is not None
        s = self.settings
        if log:
            log(plan.describe())
        out_dir = os.path.dirname(os.path.abspath(plan.output)) or "."
        os.makedirs(out_dir, exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(prefix="dtu_", dir=out_dir if s.keep_temp else None)
        try:
            self._check()
            self._prepare_subtitles(progress, log)
            self._check()
            if any(a.normalize for a in plan.audio):
                if progress:
                    progress("loudness", 0.0, "음량 측정 중 (EBU R128)")
                measure_loudness(self.ff, plan.info, plan.audio, input_args=self.input_args(),
                                 progress=(lambda f: progress("loudness", f, "음량 측정 중 (EBU R128)"))
                                 if progress else None, cancel=self._cancel.is_set)
                if log:
                    for a in plan.audio:
                        if a.normalize:
                            if a.measured_lufs is None:
                                log(f"오디오 #{a.stream.index}: 음량 측정 실패 - 정규화 생략")
                            else:
                                log(f"오디오 #{a.stream.index}: {a.measured_lufs:.1f} LUFS → "
                                    f"{a.target_lufs:g} LUFS ({a.gain_db:+.1f} dB)")
            self._check()
            cmd = self.build_command()
            self.last_command = cmd
            if log:
                log("FFmpeg 명령:\n" + args_to_display(cmd))
            try:
                self._encode(cmd, plan, progress, log)
            except BaseException:
                # never leave a half-written file that looks like a finished movie
                try:
                    if os.path.exists(plan.output):
                        os.remove(plan.output)
                        if log:
                            log(f"미완성 출력 파일 삭제: {plan.output}")
                except OSError:
                    pass
                raise
            if s.encode.container == "mp4":
                self._export_sidecar_subs(plan, log)
            if log:
                el = time.time() - t0
                log(f"완료: {plan.output} ({el / 60:.1f}분 소요)")
            return plan.output
        finally:
            if self.temp_dir and not s.keep_temp:
                shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _export_sidecar_subs(self, plan: JobPlan, log: Optional[LogCB]) -> None:
        base = os.path.splitext(plan.output)[0]
        for sp in plan.subs:
            if sp.action == "pgs" and sp.sup_path and sp.events > 0 and os.path.exists(sp.sup_path):
                lang = sp.stream.language or "und"
                dst = f"{base}.{lang}{'.forced' if sp.forced_only else ''}.sup"
                shutil.copyfile(sp.sup_path, dst)
                if log:
                    log(f"MP4는 PGS 자막을 담을 수 없어 별도 파일로 저장: {dst}")

    def _encode(self, cmd: List[str], plan: JobPlan, progress: Optional[ProgressCB],
                log: Optional[LogCB]) -> None:
        proc = popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._proc = proc
        tail: List[str] = []

        def drain():
            for raw in iter(proc.stderr.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                if not line:
                    continue
                tail.append(line)
                if len(tail) > 60:
                    del tail[:-60]
                if log and re.search(r"(error|warning|invalid|failed)", line, re.I) \
                        and not re.search(r"non monotonically|thread priority", line):
                    log("[ffmpeg] " + line)

        th = threading.Thread(target=drain, daemon=True)
        th.start()
        duration = self.sample[1] if self.sample else (plan.info.duration or 0.0)
        stats = {}
        started = time.time()
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if "=" not in line:
                continue
            k, _, val = line.partition("=")
            stats[k] = val
            if k == "progress":
                t = 0.0
                try:
                    t = int(stats.get("out_time_us", "0")) / 1e6
                except ValueError:
                    pass
                frac = min(1.0, t / duration) if duration > 0 else 0.0
                el = time.time() - started
                eta = (el / frac - el) if frac > 0.002 else 0
                detail = (f"{frac * 100:5.1f}%  {stats.get('fps', '0')} fps  속도 {stats.get('speed', '?')}"
                          f"  남은 시간 {int(eta // 3600)}:{int(eta % 3600 // 60):02d}:{int(eta % 60):02d}")
                if progress:
                    progress("encode", frac, detail)
        proc.wait()
        th.join(timeout=5)
        self._proc = None
        if self._cancel.is_set():
            raise Cancelled()
        if proc.returncode != 0:
            raise FFmpegError("FFmpeg 인코딩 실패:\n" + "\n".join(tail[-25:]), "\n".join(tail))
