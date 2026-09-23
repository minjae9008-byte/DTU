"""Before/after preview frames (numpy RGB) for the GUI and the CLI."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ffmpeg import FFmpegError, run
from .job import Job
from .subtitles.imageio import write_png
from .subtitles.pipeline import Geometry, _convert_one
from .subtitles.source import SubPicture, load_track
from .video import deinterlace_filters

_TRACK_CACHE: Dict[tuple, object] = {}


@dataclass
class Preview:
    before: np.ndarray
    after: np.ndarray
    time: float
    size: Tuple[int, int]
    notes: List[str] = field(default_factory=list)
    subtitle_text: str = ""


def _render(job: Job, t: float, chain: str, w: int, h: int, matrix: str, vulkan: bool) -> np.ndarray:
    ff = job.ff
    info = job.plan.info
    pre = min(t, 2.0)  # pre-roll for temporal filters (IVTC, denoise, interpolation)
    v = info.main_video
    args = [ff.ffmpeg, "-hide_banner", "-nostdin", "-v", "error"]
    if vulkan:
        args += ["-init_hw_device", "vulkan=vk", "-filter_hw_device", "vk"]
    spec_args = info.spec.args()
    args += ["-ss", f"{max(t - pre, 0):.3f}", *spec_args, "-ss", f"{pre:.3f}", "-frames:v", "1",
             "-filter_complex",
             f"[0:{v.index}]{chain},scale={w}:{h}:flags=bicubic:in_color_matrix={matrix}:in_range=tv"
             f":out_range=pc,format=rgb24[o]", "-map", "[o]", "-f", "rawvideo", "-"]
    p = run(args, timeout=600)
    need = w * h * 3
    if p.returncode != 0 or len(p.stdout) < need:
        raise FFmpegError("미리보기 생성 실패: " + p.stderr.decode("utf-8", "replace")[-800:])
    return np.frombuffer(p.stdout[:need], np.uint8).reshape(h, w, 3).copy()


def _composite(img: np.ndarray, pm: np.ndarray, x0: int, y0: int) -> None:
    """Alpha-blend a premultiplied float RGBA image onto an RGB uint8 frame in place."""
    H, W = img.shape[:2]
    h, w = pm.shape[:2]
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, W), min(y0 + h, H)
    if xa >= xb or ya >= yb:
        return
    part = pm[ya - y0:yb - y0, xa - x0:xb - x0]
    a = np.clip(part[..., 3:4], 0, 255) / 255.0
    dst = img[ya:yb, xa:xb].astype(np.float64)
    img[ya:yb, xa:xb] = np.clip(part[..., :3] + dst * (1 - a), 0, 255).astype(np.uint8)


def _subtitle_at(job: Job, t: float) -> Optional[Tuple[object, SubPicture]]:
    plan = job.plan
    subs = [sp for sp in plan.subs if sp.action in ("pgs", "burn") and not sp.forced_only]
    if not subs:
        return None
    st = subs[0].stream
    key = (tuple(plan.info.spec.paths), st.external, st.index)
    track = _TRACK_CACHE.get(key)
    if track is None:
        info = plan.info
        if st.external:
            track = load_track(job.ff, ["-i", st.external], st.index)
        else:
            track = load_track(job.ff, info.spec.args(), st.index, time_offset=info.start_time,
                               ifo_palette=info.ifo.palette if info.ifo else None,
                               canvas=(info.main_video.width, info.main_video.height))
        _TRACK_CACHE[key] = track
    pics = track.pictures
    if not pics:
        return None
    active = [p for p in pics if p.start <= t < p.end]
    return track, (active[0] if active else min(pics, key=lambda p: abs(p.start - t)))


def make_preview(job: Job, t: Optional[float] = None, with_subtitles: bool = True) -> Preview:
    plan = job.plan or job.analyze()
    info = plan.info
    v = info.main_video
    vp = plan.video
    if t is None or t <= 0:
        t = (info.duration or 60) / 3
    t = min(t, max(0.0, (info.duration or t) - 1.0))
    s = job.settings
    W, H = vp.out_w, vp.out_h
    # "before": what a normal player shows (deinterlace + crop + bicubic upscale)
    mode = vp.deinterlace_mode
    pre, _ = deinterlace_filters(mode if mode != "soft_telecine" else "off", v, "auto")
    cx, cy, cw, ch = vp.crop or (0, 0, v.width, v.height)
    before_chain = ",".join(pre + ([f"crop={cw}:{ch}:{cx}:{cy}"] if (cw, ch) != (v.width, v.height) else [])) or "null"
    src_matrix = "bt601" if v.is_sd else "bt709"
    before = _render(job, t, before_chain, W, H, src_matrix, False)
    out_matrix = "bt709" if (vp.upscaled and s.video.color_convert and v.is_sd and H > 576) else src_matrix
    after = _render(job, t, vp.filtergraph, W, H, out_matrix, vp.uses_vulkan)
    text = ""
    if with_subtitles and s.subtitles.mode in ("soft", "burn"):
        try:
            found = _subtitle_at(job, t)
        except Exception:  # noqa: BLE001 - preview must not fail because of subtitles
            found = None
        if found:
            track, pic = found
            vw, vh = v.width, v.height
            fx, fy = track.canvas[0] / vw, track.canvas[1] / vh
            geom = Geometry((round(cx * fx), round(cy * fy), round(cw * fx), round(ch * fy)), W, H)
            r = _convert_one((pic, geom, s.subtitles.algorithm))
            if r is not None:
                _s, _e, x0, y0, idx, pal, _f = r
                pm = pal.astype(np.float64)[idx]
                pm[..., :3] *= pm[..., 3:4] / 255.0
                _composite(after, pm, x0, y0)
                # the player's view: 720x480 bitmap scaled with the video (bilinear)
                bx = (pic.x - geom.crop[0]) * geom.sx
                by = (pic.y - geom.crop[1]) * geom.sy
                from .subtitles.upscale import upscale_picture
                pmb, bx0, by0 = upscale_picture(pic.rgba, None, None, geom.sx, geom.sy, bx, by, "lanczos")
                vis = np.argwhere(pmb[..., 3] > 8)
                if vis.size:  # same on-screen position as the processed subtitle
                    bx0 = x0 - int(vis[:, 1].min())
                    by0 = y0 - int(vis[:, 0].min())
                _composite(before, pmb, bx0, by0)
                text = f"자막 {pic.start:.1f}s~{pic.end:.1f}s"
    return Preview(before=before, after=after, time=t, size=(W, H), notes=list(vp.notes), subtitle_text=text)


def save_side_by_side(pv: Preview, path: str) -> str:
    sep = np.full((pv.before.shape[0], 8, 3), 255, np.uint8)
    write_png(path, np.concatenate([pv.before, sep, pv.after], axis=1), level=3)
    return path


def make_preview_files(source: str, settings, t: float, out_path: str):
    job = Job(source, settings)
    job.analyze()
    pv = make_preview(job, t)
    base = out_path[:-4] if out_path.lower().endswith(".png") else out_path
    write_png(base + "_before.png", pv.before, level=3)
    write_png(base + "_after.png", pv.after, level=3)
    save_side_by_side(pv, base + "_compare.png")
    return base + "_before.png", base + "_after.png"
