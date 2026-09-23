"""Convert a bitmap subtitle track to an upscaled PGS (.sup) file."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .pgs import PgsEvent, write_sup
from .quantize import quantize, trim_indexed
from .source import SubPicture, SubTrack
from .upscale import upscale_picture


@dataclass
class Geometry:
    """Maps subtitle canvas coordinates onto the output video frame.

    ``crop`` is the region of the *source video frame* that ends up in the
    output (x, y, w, h), expressed in subtitle canvas pixels.
    """

    crop: Tuple[int, int, int, int]
    out_w: int
    out_h: int

    @property
    def sx(self) -> float:
        return self.out_w / self.crop[2]

    @property
    def sy(self) -> float:
        return self.out_h / self.crop[3]


def _convert_one(args) -> Optional[Tuple[float, float, int, int, np.ndarray, np.ndarray, bool]]:
    pic, geom, algorithm = args
    cx, cy = geom.crop[0], geom.crop[1]
    fx = (pic.x - cx) * geom.sx
    fy = (pic.y - cy) * geom.sy
    pm, x0, y0 = upscale_picture(pic.rgba, pic.classes, pic.class_rgba, geom.sx, geom.sy, fx, fy,
                                 algorithm)
    idx, pal = quantize(pm)
    idx, tx, ty = trim_indexed(idx)
    if idx.size == 0:
        return None
    x0 += tx
    y0 += ty
    h, w = idx.shape
    if w > geom.out_w or h > geom.out_h:  # larger than the frame: cut to fit
        idx = idx[:geom.out_h, :geom.out_w]
        h, w = idx.shape
    # Subtitles that sat in cropped-away black bars are moved into the picture,
    # keeping a safe margin from the frame edge (like authored Blu-ray subtitles).
    mx = int(round(geom.out_w * 0.03))
    my = int(round(geom.out_h * 0.045))
    if x0 < 0 or x0 + w > geom.out_w:
        x0 = min(max(x0, mx), max(geom.out_w - w - mx, 0))
    if y0 < 0 or y0 + h > geom.out_h:
        y0 = min(max(y0, my), max(geom.out_h - h - my, 0))
    return pic.start, pic.end, x0, y0, idx, pal, pic.forced


def convert_pictures(pictures: Sequence[SubPicture], geom: Geometry, algorithm: str = "auto",
                     workers: int = 0, progress: Optional[Callable[[int, int], None]] = None,
                     cancel: Optional[Callable[[], bool]] = None) -> List[PgsEvent]:
    jobs = [(p, geom, algorithm) for p in pictures if p.end > p.start]
    out: List[PgsEvent] = []
    n = len(jobs)
    if workers <= 0:
        workers = max(1, min(4, (os.cpu_count() or 2) - 1))

    def collect(i, r):
        if r is not None:
            s, e, x, y, idx, pal, forced = r
            out.append(PgsEvent(s, e, x, y, idx, pal, forced))
        if progress:
            progress(i + 1, n)

    if workers == 1 or n < 8:
        for i, j in enumerate(jobs):
            if cancel and cancel():
                raise KeyboardInterrupt
            collect(i, _convert_one(j))
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for i, r in enumerate(ex.map(_convert_one, jobs, chunksize=4)):
                    if cancel and cancel():
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise KeyboardInterrupt
                    collect(i, r)
        except (OSError, RuntimeError):  # e.g. no multiprocessing available
            out.clear()
            for i, j in enumerate(jobs):
                collect(i, _convert_one(j))
    out.sort(key=lambda e: e.start)
    return out


def convert_track(track: SubTrack, geom: Geometry, sup_path: str, algorithm: str = "auto",
                  forced_only: bool = False, workers: int = 0,
                  progress: Optional[Callable[[int, int], None]] = None,
                  cancel: Optional[Callable[[], bool]] = None) -> int:
    pics = [p for p in track.pictures if p.forced] if forced_only else list(track.pictures)
    events = convert_pictures(pics, geom, algorithm, workers, progress, cancel)
    return write_sup(sup_path, events, geom.out_w, geom.out_h)
