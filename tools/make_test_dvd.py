#!/usr/bin/env python3
"""Create a DVD-like test file for trying DTU without a real disc.

Produces (in the output folder):
  test_dvd.mkv          MakeMKV-style: MPEG-2 720x480 16:9, AC3 5.1 + AC3 2.0,
                        two VobSub tracks (Korean with anti-aliasing, English with a forced line)
  test_dvd.vob          the same as a raw VOB (no IFO -> subtitle palette must be guessed)
  test_dvd_telecine.vob hard-telecined 29.97i version (for IVTC testing)

Needs Pillow (pip install pillow) and a Korean TrueType font for the Korean
track (NanumGothic, Malgun Gothic, AppleSDGothicNeo...; pass --font).

Usage:  python tools/make_test_dvd.py SOURCE_VIDEO OUT_DIR [--seconds 30]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dtu.subtitles.pgs import PgsEvent, write_sup  # noqa: E402

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

KO_LINES = [
    (1.0, 4.0, "안녕하세요. 이것은 테스트 자막입니다.", False),
    (4.5, 8.0, "DVD 자막은 720x480 해상도의|4가지 색으로 만들어져 있습니다.", False),
    (8.5, 12.0, "괜찮아? 뭐라고 했어?", False),
    (12.5, 16.0, "업스케일 후에도 글자가 선명해야 합니다!", False),
    (17.0, 21.0, "\"정말?\" 그래, 가자.", False),
    (22.0, 26.0, "마지막 자막입니다. 감사합니다.", False),
]
EN_LINES = [
    (1.0, 4.0, "Hello. This is a test subtitle.", False),
    (5.0, 8.0, "[SIGN: CLOSED]", True),
    (9.0, 12.5, "The quick brown fox jumps|over the lazy dog.", False),
    (13.0, 16.0, "\"Really?\" Yes, let's go!", False),
    (18.0, 22.0, "Upscaled subtitles should look sharp.", False),
]


def find_font(explicit):
    for p in ([explicit] if explicit else []) + FONT_CANDIDATES:
        if p and os.path.exists(p):
            return p
    raise SystemExit("폰트를 찾을 수 없습니다. --font 로 TTF 경로를 지정하세요.")


def render_dvd_sub(text: str, font_path: str, aa: bool, color=(235, 235, 235)):
    """Render text like a DVD authoring tool: 4 colours, anamorphic 16:9 squeeze."""
    from PIL import Image, ImageDraw, ImageFont
    ss = 4
    size = 30
    stroke = 2
    font = ImageFont.truetype(font_path, size * ss)
    lines = text.split("|")
    W = 853 * ss  # square-pixel width of a 16:9 480-line frame
    H = int(len(lines) * size * 1.35 * ss + 8 * ss)
    fill = Image.new("L", (W, H), 0)
    outer = Image.new("L", (W, H), 0)
    df, do = ImageDraw.Draw(fill), ImageDraw.Draw(outer)
    y = 4 * ss
    for line in lines:
        tw = df.textlength(line, font=font)
        x = (W - tw) / 2
        do.text((x, y), line, font=font, fill=255, stroke_width=stroke * ss, stroke_fill=255)
        df.text((x, y), line, font=font, fill=255)
        y += int(size * 1.35 * ss)
    # coverage on the anamorphic 720-wide grid
    fa = np.asarray(fill, np.float64) / 255
    oa = np.asarray(outer, np.float64) / 255
    h_out = H // ss

    def area(img):
        """Box-average onto a 720-wide grid (each output pixel covers W/720 inputs)."""
        img = img[:h_out * ss].reshape(h_out, ss, W).mean(axis=1)
        cs = np.concatenate([np.zeros((img.shape[0], 1)), np.cumsum(img, axis=1)], axis=1)
        xs = np.linspace(0, W, 721)
        i = np.minimum(np.floor(xs).astype(int), W - 1)
        f = xs - i
        c = cs[:, i] + f * (cs[:, i + 1] - cs[:, i])
        return (c[:, 1:] - c[:, :-1]) / (W / 720)
    cf, co = area(fa), np.maximum(area(oa), area(fa))
    cls = np.zeros(cf.shape, np.uint8)
    cls[co >= 0.5] = 2
    if aa:
        cls[cf >= 0.25] = 3
        cls[cf >= 0.75] = 1
    else:
        cls[cf >= 0.5] = 1
    pal = np.array([[0, 0, 0, 0], [*color, 255], [16, 16, 16, 255], [128, 128, 128, 255]], np.uint8)
    rows = np.flatnonzero(cls.any(axis=1))
    cols = np.flatnonzero(cls.any(axis=0))
    cls = cls[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return cls, pal, int(cols[0])


def make_sup(lines, path, font, aa, color=(235, 235, 235)):
    events = []
    for start, end, text, forced in lines:
        cls, pal, x = render_dvd_sub(text, font, aa, color)
        h = cls.shape[0]
        y = 480 - 40 - h
        events.append(PgsEvent(start, end, x, y, cls, pal, forced))
    # anchored at t=0: FFmpeg shifts every input by its first timestamp
    write_sup(path, events, 720, 480, anchor_zero=True)


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("out_dir")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--font")
    ns = ap.parse_args()
    os.makedirs(ns.out_dir, exist_ok=True)
    font = find_font(ns.font)
    tmp = tempfile.mkdtemp()
    ko, en = os.path.join(tmp, "ko.sup"), os.path.join(tmp, "en.sup")
    make_sup(KO_LINES, ko, font, aa=True)
    make_sup(EN_LINES, en, font, aa=False, color=(235, 235, 40))
    base = os.path.join(tmp, "base.mkv")
    # DVD-like picture: 720x480 anamorphic 16:9, BT.601, a little grain, MPEG-2 ~6 Mbps
    vf = ("scale=720:480:flags=lanczos:out_color_matrix=bt601,setsar=32/27,"
          "noise=c0s=6:c0f=t+u,format=yuv420p")
    run(["ffmpeg", "-v", "error", "-y", "-i", ns.source, "-t", str(ns.seconds), "-vf", vf, "-r", "24000/1001",
         "-c:v", "ffv1", "-af", "surround=chl_out=5.1(side),aresample=48000", "-c:a", "pcm_s16le", base])
    common = ["-map", "0:v", "-map", "0:a", "-map", "0:a", "-map", "1:0", "-map", "2:0",
              "-c:v", "mpeg2video", "-b:v", "6000k", "-maxrate", "9000k", "-bufsize", "1835k", "-g", "15",
              "-bf", "2", "-aspect", "16:9", "-color_primaries", "smpte170m", "-color_trc", "smpte170m",
              "-colorspace", "smpte170m",
              "-c:a:0", "ac3", "-b:a:0", "448k", "-c:a:1", "ac3", "-b:a:1", "192k", "-ac:a:1", "2",
              "-c:s", "dvdsub",
              "-metadata:s:a:0", "language=kor", "-metadata:s:a:1", "language=eng",
              "-metadata:s:s:0", "language=kor", "-metadata:s:s:1", "language=eng"]
    # -fix_sub_duration: PGS "clear" events become real end times of the DVD subtitles
    subs_in = ["-fix_sub_duration", "-i", ko, "-fix_sub_duration", "-i", en]
    run(["ffmpeg", "-v", "error", "-y", "-i", base, *subs_in, *common, "-r", "24000/1001",
         os.path.join(ns.out_dir, "test_dvd.mkv")])
    run(["ffmpeg", "-v", "error", "-y", "-i", base, *subs_in, *common, "-r", "24000/1001",
         "-f", "vob", os.path.join(ns.out_dir, "test_dvd.vob")])
    run(["ffmpeg", "-v", "error", "-y", "-i", base, *subs_in, *common,
         "-vf", "telecine=first_field=top:pattern=23", "-flags", "+ilme+ildct", "-r", "30000/1001",
         "-f", "vob", os.path.join(ns.out_dir, "test_dvd_telecine.vob")])
    print("done:", ns.out_dir)


if __name__ == "__main__":
    main()
