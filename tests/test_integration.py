"""End-to-end tests with real FFmpeg (skipped when FFmpeg is not installed)."""
import os
import re
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import glyph_classes  # noqa: E402

from dtu.ffmpeg import FFmpeg, FFmpegNotFound

try:
    FF = FFmpeg()
except FFmpegNotFound:  # pragma: no cover
    FF = None

pytestmark = pytest.mark.skipif(FF is None, reason="FFmpeg not installed")


def ffmpeg(*args):
    subprocess.run([FF.ffmpeg, "-hide_banner", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def dvd_mkv(tmp_path_factory):
    """4 s MPEG-2 720x480 16:9 + AC3 5.1 + one VobSub track, like a MakeMKV rip."""
    from dtu.subtitles.pgs import PgsEvent, write_sup
    d = tmp_path_factory.mktemp("dvd")
    cls, pal = glyph_classes(aa=True)
    sup = str(d / "subs.sup")
    write_sup(sup, [PgsEvent(0.5, 1.5, 300, 400, cls, pal), PgsEvent(2.0, 3.5, 320, 400, cls, pal, forced=True)],
              720, 480)
    out = str(d / "dvd.mkv")
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=720x480:r=24000/1001:d=4",
           "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:d=4",
           "-fix_sub_duration", "-i", sup,
           "-filter_complex", "[1:a]volume=-12dB,pan=5.1(side)|FL=c0|FR=c0|FC=c0|LFE=0*c0|SL=0.5*c0|SR=0.5*c0[a]",
           "-map", "0:v", "-map", "[a]", "-map", "2:0",
           "-c:v", "mpeg2video", "-b:v", "5M", "-aspect", "16:9", "-c:a", "ac3", "-b:a", "448k",
           "-c:s", "dvdsub", "-metadata:s:a:0", "language=kor", "-metadata:s:s:0", "language=kor", out)
    return out


def fast_settings():
    from dtu.settings import preset_settings
    s = preset_settings("fast")
    s.encode.codec = "h264" if FF.has_encoder("libx264") else "av1"
    s.encode.speed = "fastest"
    return s


def test_probe_and_plan(dvd_mkv):
    from dtu.job import Job
    job = Job(dvd_mkv, fast_settings(), ff=FF)
    plan = job.analyze()
    v = plan.info.main_video
    assert (v.width, v.height) == (720, 480)
    assert plan.scan.scan == "progressive"
    assert (plan.video.out_w, plan.video.out_h) == (1920, 1080)
    assert [a.stream.channels for a in plan.audio] == [6]
    assert [s.action for s in plan.subs] == ["pgs", "pgs"]  # full track + forced-only track


def test_full_encode(dvd_mkv, tmp_path):
    from dtu.job import Job
    out = str(tmp_path / "out.mkv")
    job = Job(dvd_mkv, fast_settings(), output=out, ff=FF)
    logs = []
    assert job.run(log=logs.append) == out
    data = FF.probe(["-i", out], ["-show_streams"])
    types = [(s["codec_type"], s["codec_name"]) for s in data["streams"]]
    assert types[0][0] == "video" and ("audio", "opus") in types
    subs = [s for s in data["streams"] if s["codec_type"] == "subtitle"]
    assert len(subs) == 2 and all(s["codec_name"] == "hdmv_pgs_subtitle" for s in subs)
    assert subs[1]["disposition"]["forced"] == 1
    v = data["streams"][0]
    assert (v["width"], v["height"]) == (1920, 1080) and v.get("color_space") == "bt709"
    # loudness normalised to -23 LUFS
    p = subprocess.run([FF.ffmpeg, "-hide_banner", "-nostdin", "-i", out, "-map", "0:a:0", "-af",
                        "ebur128=framelog=quiet", "-f", "null", "-"], capture_output=True, text=True)
    lufs = float(re.findall(r"^\s*I:\s*(-?[\d.]+) LUFS", p.stderr, re.M)[-1])
    assert abs(lufs - (-23.0)) < 1.5
    # the upscaled subtitle decodes at the output resolution
    p = subprocess.run([FF.ffmpeg, "-hide_banner", "-v", "error", "-i", out, "-filter_complex",
                        "[0:s:0]format=rgba,crop=1920:1080[s]", "-map", "[s]", "-ss", "1", "-frames:v", "1",
                        "-f", "rawvideo", "-"], capture_output=True)
    img = np.frombuffer(p.stdout, np.uint8).reshape(1080, 1920, 4)
    assert img[..., 3].max() == 255


def test_sample_encode_and_burn_in(dvd_mkv, tmp_path):
    from dtu.job import Job
    s = fast_settings()
    s.subtitles.mode = "burn"
    s.audio.mode = "copy"
    out = str(tmp_path / "burn.mkv")
    job = Job(dvd_mkv, s, output=out, ff=FF, sample=(0.0, 2.0))
    job.run()
    data = FF.probe(["-i", out], ["-show_streams", "-show_format"])
    assert not [x for x in data["streams"] if x["codec_type"] == "subtitle"]
    assert float(data["format"]["duration"]) < 3.0
    assert [x["codec_name"] for x in data["streams"] if x["codec_type"] == "audio"] == ["ac3"]


def test_scan_detection_on_synthetic_telecine_and_interlace(tmp_path):
    from dtu.analyze import analyze_scan
    from dtu.inputs import resolve_input
    from dtu.probe import probe_media
    src = "testsrc2=s=720x480:r=24000/1001:d=6"
    tele = str(tmp_path / "tele.mpg")
    ffmpeg("-f", "lavfi", "-i", src, "-vf", "telecine=first_field=top:pattern=23", "-c:v", "mpeg2video",
           "-b:v", "8M", "-flags", "+ilme+ildct", "-r", "30000/1001", "-f", "vob", tele)
    inter = str(tmp_path / "inter.mpg")
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=720x480:r=60000/1001:d=4", "-vf", "interlace=scan=tff",
           "-c:v", "mpeg2video", "-b:v", "8M", "-flags", "+ilme+ildct", "-r", "30000/1001", "-f", "vob", inter)
    prog = str(tmp_path / "prog.mpg")
    ffmpeg("-f", "lavfi", "-i", src, "-c:v", "mpeg2video", "-b:v", "8M", "-f", "vob", prog)
    dup = str(tmp_path / "dup.mpg")  # 24p padded to 29.97p by repeating frames (cheap DVD authoring)
    ffmpeg("-f", "lavfi", "-i", src, "-vf", "fps=30000/1001", "-c:v", "mpeg2video", "-b:v", "8M", "-f", "vob", dup)
    res = {}
    for name, path in (("tele", tele), ("inter", inter), ("prog", prog), ("dup", dup)):
        res[name] = analyze_scan(FF, probe_media(FF, resolve_input(path))).scan
    assert res == {"tele": "telecine", "inter": "interlaced", "prog": "progressive", "dup": "dup_frames"}


def test_soft_telecine_detected_and_restored(tmp_path):
    """Progressive 24p pictures with repeat-field flags (the usual NTSC film DVD)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from soft_pulldown import apply_pulldown
    from dtu.analyze import analyze_scan
    from dtu.inputs import resolve_input
    from dtu.job import Job
    from dtu.probe import probe_media
    m2v = tmp_path / "in.m2v"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=720x480:r=24000/1001:d=3", "-c:v", "mpeg2video", "-bf", "0",
           "-b:v", "6M", str(m2v))
    data = bytearray(m2v.read_bytes())
    assert apply_pulldown(data) == 72
    soft = tmp_path / "soft.m2v"
    soft.write_bytes(bytes(data))
    vob = str(tmp_path / "soft.vob")
    ffmpeg("-i", str(soft), "-c", "copy", "-f", "vob", vob)
    info = probe_media(FF, resolve_input(vob))
    assert analyze_scan(FF, info).scan == "soft_telecine"
    out = str(tmp_path / "soft.mkv")
    s = fast_settings()
    s.subtitles.mode = "none"
    Job(vob, s, output=out, ff=FF).run()
    st = FF.probe(["-i", out], ["-count_frames", "-show_streams", "-select_streams", "v:0"])["streams"][0]
    assert st["r_frame_rate"] == "24000/1001" and int(st["nb_read_frames"]) == 72


def test_crop_detection(tmp_path):
    from dtu.analyze import analyze_crop
    from dtu.inputs import resolve_input
    from dtu.probe import probe_media
    lb = str(tmp_path / "lb.mpg")
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=720x360:r=25:d=3", "-vf", "pad=720:480:0:60", "-c:v", "mpeg2video",
           "-b:v", "6M", "-f", "vob", lb)
    crop = analyze_crop(FF, probe_media(FF, resolve_input(lb)))
    assert (crop.width, crop.height, crop.x, crop.y) == (720, 360, 0, 60)


def test_preview_frames(dvd_mkv):
    from dtu.job import Job
    from dtu.preview import make_preview
    job = Job(dvd_mkv, fast_settings(), ff=FF)
    job.analyze()
    pv = make_preview(job, 1.0)
    assert pv.before.shape == pv.after.shape == (1080, 1920, 3)
    assert pv.subtitle_text  # a subtitle was found and composited
