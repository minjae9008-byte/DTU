#!/usr/bin/env python3
"""Codec efficiency on upscaled DVD material (run from the corpus folder).

A lossless 1080p master is made with DTU's default video chain; each codec
then encodes it with DTU's default quality and 'balanced' speed and is scored
with VMAF against the master.
"""
import json, os, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dtu.codecs import CODECS, encoder_args  # noqa: E402
from dtu.settings import EncodeSettings  # noqa: E402

CHAIN = ("hqdn3d=2:1.5:3:2.25,zscale=w=1920:h=1080:filter=lanczos:param_a=3:matrixin=170m:transferin=601:"
         "primariesin=170m:rangein=limited:chromalin=left:matrix=709:transfer=709:primaries=709:range=limited:"
         "dither=error_diffusion,format=yuv420p10le,deband=1thr=0.012:2thr=0.012:3thr=0.012:4thr=0.012:"
         "range=20:blur=1,cas=strength=0.40:planes=1")
CLIPS = ["dvd/sintel_clean.mpg", "dvd/park_joy_grain.mpg", "dvd/old_town_cross_clean.mpg"]


def run(args):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-800:])
    return p


def main(codecs):
    tmp = tempfile.mkdtemp()
    masters = []
    for c in CLIPS:
        m = os.path.join(tmp, os.path.basename(c) + ".mkv")
        run(["ffmpeg", "-v", "error", "-y", "-i", c, "-vf", CHAIN, "-c:v", "ffv1", m])
        masters.append(m)
    for spec in codecs:
        key, _, q = spec.partition(":")  # e.g. "av1:30" overrides the default quality
        tot_bits = tot_dur = 0.0
        vmafs, secs = [], 0.0
        for m in masters:
            out = os.path.join(tmp, f"{key}.mkv")
            es = EncodeSettings(codec=key, speed="balanced", quality=int(q) if q else -1)
            t = time.time()
            run(["ffmpeg", "-v", "error", "-y", "-i", m, *encoder_args(es), "-an", out])
            secs += time.time() - t
            info = json.loads(run(["ffprobe", "-v", "error", "-of", "json", "-show_format", out]).stdout)["format"]
            tot_bits += float(info["size"]) * 8
            tot_dur += float(info["duration"])
            log = os.path.join(tmp, "v.json")
            run(["ffmpeg", "-v", "error", "-i", out, "-i", m, "-filter_complex",
                 f"[0:v]format=yuv420p10le[a];[1:v]format=yuv420p10le[b];[a][b]libvmaf=n_threads=4:log_fmt=json:log_path={log}",
                 "-f", "null", "-"])
            vmafs.append(json.load(open(log))["pooled_metrics"]["vmaf"]["mean"])
        print(f"{key:8s} q={int(q) if q else CODECS[key].default_quality:3d}  {tot_bits / tot_dur / 1000:7.0f} kbps  "
              f"VMAF {sum(vmafs) / len(vmafs):6.2f}  encode {secs:6.1f}s  {['%.1f' % v for v in vmafs]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["av1", "hevc", "h264", "vvc"])
