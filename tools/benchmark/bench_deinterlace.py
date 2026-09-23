#!/usr/bin/env python3
"""Deinterlacing and inverse-telecine benchmark (run from the corpus folder).

Deinterlacers turn 576i/480i MPEG-2 (made from real 50p/60p footage) back into
double-rate progressive video and are scored against the original frames.
IVTC is scored against the 24p frames the telecined clip was made from.
Frames are compared directly (no timestamp matching) so dropped/duplicated
frames can't shift the comparison.
"""
import subprocess
import sys

import numpy as np


def raw(args, w, h):
    p = subprocess.run(["ffmpeg", "-v", "error", *args, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w).astype(np.float64)


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return 10 * np.log10(255 ** 2 / max(mse, 1e-9))


def best_offset(out, ref, skip=4):
    n = min(len(out), len(ref)) - 3
    res = []
    for off in (-2, -1, 0, 1, 2):
        idx = [i for i in range(skip, n) if 0 <= i + off < len(ref)]
        res.append((np.mean([psnr(out[i], ref[i + off]) for i in idx]), off))
    return max(res)


def deint():
    cases = [("dvd/park_joy_576i.mpg", "gt/park_joy_576p50.mkv", 720, 576),
             ("dvd/crowd_run_480i.mpg", "gt/crowd_run_480p60.mkv", 720, 480)]
    cands = {
        "bwdif": "bwdif=mode=send_field:parity=auto:deint=all",
        "w3fdif": "w3fdif=filter=complex:mode=field",
        "estdif": "estdif=mode=field",
        "yadif": "yadif=mode=send_field:parity=auto:deint=all",
        "bob (field resize)": "separatefields,scale=iw:ih*2:flags=bicubic",
    }
    for name, chain in cands.items():
        vals = []
        for src, gt, w, h in cases:
            out = raw(["-i", src, "-vf", chain], w, h)
            ref = raw(["-i", gt], w, h)
            vals.append(best_offset(out, ref)[0])
        print(f"deint {name:20s} PSNR {np.mean(vals):6.2f} dB  {['%.2f' % v for v in vals]}", flush=True)


def ivtc():
    ref = raw(["-i", "gt/sintel_480p24.mkv"], 720, 480)
    cands = {
        "fps+fieldmatch+bwdif+decimate": "fps=30000/1001,fieldmatch=order=auto:combmatch=full,"
                                         "bwdif=mode=send_frame:deint=interlaced,decimate",
        "pullup": "pullup,fps=24000/1001",
        "deinterlace+drop (no IVTC)": "bwdif=mode=send_frame,fps=24000/1001",
    }
    for name, chain in cands.items():
        out = raw(["-i", "dvd/sintel_telecine.mpg", "-vf", chain], 720, 480)
        p, off = best_offset(out, ref)
        print(f"ivtc  {name:30s} PSNR {p:6.2f} dB (offset {off}, {len(out)} frames)", flush=True)
    out = raw(["-i", "dvd/sintel_clean.mpg"], 720, 480)
    print(f"ivtc  {'reference: progressive encode':30s} PSNR {best_offset(out, ref)[0]:6.2f} dB", flush=True)


if __name__ == "__main__":
    {"deint": deint, "ivtc": ivtc}[sys.argv[1] if len(sys.argv) > 1 else "deint"]()
