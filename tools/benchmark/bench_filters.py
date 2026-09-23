#!/usr/bin/env python3
"""Benchmark sharpening / deblocking / denoising chains before+after upscaling."""
import sys, time, subprocess, os
from bench import score, CORPUS

GRAIN = [(f"dvd/{n}_grain.mpg", f"gt/{n}_1080p24.mkv") for n in ("old_town_cross", "park_joy", "crowd_run", "ducks_take_off")]
CLEAN = [(f"dvd/{n}_clean.mpg", f"gt/{n}_1080p24.mkv") for n in ("old_town_cross", "park_joy", "crowd_run", "ducks_take_off")] + \
        [("dvd/sintel_clean.mpg", "gt/sintel_1080p24.mkv")]
ZIN = "matrixin=170m:transferin=601:primariesin=170m:rangein=limited:chromalin=left"
ZOUT = "matrix=709:transfer=709:primaries=709:range=limited"
UP_S36 = f"zscale=w=1920:h=1080:filter=spline36:{ZIN}:{ZOUT}"
UP_L3 = f"zscale=w=1920:h=1080:filter=lanczos:param_a=3:{ZIN}:{ZOUT}"


def fps_of(chain, src="dvd/park_joy_grain.mpg", pre=""):
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", *pre.split(), "-i", os.path.join(CORPUS, src), "-vf", chain, "-f", "null", "-"]
    t = time.time(); p = subprocess.run(cmd, capture_output=True, text=True); dt = time.time() - t
    if p.returncode: return float("nan")
    return 60 / dt


def run(name, chain, clips):
    vals = [score(s, g, chain) for s, g in clips]
    vals = [v for v in vals if v]
    if not vals:
        print(f"{name:34s} FAILED", flush=True); return
    v = sum(x["vmaf"] for x in vals) / len(vals)
    p = sum(x["psnr"] for x in vals) / len(vals)
    s = sum(x["ssim"] for x in vals) / len(vals)
    VS = " ".join("%.1f" % x["vmaf"] for x in vals)
    print(f"{name:34s} VMAF {v:6.2f} PSNR {p:6.3f} SSIM {s:.4f}  [{VS}]", flush=True)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "sharpen":
        for up_name, up in (("s36", UP_S36), ("l3", UP_L3)):
            run(f"{up_name}", up, CLEAN)
            for s in (0.2, 0.4, 0.6, 0.8, 1.0):
                run(f"{up_name}+cas{s}", f"{up},format=yuv420p,cas=strength={s}", CLEAN)
            for a in (0.5, 1.0):
                run(f"{up_name}+unsharp5_{a}", f"{up},format=yuv420p,unsharp=5:5:{a}:3:3:0", CLEAN)
    elif which == "denoise":
        D = {
            "none": "",
            "hqdn3d_light": "hqdn3d=2:1.5:3:2.25",
            "hqdn3d_def": "hqdn3d",
            "hqdn3d_strong": "hqdn3d=6:4.5:9:6.75",
            "nlmeans_s1": "nlmeans=s=1",
            "nlmeans_s2": "nlmeans=s=2",
            "nlmeans_s3": "nlmeans=s=3",
            "nlmeans_s4": "nlmeans=s=4",
            "nlmeans_s3_r9": "nlmeans=s=3:r=9",
            "atadenoise": "atadenoise",
            "fftdnoiz_s4_t": "fftdnoiz=sigma=4:prev=1:next=1",
            "fftdnoiz_s8_t": "fftdnoiz=sigma=8:prev=1:next=1",
            "vague_t3": "vaguedenoiser=threshold=3",
            "hqdn3dT+nlm_s2": "hqdn3d=0:0:4:3,nlmeans=s=2",
            "hqdn3dT+nlm_s3": "hqdn3d=0:0:4:3,nlmeans=s=3",
            "bm3d_s4_g8": "bm3d=sigma=4:block=8:bstep=4:group=8:range=8",
            "bm3d_s8_g8": "bm3d=sigma=8:block=8:bstep=4:group=8:range=8",
        }
        sets = sys.argv[2] if len(sys.argv) > 2 else "grain"
        clips = GRAIN if sets == "grain" else CLEAN[:4]
        for name, f in D.items():
            chain = (f + "," if f else "") + UP_S36
            run(f"{sets}:{name}", chain, clips)
    elif which == "deblock":
        D = {
            "none": "",
            "deblock_weak": "deblock=filter=weak:block=8",
            "deblock_strong": "deblock=filter=strong:block=8",
            "pp7_auto": "pp7=qp=0:mode=medium",
            "spp4_auto": "spp=quality=4:qp=0:mode=soft",
            "fspp4": "fspp=quality=4:strength=0",
            "pp7_q4": "pp7=qp=4:mode=medium",
            "spp4_q4": "spp=quality=4:qp=4:mode=soft",
        }
        for name, f in D.items():
            chain = (f + "," if f else "") + UP_S36
            run(f"deblock:{name}", chain, CLEAN[:4])
    elif which == "speed":
        for name, f in [("hqdn3d", "hqdn3d"), ("nlmeans_s3", "nlmeans=s=3"), ("nlmeans_s3_r9", "nlmeans=s=3:r=9"),
                        ("bm3d_s4_g8", "bm3d=sigma=4:block=8:bstep=4:group=8:range=8"), ("fftdnoiz_t", "fftdnoiz=sigma=4:prev=1:next=1"),
                        ("atadenoise", "atadenoise"), ("vague", "vaguedenoiser"), ("deblock", "deblock=filter=weak:block=8"),
                        ("pp7", "pp7=qp=0:mode=medium"), ("spp4", "spp=quality=4:qp=0"), ("fspp4", "fspp=quality=4"),
                        ("upscale_s36", UP_S36), ("upscale_l3", UP_L3), ("cas", UP_S36 + ",format=yuv420p,cas=0.5"),
                        ("bwdif", "bwdif"), ("minterp_mci", "minterpolate=fps=60000/1001:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")]:
            print(f"speed {name:16s} {fps_of(f):7.1f} fps (SD input)", flush=True)
