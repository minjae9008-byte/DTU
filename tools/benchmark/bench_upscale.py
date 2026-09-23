#!/usr/bin/env python3
import sys
from bench import score

CLIPS = [("dvd/old_town_cross_clean.mpg", "gt/old_town_cross_1080p24.mkv"),
         ("dvd/park_joy_clean.mpg", "gt/park_joy_1080p24.mkv"),
         ("dvd/crowd_run_clean.mpg", "gt/crowd_run_1080p24.mkv"),
         ("dvd/ducks_take_off_clean.mpg", "gt/ducks_take_off_1080p24.mkv"),
         ("dvd/sintel_clean.mpg", "gt/sintel_1080p24.mkv")]

ZIN = "matrixin=170m:transferin=601:primariesin=170m:rangein=limited:chromalin=left"
ZOUT = "matrix=709:transfer=709:primaries=709:range=limited"
def zs(f, extra=""):
    return f"zscale=w=1920:h=1080:filter={f}{(':'+extra) if extra else ''}:{ZIN}:{ZOUT}"
SWS_CS = "in_color_matrix=bt601:out_color_matrix=bt709:in_range=tv:out_range=tv"
def sws(flags, extra=""):
    return f"scale=1920:1080:flags={flags}+accurate_rnd+full_chroma_int+full_chroma_inp:{SWS_CS}{(':'+extra) if extra else ''}"
PL_IN = "setparams=color_primaries=smpte170m:color_trc=smpte170m:colorspace=smpte170m:range=tv:chroma_location=left"
def pl(up, extra=""):
    return (f"{PL_IN},libplacebo=w=1920:h=1080:upscaler={up}:colorspace=bt709:color_primaries=bt709:"
            f"color_trc=bt709:range=tv:format=yuv420p{(':'+extra) if extra else ''}")

CPU = {
    "zs_bilinear": zs("bilinear"),
    "zs_bicubic_mitchell": zs("bicubic", "param_a=0.3333:param_b=0.3333"),
    "zs_bicubic_catrom": zs("bicubic", "param_a=0:param_b=0.5"),
    "zs_spline16": zs("spline16"),
    "zs_spline36": zs("spline36"),
    "zs_spline64": zs("spline64"),
    "zs_lanczos3": zs("lanczos", "param_a=3"),
    "zs_lanczos4": zs("lanczos", "param_a=4"),
    "sws_lanczos": sws("lanczos"),
    "sws_spline": sws("spline"),
}
GPU = {
    "pl_spline36": pl("spline36"),
    "pl_spline36_nosig": pl("spline36", "sigmoid=0"),
    "pl_lanczos": pl("lanczos"),
    "pl_ewa_lanczos": pl("ewa_lanczos"),
    "pl_ewa_lanczossharp": pl("ewa_lanczossharp"),
    "pl_ewa_lanczos4sharpest": pl("ewa_lanczos4sharpest"),
    "pl_ewa_ginseng": pl("ewa_ginseng"),
    "pl_ewa_robidouxsharp": pl("ewa_robidouxsharp"),
    "pl_ewa_jinc": pl("ewa_jinc"),
}

def run(cands, clips, vk=False):
    rows = []
    for name, chain in cands.items():
        vals = []
        for src, gt in clips:
            r = score(src, gt, chain, vk=vk)
            vals.append(r)
        v = sum(x["vmaf"] for x in vals) / len(vals)
        p = sum(x["psnr"] for x in vals) / len(vals)
        s = sum(x["ssim"] for x in vals) / len(vals)
        per = " ".join(f"{x['vmaf']:.2f}" for x in vals)
        print(f"{name:28s} VMAF {v:6.2f}  PSNR {p:6.3f}  SSIM {s:.4f}   [{per}]", flush=True)
        rows.append((name, v, p, s))
    return rows

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    if which == "cpu":
        run(CPU, CLIPS)
    else:
        run(GPU, [CLIPS[0], CLIPS[1], CLIPS[4]], vk=True)
