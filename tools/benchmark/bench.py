#!/usr/bin/env python3
"""Benchmark harness: run a filter chain on a DVD clip and score vs ground truth."""
import json, subprocess, sys, time, os, tempfile

CORPUS = os.path.dirname(os.path.abspath(__file__))

def score(src, gt, chain, gt_chain="", vk=False, extra_in="", n_threads=4, cache={}):
    """Return dict(vmaf, psnr_y, ssim, secs)."""
    key = (src, gt, chain, gt_chain)
    if key in cache:
        return cache[key]
    log = tempfile.mktemp(suffix=".json")
    hw = ["-init_hw_device", "vulkan=vk", "-filter_hw_device", "vk"] if vk else []
    d = f"[0:v]setpts=PTS-STARTPTS,{chain},format=yuv420p[d]"
    r = f"[1:v]setpts=PTS-STARTPTS{(','+gt_chain) if gt_chain else ''},format=yuv420p[r]"
    fc = f"{d};{r};[d][r]libvmaf=n_threads={n_threads}:feature=name=psnr|name=float_ssim:log_fmt=json:log_path={log}"
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", *hw, "-i", os.path.join(CORPUS, src),
           "-i", os.path.join(CORPUS, gt), "-filter_complex", fc, "-f", "null", "-"]
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t
    if p.returncode != 0:
        print("ERR", chain, p.stderr[-800:], file=sys.stderr)
        return None
    d = json.load(open(log))
    os.unlink(log)
    pm = d["pooled_metrics"]
    res = dict(vmaf=pm["vmaf"]["mean"], psnr=pm["psnr_y"]["mean"], ssim=pm["float_ssim"]["mean"], secs=dt,
               frames=len(d["frames"]))
    cache[key] = res
    return res

def speed(src, chain, vk=False, frames=None):
    """Measure processing fps of chain alone (decode + filter -> null)."""
    hw = ["-init_hw_device", "vulkan=vk", "-filter_hw_device", "vk"] if vk else []
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", *hw, "-i", os.path.join(CORPUS, src), "-vf", chain,
           "-f", "null", "-"]
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t
    return dt

if __name__ == "__main__":
    print(score(sys.argv[1], sys.argv[2], sys.argv[3]))
