"""Frame interpolation benchmark: drop every other frame of a real 50p clip,
interpolate back to 50p and score only the synthesised frames against the
real ones (run from the corpus folder after make_corpus.sh)."""
import numpy as np, subprocess, sys, time
def raw(args, W=720, H=480):
    p = subprocess.run(['ffmpeg', '-v', 'error', *args, '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], capture_output=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W).astype(np.float64)
def psnr(a, b):
    mse = ((a - b) ** 2).mean(); return 10 * np.log10(255 ** 2 / max(mse, 1e-9))
def ssim(a, b):
    # global-window SSIM approximation per 8x8 blocks
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    h, w = a.shape; a = a[:h//8*8, :w//8*8].reshape(h//8, 8, w//8, 8); b = b[:h//8*8, :w//8*8].reshape(h//8, 8, w//8, 8)
    ma, mb = a.mean((1, 3)), b.mean((1, 3)); va, vb = a.var((1, 3)), b.var((1, 3))
    cov = ((a - ma[:, None, :, None]) * (b - mb[:, None, :, None])).mean((1, 3))
    return (((2*ma*mb + C1) * (2*cov + C2)) / ((ma**2 + mb**2 + C1) * (va + vb + C2))).mean()
cands = {
    "repeat": "fps=50",
    "framerate(blend)": "framerate=fps=50",
    "minterp mci obmc": "minterpolate=fps=50:mi_mode=mci:mc_mode=obmc:me_mode=bidir:vsbmc=0",
    "minterp mci aobmc+vsbmc": "minterpolate=fps=50:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:me=epzs:vsbmc=1:scd=fdiff",
    "minterp mci aobmc umh": "minterpolate=fps=50:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:me=umh:vsbmc=1:scd=fdiff",
}
for gt in ['gt/crowd_run_480p50.mkv', 'gt/old_town_cross_480p50.mkv']:
    G = raw(['-i', gt])
    print('==', gt, G.shape[0], 'frames')
    for name, ch in cands.items():
        t = time.time()
        O = raw(['-i', gt, '-vf', "select='not(mod(n\\,2))',setpts=N/25/TB," + ch])
        dt = time.time() - t
        best = None
        for off in (-1, 0, 1):
            idx = [i for i in range(1, min(len(O), len(G)) - 2, 2) if 0 <= i + off < len(O)]
            ps = np.mean([psnr(O[i + off], G[i]) for i in idx])
            if best is None or ps > best[0]:
                best = (ps, off, np.mean([ssim(O[i + off], G[i]) for i in idx]))
        print(f'   {name:26s} PSNR {best[0]:6.2f}  SSIM {best[2]:.4f} (offset {best[1]}, {len(O)} frames, {dt:.1f}s)')
