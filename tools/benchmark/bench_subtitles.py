"""Synthetic DVD subtitle evaluation: render vector text at DVD res (4 colours)
and at true 1080p; measure how close each upscaler gets to the 1080p render."""
import sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..'))
from dtu.subtitles import upscale as U
from dtu.subtitles.imageio import write_png

OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
SS = 4                     # master supersampling vs 1080p
W, H = 1920, 1080
DW, DH = 720, 480          # DVD grid (16:9 anamorphic)
SX, SY = W / DW, H / DH

FONTS = {
    'ko': '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
    'en': '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
}
TEXT = {
    'ko': ['안녕하세요, 반갑습니다. 오늘 날씨가', '정말 좋네요! 괜찮아? 뭐라고 했어?'],
    'en': ['The quick brown fox jumps over', 'the lazy dog. "Wow!" 1234 @#&'],
}


def area_matrix(n_in, n_out, scale):
    """Box-filter weights: output pixel o averages input [o*scale, (o+1)*scale)."""
    m = np.zeros((n_out, n_in))
    for o in range(n_out):
        a, b = o * scale, (o + 1) * scale
        i0, i1 = int(np.floor(a)), int(np.ceil(b))
        for i in range(i0, min(i1, n_in)):
            ov = min(b, i + 1) - max(a, i)
            if ov > 0:
                m[o, i] = ov
        m[o] /= scale
    return m


def render_shapes(lang, size_px, stroke_px):
    """Return master-res (fill, outer) float masks for a W*SS x H*SS frame region."""
    font = ImageFont.truetype(FONTS[lang], int(size_px * SS))
    lines = TEXT[lang]
    # region: bottom band of the frame
    top = int(H * 0.72) * SS
    band_h = int(H * 0.26) * SS
    fill = Image.new('L', (W * SS, band_h), 0)
    outer = Image.new('L', (W * SS, band_h), 0)
    df, do = ImageDraw.Draw(fill), ImageDraw.Draw(outer)
    y = int(10 * SS)
    for line in lines:
        tw = df.textlength(line, font=font)
        x = (W * SS - tw) / 2
        do.text((x, y), line, font=font, fill=255, stroke_width=int(stroke_px * SS), stroke_fill=255)
        df.text((x, y), line, font=font, fill=255)
        y += int(size_px * SS * 1.25)
    return np.asarray(fill, np.float64) / 255, np.asarray(outer, np.float64) / 255, top


def make_case(lang, aa, size_px=52, stroke_px=3.0):
    fill, outer, top = render_shapes(lang, size_px, stroke_px)
    bh = fill.shape[0]
    # ---- ground truth at 1080p (4x4 box downsample)
    gy, gx = area_matrix(bh, bh // SS, SS), area_matrix(W * SS, W, SS)
    gf = gy @ fill @ gx.T
    go = np.maximum(gy @ outer @ gx.T, gf)
    white = np.array([235, 235, 235, 255.]); black = np.array([16, 16, 16, 255.])
    gt = gf[..., None] * white + (go - gf)[..., None] * black  # premultiplied
    gt_y0 = top // SS
    # ---- DVD rasterisation: coverage over each DVD pixel footprint
    dy0 = int(np.floor(gt_y0 / SY))
    y_off = dy0 * SY * SS - top  # master offset of the DVD row grid
    dh = int(np.ceil(bh / (SY * SS))) + 1
    # build area matrices on the master grid with fractional offset
    my = np.zeros((dh, bh))
    for o in range(dh):
        a, b = o * SY * SS + y_off, (o + 1) * SY * SS + y_off
        for i in range(max(int(np.floor(a)), 0), min(int(np.ceil(b)), bh)):
            ov = min(b, i + 1) - max(a, i)
            if ov > 0:
                my[o, i] = ov
    my /= SY * SS
    mx = area_matrix(W * SS, DW, SX * SS)
    cf = my @ fill @ mx.T
    co = np.maximum(my @ outer @ mx.T, cf)
    cls = np.zeros(cf.shape, np.uint8)
    if aa:
        cls[co >= 0.5] = 2          # outline
        cls[(cf >= 0.25)] = 3        # AA (grey)
        cls[cf >= 0.75] = 1          # fill
        cls[(co < 0.5) & (cf < 0.25)] = 0
    else:
        cls[co >= 0.5] = 2
        cls[cf >= 0.5] = 1
    pal = np.array([[0, 0, 0, 0], [235, 235, 235, 255], [16, 16, 16, 255], [126, 126, 126, 255]], np.uint8)
    return cls, pal, dy0, gt, gt_y0


def evaluate(cls, pal, dy0, gt, gt_y0, algo, variant=None):
    rgba = pal[cls]
    t = time.time()
    if variant is None:
        pm, x0, y0 = U.upscale_picture(rgba, cls, pal, SX, SY, 0.0, dy0 * SY, algo)
    else:
        kernel, blur = variant
        h, w = cls.shape
        fx, fy = 0.0, dy0 * SY
        x0, y0 = 0, int(np.floor(fy))
        ow, oh = int(np.ceil(w * SX)), int(np.ceil(fy + h * SY)) - y0
        wy = U.resample_matrix(h, oh, SY, (y0 - fy) / SY, kernel)
        wx = U.resample_matrix(w, ow, SX, 0.0, kernel)
        if blur:
            def g(n):
                i = np.arange(n); d = i[:, None] - i[None, :]
                k = np.exp(-0.5 * (d / blur) ** 2); return k / k.sum(1, keepdims=True)
            wy = wy @ g(h); wx = wx @ g(w)
        pm = U._contour(cls, pal, wy, wx)
    dt = time.time() - t
    # align to GT frame region
    full = np.zeros((H, W, 4))
    hh, ww = pm.shape[:2]
    full[y0:y0 + hh, x0:x0 + ww] = pm[:H - y0, :W - x0]
    ref = np.zeros((H, W, 4))
    ref[gt_y0:gt_y0 + gt.shape[0]] = gt
    region = (slice(gt_y0, gt_y0 + gt.shape[0]), slice(0, W))
    a, b = full[region], ref[region]
    mask = (a[..., 3] > 0) | (b[..., 3] > 0)
    err = np.abs(a - b)[mask]
    mae = err.mean()
    psnr = 10 * np.log10(255 ** 2 / ((a - b)[mask] ** 2).mean())
    return mae, psnr, dt, full[region]


def composite(pm, bgc=(70, 100, 130)):
    bg = np.zeros(pm.shape[:2] + (3,)); bg[:] = bgc
    a = np.clip(pm[..., 3:4], 0, 255) / 255
    return np.clip(pm[..., :3] + bg * (1 - a), 0, 255)


if __name__ == '__main__':
    algos = [('nearest', None), ('lanczos', None), ('xbr', None),
             ('contour', ('catrom', 0)), ('contour', ('bspline', 0)), ('contour', ('bspline', 0.4)),
             ('contour', ('catrom', 0.6)), ('contour', ('mitchell', 0.3))]
    for lang in ('ko', 'en'):
        for aa in (False, True):
            cls, pal, dy0, gt, gt_y0 = make_case(lang, aa)
            print(f'== {lang} aa={aa} dvd bitmap {cls.shape} classes {np.bincount(cls.ravel(), minlength=4)}')
            tiles = [composite(np.pad(gt, ((0, 0), (0, 0), (0, 0))))]
            for algo, var in algos:
                mae, psnr, dt, img = evaluate(cls, pal, dy0, gt, gt_y0, algo, var)
                name = algo + ('' if var is None else f'-{var[0]}-b{var[1]}')
                print(f'   {name:28s} MAE {mae:6.2f}  PSNR {psnr:6.2f} dB  {dt*1000:6.0f} ms')
                tiles.append(composite(img))
            # zoomed crop sheet: region around first chars
            crops = [t[20:130, 330:700] for t in tiles]
            z = [np.repeat(np.repeat(c, 2, 0), 2, 1) for c in crops]
            sheet = np.concatenate([np.concatenate([z[i], np.full((z[i].shape[0], 4, 3), 255.)], 1) for i in range(len(z))], 0) if False else None
            rows = []
            for i in range(0, len(z), 3):
                row = z[i:i + 3]
                while len(row) < 3:
                    row.append(np.full_like(z[0], 255))
                rows.append(np.concatenate([np.pad(r, ((3, 3), (3, 3), (0, 0)), constant_values=255) for r in row], 1))
            write_png(f'{OUT}/synth_{lang}_{int(aa)}.png', np.concatenate(rows, 0).astype(np.uint8))
