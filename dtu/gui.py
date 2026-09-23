"""Tkinter GUI (Korean).

Layout: job list + analysis info on the left, setting tabs on the right,
output folder / start / stop / progress / log at the bottom.  All FFmpeg work
runs in worker threads that report through a queue polled by the Tk loop.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from . import APP_NAME, __version__
from .codecs import CODECS, available_codecs
from .ffmpeg import FFmpeg, FFmpegError, FFmpegNotFound
from .settings import PRESETS, Settings, preset_settings

# ---------------------------------------------------------------------------
# option lists: (label, value)
# ---------------------------------------------------------------------------
UPSCALE = [("원본 해상도 유지", "off"), ("720p (1280x720)", "720p"), ("1080p (1920x1080) - 추천", "1080p"),
           ("1440p (2560x1440)", "1440p"), ("4K (3840x2160)", "2160p")]
SCALER = [("자동 (GPU: EWA Lanczos Sharp / CPU: Lanczos)", "auto"), ("Lanczos-3 (CPU, 선명)", "lanczos"),
          ("Spline36 (CPU, 자연스러움)", "spline36"), ("Spline64 (CPU)", "spline64"),
          ("Bicubic (CPU, 부드러움)", "bicubic"), ("EWA Lanczos Sharp (GPU, 최고 품질)", "ewa_lanczossharp"),
          ("EWA Lanczos (GPU)", "ewa_lanczos"), ("EWA Robidoux Sharp (GPU)", "ewa_robidouxsharp")]
GPU = [("자동 (하드웨어 GPU가 있을 때만)", "auto"), ("항상 사용", "on"), ("사용 안 함 (CPU만)", "off")]
DEINT = [("자동 감지 - 추천", "auto"), ("끄기", "off"), ("BWDIF (같은 프레임 수)", "bwdif"),
         ("BWDIF 2배 프레임 (부드러운 움직임)", "bwdif_double"), ("역텔레시네 IVTC (NTSC 필름 → 23.976p)", "ivtc"),
         ("필드 매칭 (PAL 필름)", "fieldmatch"), ("중복 프레임 제거 (29.97p → 23.976p)", "decimate")]
CROP = [("자동 검은 여백 제거", "auto"), ("사용 안 함", "off")]
DENOISE = [("끄기", "off"), ("약 - 추천", "light"), ("중", "medium"), ("강", "strong"),
           ("최강 (오래된 필름, 느림)", "very_strong")]
DEBLOCK = [("끄기 - 추천", "off"), ("약", "light"), ("중", "medium"), ("강", "strong")]
DEBAND = [("끄기", "off"), ("약 - 추천", "light"), ("강 (애니메이션/어두운 장면)", "strong")]
INTERP = [("끄기 (원본 프레임 수)", "off"), ("60fps (PAL 원본은 50fps)", "60"), ("2배 프레임", "2x")]
INTERP_Q = [("움직임 보상 MCI (고품질, 매우 느림)", "mci"), ("프레임 혼합 (빠름)", "blend")]
SPEED = [("가장 빠름", "fastest"), ("빠름", "fast"), ("균형", "balanced"), ("느림 (고효율)", "slow"),
         ("가장 느림 (최고 효율)", "slowest")]
TUNE = [("자동", "auto"), ("필름", "film"), ("애니메이션", "animation"), ("필름 그레인 유지", "grain")]
CONTAINER = [("MKV - 추천 (모든 자막/오디오 지원)", "mkv"), ("MP4 (PGS 자막은 별도 .sup 파일)", "mp4")]
AMODE = [("음질 개선 + 재인코딩 - 추천", "enhance"), ("단순 재인코딩", "encode"), ("원본 그대로 복사", "copy")]
ACODEC = [("Opus - 추천 (고효율)", "opus"), ("AAC (호환성)", "aac"), ("FLAC (무손실)", "flac"),
          ("AC3 (TV/리시버 호환)", "ac3"), ("E-AC3", "eac3")]
ACHAN = [("원본 채널 유지", "keep"), ("스테레오 다운믹스", "stereo"), ("대사 우선 스테레오 (TV 스피커용)", "stereo_dialog")]
LUFS = [("-16 LUFS (크게, 모바일)", -16.0), ("-18 LUFS", -18.0), ("-20 LUFS", -20.0),
        ("-23 LUFS (EBU 방송 표준) - 추천", -23.0), ("-24 LUFS (ATSC)", -24.0), ("-27 LUFS (극장 기준)", -27.0)]
DRC = [("끄기", "off"), ("완만하게", "light"), ("야간 모드 (큰 소리 억제)", "night")]
ADENOISE = [("끄기", "off"), ("약", "light"), ("강", "strong")]
TRACKS = [("모두", "all"), ("첫 번째만", "first"), ("한국어만", "kor"), ("한국어 + 영어", "kor,eng")]
SMODE = [("업스케일 PGS 자막 (켜고 끌 수 있음) - 추천", "soft"), ("영상에 입히기 (하드섭)", "burn"),
         ("원본 그대로 복사", "copy"), ("자막 제외", "none")]
SALGO = [("자동 - 추천 (자막마다 최적 선택)", "auto"), ("윤곽 복원 (벡터형, 계단 제거)", "contour"),
         ("xBR (픽셀아트형 가장자리 보간)", "xbr"), ("Lanczos (부드러움)", "lanczos"),
         ("원본 픽셀 (최근접)", "nearest")]


def config_path() -> str:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, APP_NAME, "settings.json")
    return os.path.join(os.path.expanduser("~"), ".config", "dtu", "settings.json")


def load_saved_settings() -> Settings:
    try:
        with open(config_path(), "r", encoding="utf-8") as f:
            return Settings.from_json(f.read())
    except (OSError, ValueError):
        return preset_settings("balanced")


def save_settings_file(s: Settings, path: Optional[str] = None) -> None:
    path = path or config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s.to_json())


def resize_rgb(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Area-ish downscale for display (separable linear resampling)."""
    from .subtitles.upscale import resample_matrix
    H, W = img.shape[:2]
    if (W, H) == (w, h):
        return img
    wy = resample_matrix(H, h, h / H, 0.0, "bilinear")
    wx = resample_matrix(W, w, w / W, 0.0, "bilinear")
    out = np.einsum("oy,yxc,px->opc", wy, img.astype(np.float32), wx.astype(np.float32), optimize=True)
    return np.clip(out, 0, 255).astype(np.uint8)


def photo_from_array(img: np.ndarray) -> tk.PhotoImage:
    h, w = img.shape[:2]
    data = f"P6 {w} {h} 255 ".encode() + np.ascontiguousarray(img, dtype=np.uint8).tobytes()
    return tk.PhotoImage(data=data, format="PPM")


# ---------------------------------------------------------------------------
# small widgets
# ---------------------------------------------------------------------------

class Choice:
    """Combobox bound to a (label, value) list."""

    def __init__(self, parent, options: List[Tuple[str, Any]], width: int = 42,
                 on_change: Optional[Callable[[], None]] = None):
        self.options = options
        self.var = tk.StringVar()
        self.cb = ttk.Combobox(parent, textvariable=self.var, values=[o[0] for o in options], state="readonly",
                               width=width)
        if on_change:
            self.cb.bind("<<ComboboxSelected>>", lambda e: on_change())

    def get(self):
        label = self.var.get()
        for l, v in self.options:
            if l == label:
                return v
        return self.options[0][1] if self.options else None

    def set(self, value):
        for l, v in self.options:
            if v == value or (isinstance(v, float) and isinstance(value, (int, float)) and abs(v - value) < 1e-6):
                self.var.set(l)
                return
        if self.options:
            self.var.set(self.options[0][0])

    def set_options(self, options):
        cur = self.get()
        self.options = options
        self.cb.configure(values=[o[0] for o in options])
        self.set(cur)


@dataclass
class JobItem:
    path: str
    iid: str
    status: str = "대기"
    info_text: str = ""
    scan: Any = None
    crop: Any = None
    duration: float = 0.0
    output: str = ""
    analyzing: bool = False
    done: bool = False


# ---------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.q: "queue.Queue[tuple]" = queue.Queue()
        self.settings = load_saved_settings()
        self.ff: Optional[FFmpeg] = None
        self.jobs: List[JobItem] = []
        self.running = False
        self.current = None  # running Job
        self.stop_all = False
        self._codec_keys: Optional[List[str]] = None
        root.title(f"DTU {__version__} - DVD 리마스터 (업스케일 · 디인터레이스 · 잡음 제거 · 음질 개선)")
        root.geometry("1280x860")
        root.minsize(1100, 760)
        self._style()
        self._build()
        self._load_into_widgets(self.settings)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._poll)
        self.root.after(300, self._init_ffmpeg)

    # -- look & feel ------------------------------------------------------
    def _style(self):
        st = ttk.Style()
        try:
            if "vista" in st.theme_names():
                st.theme_use("vista")
            elif "clam" in st.theme_names():
                st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("Title.TLabel", font=("TkDefaultFont", 13, "bold"))
        st.configure("Hint.TLabel", foreground="#666666")
        st.configure("Accent.TButton", font=("TkDefaultFont", 11, "bold"))

    # -- layout -------------------------------------------------------------
    def _build(self):
        root = self.root
        top = ttk.Frame(root, padding=(10, 8, 10, 0))
        top.pack(fill="x")
        ttk.Label(top, text="DVD 영상 · 이미지 자막 리마스터", style="Title.TLabel").pack(side="left")
        ttk.Label(top, text="  AI 없이 검증된 알고리즘만 사용 (BWDIF · IVTC · hqdn3d · Lanczos/EWA · CAS · "
                            "xBR 자막 · EBU R128)", style="Hint.TLabel").pack(side="left")

        main = ttk.PanedWindow(root, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=6)
        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=5)
        main.add(right, weight=6)

        # job list
        lf = ttk.LabelFrame(left, text="작업 목록", padding=6)
        lf.pack(fill="both", expand=True)
        btns = ttk.Frame(lf)
        btns.pack(fill="x")
        ttk.Button(btns, text="파일 추가…", command=self.add_files).pack(side="left")
        ttk.Button(btns, text="DVD 폴더 추가…", command=self.add_folder).pack(side="left", padx=4)
        ttk.Button(btns, text="제거", command=self.remove_selected).pack(side="left")
        ttk.Button(btns, text="미리보기", command=self.open_preview).pack(side="right")
        cols = ("status", "progress")
        self.tree = ttk.Treeview(lf, columns=cols, show="tree headings", height=8, selectmode="browse")
        self.tree.heading("#0", text="파일")
        self.tree.heading("status", text="상태")
        self.tree.heading("progress", text="진행")
        self.tree.column("#0", width=300)
        self.tree.column("status", width=150, anchor="w")
        self.tree.column("progress", width=70, anchor="e")
        self.tree.pack(fill="both", expand=True, pady=(6, 0))
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._show_info())

        inf = ttk.LabelFrame(left, text="분석 결과 / 처리 계획", padding=6)
        inf.pack(fill="both", expand=True, pady=(6, 0))
        self.info = tk.Text(inf, height=12, wrap="word", relief="flat", background="#f7f7f7")
        self.info.pack(fill="both", expand=True)
        self.info.configure(state="disabled")

        # settings
        pf = ttk.Frame(right)
        pf.pack(fill="x")
        ttk.Label(pf, text="프리셋:").pack(side="left")
        self.preset = Choice(pf, [(v["label"], k) for k, v in PRESETS.items()], width=36,
                             on_change=self.apply_preset)
        self.preset.cb.pack(side="left", padx=4)
        ttk.Button(pf, text="불러오기…", command=self.load_settings_dialog).pack(side="right")
        ttk.Button(pf, text="저장…", command=self.save_settings_dialog).pack(side="right", padx=4)

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True, pady=(6, 0))
        self.w: Dict[str, Any] = {}
        self._tab_video(nb)
        self._tab_encode(nb)
        self._tab_audio(nb)
        self._tab_subs(nb)
        self._tab_misc(nb)

        # bottom: output, start/stop, progress, log
        bottom = ttk.Frame(root, padding=(10, 0, 10, 8))
        bottom.pack(fill="both")
        of = ttk.Frame(bottom)
        of.pack(fill="x")
        ttk.Label(of, text="출력 폴더:").pack(side="left")
        self.outdir = tk.StringVar()
        ttk.Entry(of, textvariable=self.outdir, width=60).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(of, text="찾아보기…", command=self.pick_outdir).pack(side="left")
        ttk.Label(of, text=" (비우면 원본 파일과 같은 폴더)", style="Hint.TLabel").pack(side="left")
        self.stop_btn = ttk.Button(of, text="■ 중지", command=self.stop, state="disabled")
        self.stop_btn.pack(side="right")
        self.start_btn = ttk.Button(of, text="▶ 변환 시작", style="Accent.TButton", command=self.start)
        self.start_btn.pack(side="right", padx=6)
        self.sample_btn = ttk.Button(of, text="30초 시험 변환", command=self.start_sample)
        self.sample_btn.pack(side="right")

        pr = ttk.Frame(bottom)
        pr.pack(fill="x", pady=(6, 2))
        self.pbar = ttk.Progressbar(pr, maximum=1000)
        self.pbar.pack(side="left", fill="x", expand=True)
        self.status = tk.StringVar(value="파일을 추가하세요 (MKV, VOB, VIDEO_TS 폴더, ISO, MPG …)")
        ttk.Label(bottom, textvariable=self.status).pack(fill="x")
        lg = ttk.Frame(bottom)
        lg.pack(fill="both", expand=True)
        self.log = tk.Text(lg, height=9, wrap="none", font=("TkFixedFont", 9))
        sb = ttk.Scrollbar(lg, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _grid(self, frame, row, label, widget, hint: str = ""):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        widget.grid(row=row, column=1, sticky="we", pady=3)
        if hint:
            ttk.Label(frame, text=hint, style="Hint.TLabel", wraplength=520, justify="left").grid(
                row=row + 1, column=1, sticky="w", pady=(0, 4))
        frame.columnconfigure(1, weight=1)

    def _scale(self, parent, var, lo, hi, res, fmt="{:.2f}"):
        f = ttk.Frame(parent)
        lab = ttk.Label(f, width=6)

        def upd(*_):
            try:
                lab.configure(text=fmt.format(float(var.get())))
            except (tk.TclError, ValueError):
                pass
        s = tk.Scale(f, variable=var, from_=lo, to=hi, resolution=res, orient="horizontal", showvalue=False,
                     command=lambda *_: upd(), length=260)
        s.pack(side="left", fill="x", expand=True)
        lab.pack(side="left")
        var.trace_add("write", upd)
        upd()
        return f

    def _tab_video(self, nb):
        t = ttk.Frame(nb, padding=10)
        nb.add(t, text="영상")
        w = self.w
        r = 0
        w["upscale"] = Choice(t, UPSCALE)
        self._grid(t, r, "출력 해상도", w["upscale"].cb, "DVD(720x480/576)를 화면비에 맞춰 이 크기 안에 꽉 차게 키웁니다.")
        r += 2
        w["scaler"] = Choice(t, SCALER)
        self._grid(t, r, "업스케일 알고리즘", w["scaler"].cb,
                   "벤치마크 결과 Lanczos/EWA 계열 + CAS 선명화가 원본 HD에 가장 가까웠습니다.")
        r += 2
        w["gpu_scaler"] = Choice(t, GPU)
        self._grid(t, r, "GPU (Vulkan)", w["gpu_scaler"].cb)
        r += 1
        w["deinterlace"] = Choice(t, DEINT)
        self._grid(t, r, "디인터레이스", w["deinterlace"].cb,
                   "자동: 인터레이스/텔레시네/프로그레시브를 분석해 BWDIF 또는 역텔레시네를 적용합니다.")
        r += 2
        w["crop"] = Choice(t, CROP)
        self._grid(t, r, "검은 여백", w["crop"].cb)
        r += 1
        w["denoise"] = Choice(t, DENOISE)
        self._grid(t, r, "잡음(노이즈) 제거", w["denoise"].cb, "hqdn3d 시공간 필터 (최강: + 3D FFT 필터)")
        r += 2
        w["deblock"] = Choice(t, DEBLOCK)
        self._grid(t, r, "블록 노이즈 제거", w["deblock"].cb, "화면이 깍두기처럼 깨지는 디스크에만 사용하세요.")
        r += 2
        w["deband"] = Choice(t, DEBAND)
        self._grid(t, r, "디밴딩", w["deband"].cb, "하늘·어두운 장면의 계단 현상(밴딩)을 부드럽게 만듭니다.")
        r += 2
        self.v_sharpen = tk.DoubleVar(value=0.4)
        self._grid(t, r, "선명화 (CAS)", self._scale(t, self.v_sharpen, 0.0, 0.8, 0.05),
                   "AMD FidelityFX CAS. 0.3~0.5 권장 (0 = 끄기)")
        r += 2
        w["interpolate"] = Choice(t, INTERP)
        self._grid(t, r, "프레임 보간", w["interpolate"].cb)
        r += 1
        w["interpolate_quality"] = Choice(t, INTERP_Q)
        self._grid(t, r, "보간 방식", w["interpolate_quality"].cb,
                   "움직임 보상은 실제 중간 프레임과 가장 가까웠지만(측정 PSNR +3dB) 매우 느립니다.")
        r += 2
        self.v_colorconv = tk.BooleanVar(value=True)
        ttk.Checkbutton(t, text="색 공간 변환 BT.601 → BT.709 (HD 표준, 색이 틀어지지 않게)",
                        variable=self.v_colorconv).grid(row=r, column=1, sticky="w", pady=3)

    def _tab_encode(self, nb):
        t = ttk.Frame(nb, padding=10)
        nb.add(t, text="코덱")
        w = self.w
        r = 0
        w["codec"] = Choice(t, [(c.label, k) for k, c in CODECS.items()], on_change=self._codec_changed)
        self._grid(t, r, "비디오 코덱", w["codec"].cb)
        r += 1
        self.codec_note = tk.StringVar()
        ttk.Label(t, textvariable=self.codec_note, style="Hint.TLabel", wraplength=520).grid(
            row=r, column=1, sticky="w")
        r += 1
        qf = ttk.Frame(t)
        self.v_qdefault = tk.BooleanVar(value=True)
        self.v_quality = tk.IntVar(value=27)
        ttk.Checkbutton(qf, text="기본값", variable=self.v_qdefault, command=self._codec_changed).pack(side="left")
        self.q_scale = tk.Scale(qf, variable=self.v_quality, from_=10, to=50, orient="horizontal", length=260)
        self.q_scale.pack(side="left", fill="x", expand=True)
        self._grid(t, r, "품질 (CRF/QP)", qf, "숫자가 낮을수록 고화질·큰 파일. 기본값은 DVD 업스케일에 맞춘 값입니다.")
        r += 2
        w["speed"] = Choice(t, SPEED)
        self._grid(t, r, "인코딩 속도", w["speed"].cb, "느릴수록 같은 화질에서 파일이 작아집니다.")
        r += 2
        self.v_10bit = tk.BooleanVar(value=True)
        ttk.Checkbutton(t, text="10비트 인코딩 (밴딩 감소, 권장)", variable=self.v_10bit).grid(
            row=r, column=1, sticky="w", pady=3)
        r += 1
        self.v_grain = tk.IntVar(value=0)
        self._grid(t, r, "AV1 필름 그레인 합성", self._scale(t, self.v_grain, 0, 30, 1, "{:.0f}"),
                   "잡음 제거 후 자연스러운 필름 입자를 재생 시 합성 (용량 증가 거의 없음). 0 = 끄기")
        r += 2
        w["tune"] = Choice(t, TUNE)
        self._grid(t, r, "튜닝 (x264/x265)", w["tune"].cb)
        r += 1
        w["container"] = Choice(t, CONTAINER)
        self._grid(t, r, "컨테이너", w["container"].cb)
        r += 1
        self.v_extra = tk.StringVar()
        self._grid(t, r, "추가 인코더 옵션", ttk.Entry(t, textvariable=self.v_extra), "고급 사용자용 (예: -g 120)")

    def _tab_audio(self, nb):
        t = ttk.Frame(nb, padding=10)
        nb.add(t, text="오디오")
        w = self.w
        r = 0
        w["amode"] = Choice(t, AMODE)
        self._grid(t, r, "처리 방식", w["amode"].cb)
        r += 1
        w["acodec"] = Choice(t, ACODEC)
        self._grid(t, r, "오디오 코덱", w["acodec"].cb, "비트레이트는 채널 수에 맞춰 자동 (Opus 5.1: 320kbps)")
        r += 2
        w["channels"] = Choice(t, ACHAN)
        self._grid(t, r, "채널", w["channels"].cb)
        r += 1
        self.v_norm = tk.BooleanVar(value=True)
        nf = ttk.Frame(t)
        ttk.Checkbutton(nf, text="사용", variable=self.v_norm).pack(side="left")
        w["lufs"] = Choice(nf, LUFS, width=34)
        w["lufs"].cb.pack(side="left", padx=6)
        self._grid(t, r, "음량 정규화 (EBU R128)", nf,
                   "2패스: 전체 음량을 측정한 뒤 일정한 게인 + 피크 리미터 적용 (다이내믹 레인지 보존)")
        r += 2
        self.v_dialog = tk.DoubleVar(value=0.0)
        self._grid(t, r, "대사 강조 (dB)", self._scale(t, self.v_dialog, 0, 9, 0.5, "{:.1f}"),
                   "5.1: 센터 채널 증폭 / 스테레오: dialoguenhance로 음성 성분 추출 후 강조")
        r += 2
        w["drc"] = Choice(t, DRC)
        self._grid(t, r, "다이내믹 레인지 압축", w["drc"].cb)
        r += 1
        w["adenoise"] = Choice(t, ADENOISE)
        self._grid(t, r, "배경 잡음(히스) 제거", w["adenoise"].cb, "FFT 스펙트럼 차감 (afftdn, 잡음 추적)")
        r += 2
        cf = ttk.Frame(t)
        self.v_declick = tk.BooleanVar()
        self.v_declip = tk.BooleanVar()
        self.v_exciter = tk.BooleanVar()
        ttk.Checkbutton(cf, text="클릭/틱 잡음 제거", variable=self.v_declick).pack(side="left")
        ttk.Checkbutton(cf, text="클리핑 복원", variable=self.v_declip).pack(side="left", padx=8)
        ttk.Checkbutton(cf, text="고음역 보강 (익사이터)", variable=self.v_exciter).pack(side="left")
        self._grid(t, r, "추가 복원", cf)
        r += 1
        self.v_keeporig = tk.BooleanVar()
        ttk.Checkbutton(t, text="원본 오디오 트랙도 함께 보존", variable=self.v_keeporig).grid(
            row=r, column=1, sticky="w", pady=3)
        r += 1
        w["atracks"] = Choice(t, TRACKS)
        self._grid(t, r, "포함할 트랙", w["atracks"].cb)

    def _tab_subs(self, nb):
        t = ttk.Frame(nb, padding=10)
        nb.add(t, text="자막")
        w = self.w
        r = 0
        w["smode"] = Choice(t, SMODE)
        self._grid(t, r, "자막 처리", w["smode"].cb,
                   "DVD 이미지 자막(VobSub)을 출력 해상도에 맞춰 다시 그린 PGS(블루레이 자막)로 변환합니다.")
        r += 2
        w["salgo"] = Choice(t, SALGO)
        self._grid(t, r, "자막 업스케일 알고리즘", w["salgo"].cb,
                   "자동: 안티에일리어싱 픽셀이 있는 자막은 윤곽 복원, 3색 자막은 xBR (한글·영문 합성 자막 측정 결과)")
        r += 2
        w["stracks"] = Choice(t, TRACKS)
        self._grid(t, r, "포함할 트랙", w["stracks"].cb)
        r += 1
        self.v_burnidx = tk.IntVar(value=0)
        self._grid(t, r, "입힐 트랙 순번 (하드섭)", ttk.Spinbox(t, from_=0, to=31, textvariable=self.v_burnidx, width=6))
        r += 1
        self.v_forced = tk.BooleanVar(value=True)
        ttk.Checkbutton(t, text="강제 자막(외국어 대사 등)이 있으면 별도 '강제 자막' 트랙 추가",
                        variable=self.v_forced).grid(row=r, column=1, sticky="w", pady=3)

    def _tab_misc(self, nb):
        t = ttk.Frame(nb, padding=10)
        nb.add(t, text="기타")
        r = 0
        ff = ttk.Frame(t)
        self.v_ffdir = tk.StringVar()
        ttk.Entry(ff, textvariable=self.v_ffdir).pack(side="left", fill="x", expand=True)
        ttk.Button(ff, text="찾아보기…", command=self.pick_ffmpeg).pack(side="left", padx=4)
        ttk.Button(ff, text="확인", command=self._init_ffmpeg).pack(side="left")
        self._grid(t, r, "FFmpeg 폴더", ff, "비우면 PATH에서 찾습니다. 권장: FFmpeg 7.1 이상 'full' 빌드")
        r += 2
        self.v_suffix = tk.StringVar(value="_DTU")
        self._grid(t, r, "출력 파일 이름 접미사", ttk.Entry(t, textvariable=self.v_suffix, width=20))
        r += 1
        self.v_workers = tk.IntVar(value=0)
        self._grid(t, r, "자막 처리 프로세스 수", ttk.Spinbox(t, from_=0, to=32, textvariable=self.v_workers, width=6),
                   "0 = 자동")
        r += 2
        self.v_keeptemp = tk.BooleanVar()
        ttk.Checkbutton(t, text="임시 파일(업스케일된 .sup 자막) 보존", variable=self.v_keeptemp).grid(
            row=r, column=1, sticky="w")
        r += 1
        self.caps = tk.Text(t, height=14, wrap="word", relief="flat", background="#f7f7f7")
        self.caps.grid(row=r, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        t.rowconfigure(r, weight=1)

    # -- settings <-> widgets -----------------------------------------------
    def _load_into_widgets(self, s: Settings):
        w = self.w
        v, e, a, t = s.video, s.encode, s.audio, s.subtitles
        w["upscale"].set(v.upscale if v.upscale != "custom" else "1080p")
        w["scaler"].set(v.scaler)
        w["gpu_scaler"].set(v.gpu_scaler)
        w["deinterlace"].set(v.deinterlace)
        w["crop"].set(v.crop if v.crop != "manual" else "auto")
        w["denoise"].set(v.denoise)
        w["deblock"].set(v.deblock)
        w["deband"].set(v.deband)
        self.v_sharpen.set(v.sharpen)
        w["interpolate"].set(v.interpolate)
        w["interpolate_quality"].set(v.interpolate_quality)
        self.v_colorconv.set(v.color_convert)
        w["codec"].set(e.codec)
        self.v_qdefault.set(e.quality < 0)
        if e.quality >= 0:
            self.v_quality.set(e.quality)
        w["speed"].set(e.speed)
        self.v_10bit.set(e.bit_depth >= 10)
        self.v_grain.set(e.film_grain)
        w["tune"].set(e.tune)
        w["container"].set(e.container)
        self.v_extra.set(e.extra_args)
        w["amode"].set(a.mode)
        w["acodec"].set(a.codec)
        w["channels"].set(a.channels)
        self.v_norm.set(a.normalize)
        w["lufs"].set(a.target_lufs)
        self.v_dialog.set(a.dialog_boost)
        w["drc"].set(a.drc)
        w["adenoise"].set(a.denoise)
        self.v_declick.set(a.declick)
        self.v_declip.set(a.declip)
        self.v_exciter.set(a.exciter)
        self.v_keeporig.set(a.keep_original)
        w["atracks"].set(a.tracks)
        w["smode"].set(t.mode)
        w["salgo"].set(t.algorithm)
        w["stracks"].set(t.tracks)
        self.v_burnidx.set(t.burn_index)
        self.v_forced.set(t.forced_track)
        self.v_ffdir.set(s.ffmpeg_dir)
        self.v_suffix.set(s.name_suffix)
        self.v_workers.set(s.workers)
        self.v_keeptemp.set(s.keep_temp)
        self.outdir.set(s.output_dir)
        self.preset.set(s.preset if s.preset in PRESETS else "balanced")
        self._codec_changed()

    def collect_settings(self) -> Settings:
        w = self.w
        s = Settings()
        v, e, a, t = s.video, s.encode, s.audio, s.subtitles
        v.upscale = w["upscale"].get()
        v.scaler = w["scaler"].get()
        v.gpu_scaler = w["gpu_scaler"].get()
        v.deinterlace = w["deinterlace"].get()
        v.crop = w["crop"].get()
        v.denoise = w["denoise"].get()
        v.deblock = w["deblock"].get()
        v.deband = w["deband"].get()
        v.sharpen = round(float(self.v_sharpen.get()), 2)
        v.interpolate = w["interpolate"].get()
        v.interpolate_quality = w["interpolate_quality"].get()
        v.color_convert = bool(self.v_colorconv.get())
        e.codec = w["codec"].get()
        e.quality = -1 if self.v_qdefault.get() else int(self.v_quality.get())
        e.speed = w["speed"].get()
        e.bit_depth = 10 if self.v_10bit.get() else 8
        e.film_grain = int(self.v_grain.get())
        e.tune = w["tune"].get()
        e.container = w["container"].get()
        e.extra_args = self.v_extra.get()
        a.mode = w["amode"].get()
        a.codec = w["acodec"].get()
        a.channels = w["channels"].get()
        a.normalize = bool(self.v_norm.get())
        a.target_lufs = float(w["lufs"].get())
        a.dialog_boost = float(self.v_dialog.get())
        a.drc = w["drc"].get()
        a.denoise = w["adenoise"].get()
        a.declick = bool(self.v_declick.get())
        a.declip = bool(self.v_declip.get())
        a.exciter = bool(self.v_exciter.get())
        a.keep_original = bool(self.v_keeporig.get())
        a.tracks = w["atracks"].get()
        t.mode = w["smode"].get()
        t.algorithm = w["salgo"].get()
        t.tracks = w["stracks"].get()
        t.burn_index = int(self.v_burnidx.get() or 0)
        t.forced_track = bool(self.v_forced.get())
        s.ffmpeg_dir = self.v_ffdir.get().strip()
        s.name_suffix = self.v_suffix.get()
        s.workers = int(self.v_workers.get() or 0)
        s.keep_temp = bool(self.v_keeptemp.get())
        s.output_dir = self.outdir.get().strip()
        s.preset = self.preset.get()
        return s

    def _codec_changed(self):
        key = self.w["codec"].get()
        cd = CODECS.get(key)
        if not cd:
            return
        lo, hi = cd.quality_range
        self.q_scale.configure(from_=lo, to=hi)
        if self.v_qdefault.get():
            self.v_quality.set(cd.default_quality)
            self.q_scale.configure(state="disabled")
        else:
            self.q_scale.configure(state="normal")
        note = cd.note
        if key == "h264":
            self.v_10bit.set(False)
        if key == "vvc":
            self.v_10bit.set(True)
        self.codec_note.set(f"{cd.quality_label} 기본값 {cd.default_quality}. {note}")

    def apply_preset(self):
        name = self.preset.get()
        s = preset_settings(name, self.collect_settings())
        self._load_into_widgets(s)
        self.status.set(f"프리셋 적용: {PRESETS[name]['label']}")

    def save_settings_dialog(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("DTU 설정", "*.json")])
        if p:
            save_settings_file(self.collect_settings(), p)

    def load_settings_dialog(self):
        p = filedialog.askopenfilename(filetypes=[("DTU 설정", "*.json"), ("모든 파일", "*.*")])
        if p:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    self._load_into_widgets(Settings.from_json(f.read()))
            except (OSError, ValueError) as e:
                messagebox.showerror("오류", f"설정을 읽을 수 없습니다:\n{e}")

    # -- ffmpeg ---------------------------------------------------------------
    def _init_ffmpeg(self):
        folder = self.v_ffdir.get().strip() or None
        try:
            self.ff = FFmpeg(folder=folder)
        except FFmpegNotFound:
            self.ff = None
            self._set_caps("FFmpeg를 찾을 수 없습니다.\n\n"
                           "1) https://www.gyan.dev/ffmpeg/builds/ 에서 'ffmpeg-release-full' 을 받거나\n"
                           "   Windows 터미널에서:  winget install Gyan.FFmpeg\n"
                           "2) 압축을 푼 폴더의 bin 폴더를 위의 'FFmpeg 폴더'에 지정하세요.")
            messagebox.showwarning("FFmpeg 필요", "FFmpeg를 찾을 수 없습니다. '기타' 탭에서 FFmpeg 폴더를 지정하세요.")
            return
        self._set_caps(f"{self.ff.version_string}\n\n코덱/GPU 확인 중…")
        threading.Thread(target=self._caps_worker, daemon=True).start()

    def _caps_worker(self):
        try:
            ff = self.ff
            keys = available_codecs(ff)
            ok, sw, name = ff.vulkan_device()
            lines = [ff.version_string, "", "사용 가능한 비디오 코덱:"]
            lines += [f"  • {CODECS[k].label}" for k in keys]
            gpu = "없음"
            if ok:
                gpu = f"{name} ({'소프트웨어 - 자동 모드에서 사용 안 함' if sw else '하드웨어'})"
            lines += ["", f"GPU(Vulkan/libplacebo): {gpu}",
                      f"DVD-Video(VIDEO_TS/ISO) 직접 읽기: {'가능' if ff.has_demuxer('dvdvideo') else '불가 (VOB 결합 사용)'}"]
            missing = [f for f in ("zscale", "bwdif", "fieldmatch", "cas", "deband", "dialoguenhance", "afftdn",
                                   "dynaudnorm", "minterpolate") if not ff.has_filter(f)]
            if missing:
                lines += ["", "⚠ 이 FFmpeg에 없는 필터: " + ", ".join(missing) + " (full 빌드 권장)"]
            self.q.put(("caps", "\n".join(lines), keys))
        except Exception as e:  # noqa: BLE001
            self.q.put(("caps", f"FFmpeg 확인 실패: {e}", None))

    def _set_caps(self, text):
        self.caps.configure(state="normal")
        self.caps.delete("1.0", "end")
        self.caps.insert("end", text)
        self.caps.configure(state="disabled")

    def pick_ffmpeg(self):
        d = filedialog.askdirectory(title="ffmpeg(.exe)가 있는 폴더 선택")
        if d:
            self.v_ffdir.set(d)
            self._init_ffmpeg()

    def pick_outdir(self):
        d = filedialog.askdirectory(title="출력 폴더 선택")
        if d:
            self.outdir.set(d)

    # -- jobs -------------------------------------------------------------------
    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="DVD 영상 파일 선택",
            filetypes=[("영상 파일", "*.mkv *.vob *.mpg *.mpeg *.m2v *.mp4 *.m4v *.avi *.ts *.m2ts *.iso *.ifo"),
                       ("모든 파일", "*.*")])
        self._add_paths(paths)

    def add_folder(self):
        d = filedialog.askdirectory(title="VIDEO_TS 폴더(또는 그 상위 폴더) 선택")
        if d:
            self._add_paths([d])

    def _add_paths(self, paths):
        for p in paths:
            p = os.path.abspath(p)
            name = os.path.basename(p.rstrip("/\\")) or p
            iid = self.tree.insert("", "end", text=name, values=("분석 중…", ""))
            job = JobItem(path=p, iid=iid, analyzing=True)
            self.jobs.append(job)
            threading.Thread(target=self._analyze_worker, args=(job,), daemon=True).start()
        if paths:
            self.tree.selection_set(self.jobs[-1].iid)

    def _analyze_worker(self, item: JobItem):
        from .job import Job
        try:
            if self.ff is None:
                raise FFmpegNotFound("FFmpeg 없음")
            s = self.collect_settings_threadsafe()
            job = Job(item.path, s, ff=self.ff)
            plan = job.analyze()
            item.scan, item.crop = plan.scan, plan.crop
            item.duration = plan.info.duration
            text = plan.info.summary() + "\n\n" + plan.describe()
            self.q.put(("analyzed", item, text, None))
        except Exception as e:  # noqa: BLE001
            self.q.put(("analyzed", item, "", f"{e}"))

    def collect_settings_threadsafe(self) -> Settings:
        """Read widgets on the Tk thread (called from workers)."""
        box: Dict[str, Settings] = {}
        ev = threading.Event()

        def get():
            box["s"] = self.collect_settings()
            ev.set()
        self.root.after(0, get)
        ev.wait(10)
        return box.get("s") or Settings()

    def remove_selected(self):
        sel = self.tree.selection()
        for iid in sel:
            job = self._job_by_iid(iid)
            if job and self.running and self.current is not None and getattr(self.current, "_item", None) is job:
                messagebox.showinfo("알림", "변환 중인 작업은 제거할 수 없습니다.")
                continue
            self.tree.delete(iid)
            self.jobs = [j for j in self.jobs if j.iid != iid]

    def _job_by_iid(self, iid) -> Optional[JobItem]:
        for j in self.jobs:
            if j.iid == iid:
                return j
        return None

    def _selected_job(self) -> Optional[JobItem]:
        sel = self.tree.selection()
        return self._job_by_iid(sel[0]) if sel else None

    def _show_info(self):
        job = self._selected_job()
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        if job:
            self.info.insert("end", job.info_text or ("분석 중…" if job.analyzing else job.status))
        self.info.configure(state="disabled")

    # -- run --------------------------------------------------------------------
    def start(self):
        if self.running:
            return
        if self.ff is None:
            self._init_ffmpeg()
            if self.ff is None:
                return
        todo = [j for j in self.jobs if not j.done and not j.analyzing]
        if not todo:
            messagebox.showinfo("알림", "변환할 파일이 없습니다. 파일을 추가하고 분석이 끝나기를 기다리세요.")
            return
        s = self.collect_settings()
        save_settings_file(s)
        self.running = True
        self.stop_all = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        threading.Thread(target=self._run_worker, args=(todo, s), daemon=True).start()

    def start_sample(self):
        """Convert 30 s from the middle of the selected file to check quality and size."""
        if self.running:
            return
        item = self._selected_job()
        if item is None or item.analyzing:
            messagebox.showinfo("알림", "분석이 끝난 파일을 목록에서 선택하세요.")
            return
        if self.ff is None:
            return
        s = self.collect_settings()
        self.running = True
        self.stop_all = False
        self.start_btn.configure(state="disabled")
        self.sample_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        threading.Thread(target=self._run_worker, args=([item], s, (-1.0, 30.0)), daemon=True).start()

    def stop(self):
        self.stop_all = True
        if self.current is not None:
            self.current.cancel()
        self.status.set("중지하는 중…")

    def _run_worker(self, todo: List[JobItem], s: Settings, sample=None):
        from .job import Cancelled, Job
        for item in todo:
            if self.stop_all:
                break
            job = Job(item.path, s.copy(), ff=self.ff, sample=sample)
            job._item = item
            self.current = job
            self.q.put(("status", item, "변환 중", ""))

            def progress(stage, frac, detail, _item=item):
                self.q.put(("progress", _item, stage, frac, detail))

            def log(msg):
                self.q.put(("log", msg))
            try:
                job.analyze(progress, log, scan=item.scan, crop=item.crop)
                out = job.run(progress, log)
                if sample is None:
                    item.done = True
                item.output = out
                self.q.put(("status", item, "시험 변환 완료" if sample else "완료", "100%"))
                if sample:
                    size = os.path.getsize(out) / 1e6
                    est = size / sample[1] * (item.duration or sample[1]) / 1000
                    self.q.put(("log", f"시험 변환 결과: {out} ({size:.1f} MB / 30초 → 전체 예상 약 {est:.1f} GB)"))
            except Cancelled:
                self.q.put(("status", item, "중지됨", ""))
                break
            except Exception as e:  # noqa: BLE001
                self.q.put(("log", "오류: " + str(e)))
                self.q.put(("log", traceback.format_exc()))
                self.q.put(("status", item, "실패", ""))
        self.current = None
        self.q.put(("finished",))

    # -- queue polling ------------------------------------------------------
    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self._handle(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _handle(self, msg):
        kind = msg[0]
        if kind == "analyzed":
            _, item, text, err = msg
            item.analyzing = False
            if err:
                item.status = "분석 실패"
                item.info_text = "분석 실패: " + err
            else:
                item.status = "준비됨"
                item.info_text = text
            if self.tree.exists(item.iid):
                self.tree.item(item.iid, values=(item.status, ""))
            if self._selected_job() is item:
                self._show_info()
        elif kind == "status":
            _, item, status, prog = msg
            item.status = status
            if self.tree.exists(item.iid):
                self.tree.item(item.iid, values=(status, prog))
        elif kind == "progress":
            _, item, stage, frac, detail = msg
            names = {"analyze": "분석", "subtitles": "자막 업스케일", "loudness": "음량 측정", "encode": "인코딩"}
            weight = {"analyze": (0, 0.02), "subtitles": (0.02, 0.08), "loudness": (0.08, 0.1),
                      "encode": (0.1, 1.0)}.get(stage, (0, 1))
            total = weight[0] + (weight[1] - weight[0]) * max(0.0, min(1.0, frac))
            self.pbar["value"] = total * 1000
            self.status.set(f"{os.path.basename(item.path)} - {names.get(stage, stage)}: {detail}")
            if self.tree.exists(item.iid):
                self.tree.item(item.iid, values=(names.get(stage, stage), f"{total * 100:.1f}%"))
        elif kind == "log":
            self.log.insert("end", msg[1] + "\n")
            self.log.see("end")
        elif kind == "caps":
            _, text, keys = msg
            self._set_caps(text)
            if keys:
                self._codec_keys = keys
                self.w["codec"].set_options([(CODECS[k].label, k) for k in keys])
                self._codec_changed()
        elif kind == "finished":
            self.running = False
            self.start_btn.configure(state="normal")
            self.sample_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            done = [j for j in self.jobs if j.done]
            self.status.set(f"작업 종료 - 완료 {len(done)}개")
            self.pbar["value"] = 1000 if done else 0
        elif kind == "preview":
            _, win, pv, err = msg
            win.show(pv, err)

    # -- preview ----------------------------------------------------------------
    def open_preview(self):
        job = self._selected_job()
        if job is None:
            messagebox.showinfo("알림", "목록에서 파일을 선택하세요.")
            return
        if job.analyzing:
            messagebox.showinfo("알림", "분석이 끝난 뒤 미리보기를 사용할 수 있습니다.")
            return
        PreviewWindow(self, job)

    def on_close(self):
        if self.running and not messagebox.askyesno("종료", "변환 중입니다. 중지하고 종료할까요?"):
            return
        self.stop()
        try:
            save_settings_file(self.collect_settings())
        except OSError:
            pass
        self.root.destroy()


class PreviewWindow:
    """Before/after comparison with a time slider."""

    def __init__(self, app: App, item: JobItem):
        self.app = app
        self.item = item
        self.pv = None
        self.photo = None
        top = self.top = tk.Toplevel(app.root)
        top.title(f"미리보기 - {os.path.basename(item.path)}")
        top.geometry("1200x800")
        bar = ttk.Frame(top, padding=6)
        bar.pack(fill="x")
        dur = max(1.0, item.duration or 60.0)
        self.t = tk.DoubleVar(value=round(dur / 3, 1))
        ttk.Label(bar, text="시간(초):").pack(side="left")
        tk.Scale(bar, variable=self.t, from_=0, to=max(1.0, dur - 1), resolution=0.5, orient="horizontal",
                 length=420).pack(side="left")
        ttk.Button(bar, text="이 장면 미리보기", command=self.render).pack(side="left", padx=6)
        self.mode = tk.StringVar(value="split")
        for label, val in (("나란히 (좌: 원본, 우: 처리 후)", "split"), ("원본", "before"), ("처리 후", "after")):
            ttk.Radiobutton(bar, text=label, value=val, variable=self.mode, command=self.redraw).pack(side="left")
        self.zoom = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="100% 확대 (중앙)", variable=self.zoom, command=self.redraw).pack(side="left", padx=8)
        self.msg = tk.StringVar(value="'이 장면 미리보기'를 누르세요. 현재 설정이 적용됩니다.")
        ttk.Label(top, textvariable=self.msg, padding=(6, 0)).pack(fill="x")
        self.canvas = tk.Canvas(top, background="#202020", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.render()

    def render(self):
        self.msg.set("미리보기 만드는 중… (필터 체인 전체를 적용합니다)")
        s = self.app.collect_settings()
        t = float(self.t.get())
        threading.Thread(target=self._worker, args=(s, t), daemon=True).start()

    def _worker(self, s, t):
        from .job import Job
        from .preview import make_preview
        try:
            job = Job(self.item.path, s, ff=self.app.ff)
            job.analyze(scan=self.item.scan, crop=self.item.crop)
            pv = make_preview(job, t)
            self.app.q.put(("preview", self, pv, None))
        except Exception as e:  # noqa: BLE001
            self.app.q.put(("preview", self, None, str(e)))

    def show(self, pv, err):
        if not self.top.winfo_exists():
            return
        if err:
            self.msg.set("미리보기 실패: " + err[-300:])
            return
        self.pv = pv
        extra = f" · {pv.subtitle_text}" if pv.subtitle_text else ""
        self.msg.set(f"{pv.time:.1f}초, {pv.size[0]}x{pv.size[1]} · " + " · ".join(pv.notes) + extra)
        self.redraw()

    def redraw(self):
        pv = self.pv
        if pv is None:
            return
        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 200)
        mode = self.mode.get()
        if self.zoom.get():
            H, W = pv.after.shape[:2]
            if mode == "split":
                half = cw // 2 - 2
                x0 = max(0, W // 2 - half // 2)
                y0 = max(0, H // 2 - ch // 2)
                a = pv.before[y0:y0 + ch, x0:x0 + half]
                b = pv.after[y0:y0 + ch, x0:x0 + half]
                img = np.concatenate([a, np.full((a.shape[0], 4, 3), 255, np.uint8), b], axis=1)
            else:
                src = pv.before if mode == "before" else pv.after
                x0 = max(0, W // 2 - cw // 2)
                y0 = max(0, H // 2 - ch // 2)
                img = src[y0:y0 + ch, x0:x0 + cw]
        else:
            if mode == "split":
                img = np.concatenate([pv.before, np.full((pv.before.shape[0], 8, 3), 255, np.uint8), pv.after],
                                     axis=1)
            else:
                img = pv.before if mode == "before" else pv.after
            h, w = img.shape[:2]
            sc = min(cw / w, ch / h, 1.0)
            if sc < 1.0:
                img = resize_rgb(img, max(1, int(w * sc)), max(1, int(h * sc)))
        self.photo = photo_from_array(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor="center")


def _enable_dpi_awareness():
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    _enable_dpi_awareness()
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD  # optional drag & drop
        root = TkinterDnD.Tk()
    except Exception:  # noqa: BLE001
        root = tk.Tk()
        DND_FILES = None
    app = App(root)
    if DND_FILES:
        def on_drop(event):
            app._add_paths(root.tk.splitlist(event.data))
        root.drop_target_register(DND_FILES)
        root.dnd_bind("<<Drop>>", on_drop)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
