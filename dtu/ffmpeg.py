"""Locating FFmpeg, querying its capabilities and running it."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set

IS_WINDOWS = os.name == "nt"
# Hide console windows of child processes when running the GUI on Windows.
_POPEN_FLAGS = 0x08000000 if IS_WINDOWS else 0  # CREATE_NO_WINDOW


class FFmpegNotFound(RuntimeError):
    pass


class FFmpegError(RuntimeError):
    def __init__(self, msg: str, stderr: str = ""):
        super().__init__(msg)
        self.stderr = stderr


def _candidates(name: str) -> List[str]:
    exe = name + (".exe" if IS_WINDOWS else "")
    out = []
    env = os.environ.get("DTU_" + name.upper())
    if env:
        out.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    base = getattr(sys, "_MEIPASS", None)  # PyInstaller bundle
    for d in filter(None, [base, os.path.dirname(here), here, os.path.dirname(sys.executable)]):
        out.append(os.path.join(d, "ffmpeg", "bin", exe))
        out.append(os.path.join(d, "bin", exe))
        out.append(os.path.join(d, exe))
    found = shutil.which(name)
    if found:
        out.append(found)
    return out


def find_tool(name: str, explicit: Optional[str] = None) -> str:
    if explicit:
        if os.path.isdir(explicit):
            explicit = os.path.join(explicit, name + (".exe" if IS_WINDOWS else ""))
        if os.path.isfile(explicit):
            return explicit
        raise FFmpegNotFound(f"{name} not found at {explicit}")
    for c in _candidates(name):
        if c and os.path.isfile(c):
            return c
    raise FFmpegNotFound(
        f"'{name}' was not found. Install FFmpeg (6.1 or newer, 'full' build recommended) "
        f"and add it to PATH, or set the FFmpeg folder in the settings.")


def run(args: Sequence[str], timeout: Optional[float] = None, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, timeout=timeout,
                          creationflags=_POPEN_FLAGS, **kw)


def popen(args: Sequence[str], **kw) -> subprocess.Popen:
    return subprocess.Popen(list(args), creationflags=_POPEN_FLAGS, **kw)


class FFmpeg:
    """Paths to ffmpeg/ffprobe plus cached capability queries."""

    def __init__(self, ffmpeg: Optional[str] = None, ffprobe: Optional[str] = None,
                 folder: Optional[str] = None):
        if folder and not ffmpeg:
            ffmpeg = folder
        if folder and not ffprobe:
            ffprobe = folder
        self.ffmpeg = find_tool("ffmpeg", ffmpeg)
        try:
            self.ffprobe = find_tool("ffprobe", ffprobe)
        except FFmpegNotFound:
            # ffprobe usually sits next to ffmpeg
            cand = os.path.join(os.path.dirname(self.ffmpeg), "ffprobe" + (".exe" if IS_WINDOWS else ""))
            if os.path.isfile(cand):
                self.ffprobe = cand
            else:
                raise
        self._lock = threading.Lock()
        self._cache: Dict[str, object] = {}

    # -- capability queries -------------------------------------------------
    def _list(self, what: str) -> str:
        with self._lock:
            if what not in self._cache:
                p = run([self.ffmpeg, "-hide_banner", "-nostdin", f"-{what}"], timeout=60)
                self._cache[what] = p.stdout.decode("utf-8", "replace")
            return self._cache[what]  # type: ignore

    @property
    def version_string(self) -> str:
        with self._lock:
            if "version" not in self._cache:
                p = run([self.ffmpeg, "-hide_banner", "-version"], timeout=30)
                self._cache["version"] = p.stdout.decode("utf-8", "replace").splitlines()[0] if p.stdout else ""
            return self._cache["version"]  # type: ignore

    @property
    def version(self) -> tuple:
        """(major, minor) or (99, 0) for git master builds."""
        m = re.search(r"version n?(\d+)\.(\d+)", self.version_string)
        if m:
            return int(m.group(1)), int(m.group(2))
        return (99, 0)

    def _names(self, what: str, pattern: str) -> Set[str]:
        key = "set_" + what
        with self._lock:
            cached = self._cache.get(key)
        if cached is None:
            text = self._list(what)
            cached = set(re.findall(pattern, text, flags=re.M))
            with self._lock:
                self._cache[key] = cached
        return cached  # type: ignore

    def encoders(self) -> Set[str]:
        return self._names("encoders", r"^\s[VAS][A-Z.]{5}\s+(\S+)")

    def filters(self) -> Set[str]:
        return self._names("filters", r"^\s[T.][S.]\s+(\S+)\s+\S+->\S+")

    def demuxers(self) -> Set[str]:
        return self._names("demuxers", r"^\s*D[\sE]?\s+([\w,]+)")

    def bsfs(self) -> Set[str]:
        text = self._list("bsfs")
        return {l.strip() for l in text.splitlines()[1:] if l.strip()}

    def has_filter(self, name: str) -> bool:
        return name in self.filters()

    def has_encoder(self, name: str) -> bool:
        return name in self.encoders()

    def has_demuxer(self, name: str) -> bool:
        return any(name in d.split(",") for d in self.demuxers())

    @lru_cache(maxsize=64)
    def encoder_works(self, name: str, pix_fmt: str = "yuv420p") -> bool:
        """Try a tiny encode: listed hardware encoders often lack a device."""
        if not self.has_encoder(name):
            return False
        args = [self.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-f", "lavfi",
                "-i", "testsrc2=s=256x144:r=24:d=0.25", "-pix_fmt", pix_fmt, "-c:v", name,
                "-frames:v", "3", "-f", "null", "-"]
        try:
            p = run(args, timeout=60)
        except (subprocess.TimeoutExpired, OSError):
            return False
        return p.returncode == 0

    def vulkan_works(self, allow_software: bool = False) -> bool:
        """True if libplacebo can run on a Vulkan device.

        Software rasterisers (llvmpipe/lavapipe, SwiftShader) technically work
        but are far slower than the CPU zimg path, so they only count when
        ``allow_software`` is set.
        """
        ok, software, _name = self.vulkan_device()
        return ok and (allow_software or not software)

    @lru_cache(maxsize=4)
    def vulkan_device(self):
        """(works, is_software, device name) for the default Vulkan device."""
        if not self.has_filter("libplacebo"):
            return False, False, ""
        args = [self.ffmpeg, "-hide_banner", "-nostdin", "-v", "verbose",
                "-init_hw_device", "vulkan=vk", "-filter_hw_device", "vk",
                "-f", "lavfi", "-i", "testsrc2=s=128x72:r=24:d=0.1",
                "-vf", "libplacebo=w=256:h=144:upscaler=ewa_lanczossharp:format=yuv420p",
                "-f", "null", "-"]
        try:
            p = run(args, timeout=60)
        except (subprocess.TimeoutExpired, OSError):
            return False, False, ""
        text = p.stderr.decode("utf-8", "replace")
        m = re.search(r"Device \d+ selected:\s*(.+)", text)
        name = m.group(1).strip() if m else ""
        software = bool(re.search(r"\(software\)|llvmpipe|lavapipe|swiftshader", name, re.I))
        return p.returncode == 0, software, name

    # -- probing --------------------------------------------------------------
    def probe(self, input_args: Sequence[str], extra: Sequence[str] = (),
              timeout: Optional[float] = 300) -> dict:
        args = [self.ffprobe, "-hide_banner", "-v", "error", "-of", "json", *extra, *input_args]
        p = run(args, timeout=timeout)
        if p.returncode != 0:
            raise FFmpegError("ffprobe failed: " + p.stderr.decode("utf-8", "replace")[-2000:],
                              p.stderr.decode("utf-8", "replace"))
        return json.loads(p.stdout.decode("utf-8", "replace") or "{}")


_default: Optional[FFmpeg] = None
_default_lock = threading.Lock()


def get_ffmpeg(folder: Optional[str] = None) -> FFmpeg:
    """Shared FFmpeg instance (re-created when a different folder is given)."""
    global _default
    with _default_lock:
        if _default is None or folder:
            _default = FFmpeg(folder=folder)
        return _default


def hexdump_to_bytes(s: str) -> bytes:
    """Parse the hexdump format produced by ``ffprobe -show_data``."""
    out = bytearray()
    for line in s.split("\n"):
        if len(line) < 11 or line[8] != ":":
            continue
        out += bytes.fromhex(line[10:51].replace(" ", ""))
    return bytes(out)


def quote_filter_path(path: str) -> str:
    """Escape a file path for use inside a filtergraph option value."""
    p = path.replace("\\", "/")
    for ch in ("'", ":", ",", "[", "]", ";"):
        p = p.replace(ch, "\\" + ch)
    return p


def args_to_display(args: Iterable[str]) -> str:
    """Render a command line for logs (quoted where needed)."""
    out = []
    for a in args:
        a = str(a)
        if not a or re.search(r"[\s\"';|&<>()\[\]]", a):
            out.append('"' + a.replace('"', '\\"') + '"')
        else:
            out.append(a)
    return " ".join(out)
