"""Planning logic (no FFmpeg needed)."""
import json
import struct
from fractions import Fraction

import pytest

from dtu.analyze import CropAnalysis, ScanAnalysis
from dtu.audio import build_enhance_filters, default_bitrate, plan_audio
from dtu.codecs import CODECS, encoder_args
from dtu.inputs import InputSpec, parse_vts_ifo, resolve_input, vob_set
from dtu.probe import AudioStream, MediaInfo, SubtitleStream, VideoStream, _media_duration
from dtu.settings import AudioSettings, EncodeSettings, PRESETS, Settings, VideoSettings, preset_settings
from dtu.video import build_video_plan, fit_box, parse_manual_crop


def ntsc_info(fps=Fraction(30000, 1001), sar=Fraction(32, 27), h=480, tags=True):
    v = VideoStream(index=0, codec="mpeg2video", width=720, height=h, sar=sar, fps=fps, avg_fps=fps,
                    field_order="tt", pix_fmt="yuv420p", color_space=None, color_primaries=None,
                    color_transfer=None, color_range="tv", chroma_location="left")
    a = AudioStream(index=1, codec="ac3", channels=6, layout="5.1(side)", sample_rate=48000, bit_rate=448000,
                    language="kor", title=None, default=True)
    s = SubtitleStream(index=2, codec="dvd_subtitle", language="kor", title=None, default=False, forced=False)
    return MediaInfo(spec=InputSpec(display="x.vob", paths=["x.vob"]), format_name="mpeg", duration=6000,
                     start_time=0.3, video=[v], audio=[a], subtitles=[s])


class NoCaps:
    """Capability stub: every filter exists, no GPU."""

    def has_filter(self, name):
        return True

    def vulkan_works(self, allow_software=False):
        return False


def test_fit_box():
    assert fit_box(16 / 9, 1920, 1080) == (1920, 1080)
    assert fit_box(4 / 3, 1920, 1080) == (1440, 1080)
    assert fit_box(2.37, 1920, 1080) == (1920, 810)


def test_telecine_plan_full_chain():
    info = ntsc_info()
    scan = ScanAnalysis(scan="telecine")
    crop = CropAnalysis(720, 360, 0, 60, 720, 480)
    vs = VideoSettings(denoise="light", deband="light", sharpen=0.4)
    plan = build_video_plan(info, vs, EncodeSettings(), scan, crop, NoCaps())
    g = plan.filtergraph
    order = ["fps=30000/1001", "fieldmatch", "decimate", "crop=720:360:0:60", "hqdn3d", "zscale", "deband",
             "cas=", "setsar=1"]
    positions = [g.index(k) for k in order]
    assert positions == sorted(positions), g
    assert plan.out_fps == Fraction(24000, 1001)
    assert (plan.out_w, plan.out_h) == (1920, 810)
    assert "matrixin=170m" in g and "primaries=709" in g
    assert "format=yuv420p10le" in g


def test_interlaced_pal_plan_double_rate_and_primaries():
    info = ntsc_info(fps=Fraction(25), sar=Fraction(64, 45), h=576)
    plan = build_video_plan(info, VideoSettings(), EncodeSettings(), ScanAnalysis(scan="interlaced"), None,
                            NoCaps())
    g = plan.filtergraph
    assert "bwdif=mode=send_field" in g
    assert plan.out_fps == 50
    # zscale needs "bt470bg" (not "470bg") for EBU primaries
    assert "primariesin=bt470bg" in g and "matrixin=470bg" in g


def test_progressive_interpolation_and_no_upscale():
    info = ntsc_info(fps=Fraction(24000, 1001))
    vs = VideoSettings(upscale="off", interpolate="60", denoise="off", sharpen=0.0, deband="off")
    plan = build_video_plan(info, vs, EncodeSettings(), ScanAnalysis(scan="progressive"), None, NoCaps())
    assert "minterpolate=fps=60000/1001" in plan.filtergraph
    assert "zscale" not in plan.filtergraph
    assert (plan.out_w, plan.out_h) == (720, 480)
    assert plan.out_fps == Fraction(60000, 1001)


def test_manual_crop_parsing():
    assert parse_manual_crop("704:480:8:0", 720, 480) == (8, 0, 704, 480)
    assert parse_manual_crop("800:480:0:0", 720, 480) is None


@pytest.mark.parametrize("key", sorted(CODECS))
def test_encoder_args_are_well_formed(key):
    es = EncodeSettings(codec=key)
    args = encoder_args(es, Fraction(24000, 1001))
    assert args[0] == "-c:v" and args[1] == CODECS[key].encoder
    assert "-pix_fmt" in args
    assert len(args) % 2 == 0


def test_h264_is_8bit_vvc_is_10bit():
    assert encoder_args(EncodeSettings(codec="h264"))[-1] == "yuv420p"
    assert encoder_args(EncodeSettings(codec="vvc", bit_depth=8))[-1] == "yuv420p10le"


def test_audio_downmix_dialog_and_opus_layout():
    st = AudioStream(1, "ac3", 6, "5.1(side)", 48000, 448000, "kor", None, True)
    f, layout, notes = build_enhance_filters(st, AudioSettings(channels="stereo_dialog", dialog_boost=4.0,
                                                                drc="night"))
    joined = ",".join(f)
    assert "FC=1.5849*FC" in joined  # +4 dB centre
    assert "pan=stereo|FL<FC+0.30*FL+0.30*SL" in joined
    assert "dynaudnorm" in joined and layout == "stereo"
    plans = plan_audio(ntsc_info(), AudioSettings(mode="enhance", codec="opus"))
    assert plans[0].bitrate == 320
    p = plans[0]
    p.measured_lufs, p.gain_db = -30.0, 7.0
    full = ",".join(p.full_filters())
    assert "volume=+7.00dB" in full and "alimiter" in full and "channelmap=channel_layout=5.1" in full


def test_audio_copy_and_keep_original():
    plans = plan_audio(ntsc_info(), AudioSettings(mode="copy"))
    assert plans[0].action == "copy" and plans[0].codec_args(0) == ["-c:a:0", "copy"]
    plans = plan_audio(ntsc_info(), AudioSettings(mode="enhance", keep_original=True))
    assert [p.action for p in plans] == ["encode", "copy"]
    assert default_bitrate("opus", 2) == 160 and default_bitrate("ac3", 6) == 640


def test_settings_json_roundtrip_and_presets():
    s = preset_settings("quality")
    s.encode.codec = "hevc"
    s2 = Settings.from_json(s.to_json())
    assert s2.to_dict() == s.to_dict()
    s3 = preset_settings("night", s2)
    assert s3.encode.codec == "hevc"  # presets never switch the codec
    assert s3.audio.drc == "night" and s3.preset == "night"
    for name in PRESETS:
        assert preset_settings(name).video.denoise in ("off", "light", "medium", "strong", "very_strong")
    # unknown keys in saved settings are ignored
    assert Settings.from_dict({"video": {"nope": 1, "denoise": "strong"}}).video.denoise == "strong"


def test_media_duration_ignores_broken_subtitle_duration():
    streams = [{"codec_type": "video", "tags": {"DURATION": "01:30:00.000000000"}},
               {"codec_type": "subtitle", "tags": {"DURATION": "1193:02:47.295000000"}}]
    assert _media_duration({"duration": "4294967.295"}, streams) == 5400.0


def _fake_ifo(path):
    data = bytearray(0x1000)
    data[0:12] = b"DVDVIDEO-VTS"
    data[0xCC:0xD0] = struct.pack(">I", 1)  # PGCITI at sector 1
    data[0x202:0x204] = struct.pack(">H", 2)
    data[0x204] = 0x04  # AC3, language present
    data[0x206:0x208] = b"ko"
    data[0x20C] = 0x04
    data[0x20E:0x210] = b"en"
    data[0x254:0x256] = struct.pack(">H", 1)
    data[0x256] = 0x01
    data[0x258:0x25A] = b"ko"
    base = 2048
    data[base:base + 2] = struct.pack(">H", 1)
    data[base + 12:base + 16] = struct.pack(">I", 16)
    pgc = base + 16
    data[pgc + 0x0C:pgc + 0x0E] = struct.pack(">H", 0x8000 | (0 << 8))
    data[pgc + 0x0E:pgc + 0x10] = struct.pack(">H", 0x8000 | (1 << 8))
    data[pgc + 0x1C:pgc + 0x20] = struct.pack(">I", 0x80000000 | (0 << 24) | (0 << 16))
    for i in range(16):
        data[pgc + 0xA4 + 4 * i:pgc + 0xA8 + 4 * i] = bytes([0, 235 if i == 1 else 16, 128, 128])
    with open(path, "wb") as f:
        f.write(bytes(data))


def test_ifo_palette_and_languages(tmp_path):
    p = tmp_path / "VTS_01_0.IFO"
    _fake_ifo(p)
    info = parse_vts_ifo(str(p))
    assert info.palette[1] == (255, 255, 255) and info.palette[0] == (0, 0, 0)
    assert info.audio_langs == ["ko", "en"]
    assert info.audio_lang_for_id(0x80) == "ko" and info.audio_lang_for_id(0x81) == "en"
    assert info.sub_lang_for_id(0x20) == "ko"


def test_vob_set_and_resolve(tmp_path):
    for n in ("VTS_01_0.VOB", "VTS_01_1.VOB", "VTS_01_2.VOB", "VTS_02_1.VOB"):
        (tmp_path / n).write_bytes(b"\0" * 10)
    _fake_ifo(tmp_path / "VTS_01_0.IFO")
    vobs, ifo = vob_set(str(tmp_path / "VTS_01_1.VOB"))
    assert [v.split("/")[-1] for v in vobs] == ["VTS_01_1.VOB", "VTS_01_2.VOB"]
    assert ifo.endswith("VTS_01_0.IFO")
    spec = resolve_input(str(tmp_path / "VTS_01_2.VOB"))
    assert spec.url.startswith("concat:") and spec.ifo
    assert spec.args()[-2:] == ["-i", spec.url]
