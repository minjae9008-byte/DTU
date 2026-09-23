#!/usr/bin/env python3
"""Apply soft 3:2 pulldown flags to a progressive 23.976p MPEG-2 elementary stream.

This mirrors what DVD authoring tools (and DGPulldown) do for NTSC film
discs: the pictures stay progressive 24p, the stream is flagged 29.97 fps and
the repeat_first_field / top_field_first flags make players display 3:2
pulldown.  Used to build soft-telecine test material, which FFmpeg itself
cannot produce.

The input must be encoded without B-frames (coding order = display order):
    ffmpeg -i in -r 24000/1001 -c:v mpeg2video -bf 0 -an in.m2v
    python tools/soft_pulldown.py in.m2v out.m2v
"""
import sys

# display-order cadence: (top_field_first, repeat_first_field)
CADENCE = [(1, 1), (0, 0), (0, 1), (1, 0)]


def apply_pulldown(data: bytearray) -> int:
    n = 0
    i = 0
    frame = 0
    while True:
        i = data.find(b"\x00\x00\x01", i)
        if i < 0 or i + 8 >= len(data):
            break
        code = data[i + 3]
        p = i + 4
        if code == 0xB3:  # sequence header: frame_rate_code -> 4 (30000/1001)
            data[p + 3] = (data[p + 3] & 0xF0) | 0x04
        elif code == 0xB5:
            ext = data[p] >> 4
            if ext == 1:  # sequence extension: progressive_sequence = 0
                data[p + 1] &= ~0x08 & 0xFF
            elif ext == 8:  # picture coding extension
                tff, rff = CADENCE[frame % 4]
                b3 = data[p + 3] & ~(0x80 | 0x02) & 0xFF
                data[p + 3] = b3 | (0x80 if tff else 0) | (0x02 if rff else 0)
                data[p + 4] |= 0x80  # progressive_frame
                frame += 1
                n += 1
        i += 4
    return n


def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = bytearray(open(src, "rb").read())
    n = apply_pulldown(data)
    open(dst, "wb").write(bytes(data))
    print(f"{n} pictures flagged")


if __name__ == "__main__":
    main()
