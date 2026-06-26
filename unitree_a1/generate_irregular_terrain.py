"""Generate a deterministic irregular grayscale heightmap for hfield terrain."""
import math
import random
import struct
import zlib
from pathlib import Path


def _crc(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + chunk_type + data + _crc(chunk_type, data)


def write_grayscale_png(path: Path, pixels, width: int, height: int) -> None:
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y])

    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png.extend(_chunk(b"IHDR", ihdr))
    png.extend(_chunk(b"IDAT", zlib.compress(bytes(raw), level=6)))
    png.extend(_chunk(b"IEND", b""))
    path.write_bytes(png)


def _smooth(values, width, height, rounds=2):
    current = values
    for _ in range(rounds):
        nxt = [[0.0 for _ in range(width)] for _ in range(height)]
        for y in range(height):
            for x in range(width):
                total = 0.0
                weight = 0.0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy = min(max(y + dy, 0), height - 1)
                        xx = min(max(x + dx, 0), width - 1)
                        w = 2.0 if dx == 0 and dy == 0 else 1.0
                        total += current[yy][xx] * w
                        weight += w
                nxt[y][x] = total / weight
        current = nxt
    return current


def generate_heightmap(width: int, height: int, seed: int = 20260625):
    rng = random.Random(seed)
    bumps = []
    for _ in range(38):
        bumps.append(
            (
                rng.uniform(0.0, 1.0),
                rng.uniform(0.0, 1.0),
                rng.uniform(-0.75, 0.8),
                rng.uniform(0.025, 0.10),
            )
        )

    values = []
    for y in range(height):
        row = []
        fy = y / (height - 1)
        for x in range(width):
            fx = x / (width - 1)
            h = (
                0.25 * math.sin(2.0 * math.pi * (2.7 * fx + 0.55 * fy))
                + 0.20 * math.cos(2.0 * math.pi * (2.1 * fy - 0.35 * fx))
                + 0.18 * math.sin(2.0 * math.pi * (5.7 * fx + 3.8 * fy))
                + 0.12 * math.cos(2.0 * math.pi * (8.0 * fx - 5.1 * fy))
                + 0.06 * math.sin(2.0 * math.pi * (12.5 * fx + 1.7 * fy))
            )
            for cx, cy, amp, sigma in bumps:
                dx = fx - cx
                dy = fy - cy
                h += amp * math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            row.append(h)
        values.append(row)

    values = _smooth(values, width, height, rounds=1)
    flat = [v for row in values for v in row]
    lo, hi = min(flat), max(flat)
    pixels = []
    for row in values:
        png_row = bytearray()
        for h in row:
            v = int((h - lo) / max(hi - lo, 1e-8) * 255)
            png_row.append(0 if v < 0 else 255 if v > 255 else v)
        pixels.append(png_row)
    return pixels


def main():
    width = 256
    height = 256
    output = Path(__file__).resolve().parent / "assets" / "terrain_irregular.png"
    write_grayscale_png(output, generate_heightmap(width, height), width, height)
    print(f"{output} generated")


if __name__ == "__main__":
    main()
