"""Generate a deterministic grayscale heightmap PNG for hfield terrain."""
import math
import struct
import zlib


def _crc(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + chunk_type + data + _crc(chunk_type, data)


def write_grayscale_png(path: str, pixels, width: int, height: int) -> None:
    # Each scanline: filter byte 0 + raw pixels
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        row = pixels[y]
        raw.extend(row)
    compressed = zlib.compress(bytes(raw), level=6)

    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png.extend(_chunk(b"IHDR", ihdr))
    png.extend(_chunk(b"IDAT", compressed))
    png.extend(_chunk(b"IEND", b""))

    with open(path, "wb") as f:
        f.write(png)


def generate_heightmap(width: int, height: int):
    pixels = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            fx = x / (width - 1)
            fy = y / (height - 1)
            h = (
                0.6 * math.sin(2.0 * math.pi * fx * 2.0)
                + 0.4 * math.cos(2.0 * math.pi * fy * 1.5)
                + 0.3 * math.sin(2.0 * math.pi * (fx + fy) * 1.0)
            )
            # Normalize to 0..255
            v = int((h + 1.3) / 2.6 * 255)
            v = 0 if v < 0 else 255 if v > 255 else v
            row.append(v)
        pixels.append(row)
    return pixels


def main():
    width = 256
    height = 256
    pixels = generate_heightmap(width, height)
    write_grayscale_png("terrain.png", pixels, width, height)
    print("terrain.png generated")


if __name__ == "__main__":
    main()
