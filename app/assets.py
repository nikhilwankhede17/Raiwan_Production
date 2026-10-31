"""Reference-image validation for the local asset library.

Nothing here touches the network and no new dependency is required: formats are detected and
structurally walked with the standard library. If Pillow happens to be installed it is used as an
extra, deeper check (``Image.verify()``), and the asset records which level of checking was done.

An image is only accepted when its container actually parses end to end, so truncated or
mislabelled files are rejected instead of being copied into the project.
"""
from __future__ import annotations

import os
import struct
import zlib
from typing import Any

# Smallest size that still lets every header parser read its fields; real images can be tiny
# (a 13x7 GIF is 56 bytes), so the per-format structural walk does the actual rejecting.
MIN_BYTES = 26
DEFAULT_MAX_MB = 40.0
MAX_PIXELS_PER_SIDE = 30_000
MAX_TOTAL_PIXELS = 80_000_000

# detected format -> canonical suffix (the stored file always uses the detected format's suffix)
FORMAT_SUFFIX = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp", "TIFF": ".tiff"}
ALLOWED_FORMATS = tuple(FORMAT_SUFFIX)
SUFFIX_ALIASES = {".jpeg": ".jpg", ".tif": ".tiff"}


def max_bytes() -> int:
    try:
        megabytes = float(os.environ.get("FILM_HOUSE_MAX_ASSET_MB", DEFAULT_MAX_MB))
    except (TypeError, ValueError):
        megabytes = DEFAULT_MAX_MB
    return int(max(0.01, megabytes) * 1024 * 1024)


class ImageError(Exception):
    def __init__(self, code: str, message: str, detail: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


def _corrupt(message: str) -> ImageError:
    return ImageError("CORRUPT_IMAGE", f"This image file is not complete or not valid: {message}")


# --------------------------------------------------------------------------- per-format walkers

def _png(data: bytes) -> tuple[int, int]:
    position, size = 8, len(data)
    width = height = None
    seen_end = False
    while position < size:
        if position + 8 > size:
            raise _corrupt("a PNG chunk header runs past the end of the file")
        (length,) = struct.unpack(">I", data[position:position + 4])
        tag = data[position + 4:position + 8]
        end = position + 8 + length + 4
        if length > size or end > size:
            raise _corrupt(f"PNG chunk '{tag.decode('ascii', 'replace')}' claims {length} bytes but the file ends")
        body = data[position + 8:position + 8 + length]
        (declared_crc,) = struct.unpack(">I", data[end - 4:end])
        if zlib.crc32(tag + body) & 0xFFFFFFFF != declared_crc:
            raise _corrupt(f"PNG chunk '{tag.decode('ascii', 'replace')}' failed its checksum")
        if tag == b"IHDR":
            if length < 13:
                raise _corrupt("the PNG header chunk is too short")
            width, height = struct.unpack(">II", body[:8])
        elif tag == b"IEND":
            seen_end = True
            position = end
            break
        position = end
    if width is None:
        raise _corrupt("the PNG has no IHDR header chunk")
    if not seen_end:
        raise _corrupt("the PNG is truncated (no IEND chunk)")
    return width, height


_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _jpeg(data: bytes) -> tuple[int, int]:
    size = len(data)
    if data[-2:] != b"\xff\xd9":
        raise _corrupt("the JPEG is truncated (no end-of-image marker)")
    position = 2
    dimensions: tuple[int, int] | None = None
    while position < size - 1:
        if data[position] != 0xFF:
            position += 1
            continue
        marker = data[position + 1]
        if marker in (0xFF, 0x00):
            position += 1
            continue
        if marker == 0xD9:
            break
        if 0xD0 <= marker <= 0xD8 or marker == 0x01:
            position += 2
            continue
        if position + 4 > size:
            raise _corrupt("a JPEG segment header runs past the end of the file")
        (length,) = struct.unpack(">H", data[position + 2:position + 4])
        if length < 2 or position + 2 + length > size:
            raise _corrupt("a JPEG segment length runs past the end of the file")
        if marker in _SOF_MARKERS and dimensions is None:
            segment = data[position + 4:position + 2 + length]
            if len(segment) < 5:
                raise _corrupt("the JPEG frame header is too short")
            height, width = struct.unpack(">HH", segment[1:5])
            dimensions = (width, height)
        if marker == 0xDA:              # start of scan: entropy data follows, stop header parsing
            break
        position += 2 + length
    if dimensions is None:
        raise _corrupt("the JPEG has no frame header")
    return dimensions


def _webp(data: bytes) -> tuple[int, int]:
    (riff_size,) = struct.unpack("<I", data[4:8])
    if riff_size + 8 > len(data):
        raise _corrupt(f"the WebP header claims {riff_size + 8} bytes but the file has {len(data)}")
    position = 12
    while position + 8 <= len(data):
        tag = data[position:position + 4]
        (length,) = struct.unpack("<I", data[position + 4:position + 8])
        body = data[position + 8:position + 8 + length]
        if len(body) < length:
            raise _corrupt(f"WebP chunk '{tag.decode('ascii', 'replace')}' is truncated")
        if tag == b"VP8X" and length >= 10:
            width = int.from_bytes(body[4:7], "little") + 1
            height = int.from_bytes(body[7:10], "little") + 1
            return width, height
        if tag == b"VP8 " and length >= 10:
            if body[3:6] != b"\x9d\x01\x2a":
                raise _corrupt("the WebP keyframe header is invalid")
            width = struct.unpack("<H", body[6:8])[0] & 0x3FFF
            height = struct.unpack("<H", body[8:10])[0] & 0x3FFF
            return width, height
        if tag == b"VP8L" and length >= 5:
            bits = int.from_bytes(body[1:5], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        position += 8 + length + (length % 2)
    raise _corrupt("the WebP file has no image data chunk")


def _gif(data: bytes) -> tuple[int, int]:
    if data[-1:] != b"\x3b":
        raise _corrupt("the GIF is truncated (no trailer byte)")
    width, height = struct.unpack("<HH", data[6:10])
    return width, height


def _bmp(data: bytes) -> tuple[int, int]:
    (declared,) = struct.unpack("<I", data[2:6])
    if declared > len(data):
        raise _corrupt(f"the BMP header claims {declared} bytes but the file has {len(data)}")
    (header_size,) = struct.unpack("<I", data[14:18])
    if header_size >= 40:
        width, height = struct.unpack("<ii", data[18:26])
    elif header_size == 12:
        width, height = struct.unpack("<hh", data[18:22])
    else:
        raise _corrupt("unsupported BMP header size")
    return abs(width), abs(height)


def _tiff(data: bytes) -> tuple[int | None, int | None]:
    little = data[:2] == b"II"
    order = "<" if little else ">"
    (offset,) = struct.unpack(order + "I", data[4:8])
    if offset + 2 > len(data):
        raise _corrupt("the TIFF directory offset is past the end of the file")
    (count,) = struct.unpack(order + "H", data[offset:offset + 2])
    width = height = None
    for index in range(count):
        entry = offset + 2 + index * 12
        if entry + 12 > len(data):
            raise _corrupt("the TIFF directory is truncated")
        tag, field_type, _count = struct.unpack(order + "HHI", data[entry:entry + 8])
        raw = data[entry + 8:entry + 12]
        value = struct.unpack(order + "H", raw[:2])[0] if field_type == 3 else struct.unpack(order + "I", raw)[0]
        if tag == 256:
            width = value
        elif tag == 257:
            height = value
    if width is None or height is None:
        raise _corrupt("the TIFF directory has no image size tags")
    return width, height


def detect_format(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF"
    if data[:2] == b"BM":
        return "BMP"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    return None


_WALKERS = {"PNG": _png, "JPEG": _jpeg, "WEBP": _webp, "GIF": _gif, "BMP": _bmp, "TIFF": _tiff}


# --------------------------------------------------------------------------- public API

def _pillow_check(data: bytes) -> tuple[int, int] | None:
    """Optional deeper verification. Returns dimensions, or None when Pillow is absent."""
    try:
        from PIL import Image  # optional; not in requirements.txt
    except Exception:
        return None
    import io
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            return int(image.width), int(image.height)
    except Exception as exc:
        raise ImageError("CORRUPT_IMAGE",
                         f"This image file is not complete or not valid: {type(exc).__name__}: {exc}"[:300]) from exc


def inspect_image(data: bytes, filename: str = "") -> dict[str, Any]:
    """Validate type, size and structure. Raises ImageError with a readable reason."""
    limit = max_bytes()
    if not data:
        raise ImageError("EMPTY_FILE", "The file is empty.")
    if len(data) < MIN_BYTES:
        raise ImageError("CORRUPT_IMAGE", f"The file is only {len(data)} bytes, too small to be an image.")
    if len(data) > limit:
        raise ImageError("FILE_TOO_LARGE",
                         f"The file is {len(data) / 1024 / 1024:.1f} MB; the limit is "
                         f"{limit / 1024 / 1024:.0f} MB (set FILM_HOUSE_MAX_ASSET_MB to change it).")

    image_format = detect_format(data)
    if image_format is None:
        raise ImageError("UNSUPPORTED_TYPE",
                         "That file is not a supported image. Allowed types: "
                         f"{', '.join(ALLOWED_FORMATS)} (detected by content, not by file name).")

    width, height = _WALKERS[image_format](data)
    pillow_size = _pillow_check(data)
    integrity = "stdlib-header"
    if pillow_size is not None:
        integrity = "verified"
        if width and pillow_size[0] and (abs(pillow_size[0] - width) > 1 or abs(pillow_size[1] - height) > 1):
            width, height = pillow_size        # trust the full decoder when they disagree
        else:
            width, height = pillow_size

    for value, label in ((width, "width"), (height, "height")):
        if value is not None and not 1 <= value <= MAX_PIXELS_PER_SIDE:
            raise ImageError("BAD_DIMENSIONS", f"Image {label} of {value} px is outside the supported range "
                                               f"(1-{MAX_PIXELS_PER_SIDE}).")
    if width and height and width * height > MAX_TOTAL_PIXELS:
        raise ImageError("BAD_DIMENSIONS", f"Image is {width}x{height} px, larger than the "
                                           f"{MAX_TOTAL_PIXELS // 1_000_000} megapixel limit.")

    suffix = FORMAT_SUFFIX[image_format]
    given = os.path.splitext(filename or "")[1].lower()
    given = SUFFIX_ALIASES.get(given, given)
    return {
        "format": image_format, "width": width, "height": height, "suffix": suffix,
        "bytes": len(data), "integrity": integrity,
        "extension_mismatch": bool(given) and given != suffix,
        "given_extension": given,
    }


def read_source_file(path_text: str) -> tuple[bytes, str]:
    """Read an image the user picked from their own machine. Local filesystem only."""
    from pathlib import Path
    if not path_text or not path_text.strip():
        raise ImageError("BAD_SOURCE_PATH", "No source path was given.")
    candidate = Path(os.path.expanduser(path_text.strip()))
    if not candidate.is_absolute():
        raise ImageError("BAD_SOURCE_PATH", f"'{path_text}' must be an absolute path to a file on this computer.")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ImageError("SOURCE_NOT_FOUND", f"'{path_text}' could not be opened: {exc}") from exc
    if not resolved.is_file():
        raise ImageError("SOURCE_NOT_FOUND", f"'{path_text}' is not a file.")
    size = resolved.stat().st_size
    if size > max_bytes():
        raise ImageError("FILE_TOO_LARGE",
                         f"'{resolved.name}' is {size / 1024 / 1024:.1f} MB; the limit is "
                         f"{max_bytes() / 1024 / 1024:.0f} MB.")
    try:
        return resolved.read_bytes(), resolved.name
    except OSError as exc:
        raise ImageError("SOURCE_UNREADABLE", f"'{path_text}' could not be read: {exc}") from exc


def decode_base64(payload: str, filename: str) -> bytes:
    """Decode a browser upload. Accepts a bare base64 string or a data: URL."""
    import base64
    text = (payload or "").strip()
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    text = "".join(text.split())
    if not text:
        raise ImageError("EMPTY_FILE", f"No image data was sent for '{filename or 'the file'}'.")
    if len(text) > max_bytes() * 4 // 3 + 1024:
        raise ImageError("FILE_TOO_LARGE", f"The upload is larger than the "
                                           f"{max_bytes() / 1024 / 1024:.0f} MB limit.")
    try:
        return base64.b64decode(text, validate=True)
    except Exception as exc:
        raise ImageError("BAD_ENCODING", f"The uploaded data is not valid base64 ({exc}).") from exc
