"""Phase 3 unit tests: image validation and asset file storage. No network, no ComfyUI."""
import struct
import zlib

import pytest

from app import assets, storage
from app.assets import (
    ALLOWED_FORMATS, ImageError, decode_base64, detect_format, inspect_image, max_bytes, read_source_file,
)

PROJECT_ID = "a" * 32


# --------------------------------------------------------------------------- image builders

def png_bytes(width=8, height=6, rgb=(200, 120, 40), noisy=False):
    """A real, decodable PNG built with the standard library only."""
    import os
    if noisy:                                   # incompressible pixels, for size-limit tests
        raw = b"".join(b"\x00" + os.urandom(width * 3) for _ in range(height))
    else:
        raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def jpeg_bytes(width=16, height=12):
    sof = struct.pack(">BHHB", 8, height, width, 1) + bytes([1, 0x11, 0])
    return (b"\xff\xd8\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            + b"\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof
            + b"\xff\xda" + struct.pack(">H", 8) + bytes([1, 1, 0, 0, 63, 0])
            + b"\x00" * 64 + b"\xff\xd9")


def real_image(image_format, width, height):
    """Encode a genuine image with Pillow (skips the test when Pillow is not installed)."""
    pil = pytest.importorskip("PIL.Image", reason="needs Pillow to encode a real sample image")
    import io
    buffer = io.BytesIO()
    pil.new("RGB", (width, height), (120, 60, 180)).save(buffer, format=image_format)
    return buffer.getvalue()


def gif_bytes(width=5, height=7):
    return (b"GIF89a" + struct.pack("<HH", width, height) + b"\xf7\x00\x00" + b"\x00" * 60
            + b"\x21\xf9\x04\x00\x00\x00\x00\x00" + b"\x3b")


def bmp_bytes(width=4, height=3):
    header = b"BM" + struct.pack("<IHHI", 122, 0, 0, 54)
    dib = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, 68, 2835, 2835, 0, 0)
    return header + dib + b"\x00" * 68


def webp_bytes(width=10, height=9):
    vp8 = b"\x00\x00\x00" + b"\x9d\x01\x2a" + struct.pack("<HH", width, height) + b"\x00" * 40
    chunk = b"VP8 " + struct.pack("<I", len(vp8)) + vp8
    body = b"WEBP" + chunk
    return b"RIFF" + struct.pack("<I", len(body)) + body


def tiff_bytes(width=6, height=4):
    entries = [(256, 3, 1, width), (257, 3, 1, height), (258, 3, 1, 8)]
    data = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", len(entries))
    for tag, field_type, count, value in entries:
        data += struct.pack("<HHI", tag, field_type, count) + struct.pack("<H", value) + b"\x00\x00"
    return data + struct.pack("<I", 0) + b"\x00" * 40


# --------------------------------------------------------------------------- happy paths

@pytest.mark.parametrize("builder,expected_format,size", [
    (png_bytes, "PNG", (8, 6)), (jpeg_bytes, "JPEG", (16, 12)),
])
def test_stdlib_built_images_are_accepted(builder, expected_format, size):
    data = builder()
    assert detect_format(data) == expected_format
    info = inspect_image(data, f"x.{expected_format.lower()}")
    assert info["format"] == expected_format
    assert (info["width"], info["height"]) == size
    assert info["bytes"] == len(data)
    assert info["integrity"] in ("verified", "stdlib-header")


@pytest.mark.parametrize("image_format,expected,size", [
    ("PNG", "PNG", (17, 11)), ("JPEG", "JPEG", (24, 18)), ("GIF", "GIF", (13, 7)),
    ("BMP", "BMP", (9, 5)), ("WEBP", "WEBP", (20, 14)), ("TIFF", "TIFF", (6, 4)),
])
def test_every_supported_format_round_trips(image_format, expected, size):
    """Real encoded images of each accepted type are read correctly (needs Pillow to encode them)."""
    data = real_image(image_format, *size)
    assert detect_format(data) == expected
    info = inspect_image(data, f"sample.{image_format.lower()}")
    assert info["format"] == expected and (info["width"], info["height"]) == size


@pytest.mark.parametrize("builder,expected_format,size", [
    (lambda: png_bytes(12, 9), "PNG", (12, 9)),
    (lambda: jpeg_bytes(30, 20), "JPEG", (30, 20)),
    (gif_bytes, "GIF", (5, 7)), (bmp_bytes, "BMP", (4, 3)),
    (webp_bytes, "WEBP", (10, 9)), (tiff_bytes, "TIFF", (6, 4)),
])
def test_stdlib_parsers_read_every_container_without_pillow(builder, expected_format, size, monkeypatch):
    """Pillow is optional: the header parsers alone must identify and size each format."""
    monkeypatch.setattr(assets, "_pillow_check", lambda data: None)
    info = inspect_image(builder(), f"x.{expected_format.lower()}")
    assert info["format"] == expected_format
    assert (info["width"], info["height"]) == size
    assert info["integrity"] == "stdlib-header"


def test_allowed_formats_are_advertised():
    assert set(ALLOWED_FORMATS) == {"PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF"}


def test_stdlib_only_path_works_without_pillow(monkeypatch):
    """Pillow is optional: validation must still work (and say so) when it is absent."""
    monkeypatch.setattr(assets, "_pillow_check", lambda data: None)
    info = inspect_image(png_bytes(20, 10), "a.png")
    assert info["integrity"] == "stdlib-header" and (info["width"], info["height"]) == (20, 10)


def test_extension_mismatch_is_reported_and_storage_uses_detected_format():
    info = inspect_image(png_bytes(), "photo.jpg")
    assert info["extension_mismatch"] is True and info["given_extension"] == ".jpg"
    assert info["suffix"] == ".png"
    assert inspect_image(jpeg_bytes(), "photo.jpeg")["extension_mismatch"] is False


# --------------------------------------------------------------------------- rejections

@pytest.mark.parametrize("data,code", [
    (b"", "EMPTY_FILE"),
    (b"short", "CORRUPT_IMAGE"),
    (b"%PDF-1.7" + b"x" * 200, "UNSUPPORTED_TYPE"),
    (b"<html><body>not an image</body></html>" * 5, "UNSUPPORTED_TYPE"),
    (b"\x00" * 500, "UNSUPPORTED_TYPE"),
])
def test_unsupported_and_empty_files_are_refused(data, code):
    with pytest.raises(ImageError) as info:
        inspect_image(data, "x.png")
    assert info.value.code == code


@pytest.mark.parametrize("mangle", [
    lambda d: d[:-6],                                   # cut the tail off
    lambda d: d[:len(d) // 2],                          # half a file
    lambda d: d[:30] + b"\x00" + d[31:],                # flipped byte inside IHDR
    lambda d: d.replace(b"IEND", b"IENX"),              # no end marker
    lambda d: d[:8] + d[33:],                           # IHDR removed
    lambda d: d + b"trailing garbage",                  # extra bytes after IEND are tolerated
])
def test_damaged_png_is_either_refused_or_clean(mangle):
    """Anything structurally broken must be refused with a readable reason, never copied in."""
    data = mangle(png_bytes(64, 64, noisy=True))
    try:
        info = inspect_image(data, "x.png")
    except ImageError as exc:
        assert exc.code == "CORRUPT_IMAGE" and len(exc.message) > 30
        return
    assert info["format"] == "PNG"                      # only the trailing-garbage case may pass


def test_truncation_and_checksum_messages_are_specific():
    data = png_bytes(64, 64, noisy=True)
    with pytest.raises(ImageError) as info:
        inspect_image(data[:-6], "x.png")
    assert "runs past the end" in info.value.message
    with pytest.raises(ImageError) as info:
        inspect_image(data[:30] + b"\x00" + data[31:], "x.png")
    assert "failed its checksum" in info.value.message
    with pytest.raises(ImageError) as info:
        inspect_image(png_bytes()[:8] + b"\x00" * 100, "x.png")
    assert info.value.code == "CORRUPT_IMAGE"


def test_truncated_jpeg_and_gif_are_refused():
    with pytest.raises(ImageError) as info:
        inspect_image(jpeg_bytes()[:-2], "x.jpg")
    assert info.value.code == "CORRUPT_IMAGE" and "end-of-image" in info.value.message
    with pytest.raises(ImageError) as info:
        inspect_image(gif_bytes()[:-1], "x.gif")
    assert info.value.code == "CORRUPT_IMAGE" and "trailer" in info.value.message


def test_webp_with_a_lying_header_is_refused():
    data = bytearray(webp_bytes())
    data[4:8] = struct.pack("<I", 999_999)
    with pytest.raises(ImageError) as info:
        inspect_image(bytes(data), "x.webp")
    assert info.value.code == "CORRUPT_IMAGE" and "claims" in info.value.message


def test_file_size_limit(monkeypatch):
    monkeypatch.setenv("FILM_HOUSE_MAX_ASSET_MB", "0.01")      # 0.01 MB is the floor the app allows
    assert max_bytes() == 10485
    with pytest.raises(ImageError) as info:
        inspect_image(png_bytes(200, 200, noisy=True), "big.png")
    assert info.value.code == "FILE_TOO_LARGE" and "FILM_HOUSE_MAX_ASSET_MB" in info.value.message
    monkeypatch.setenv("FILM_HOUSE_MAX_ASSET_MB", "not-a-number")
    assert max_bytes() == int(assets.DEFAULT_MAX_MB * 1024 * 1024)


def test_absurd_dimensions_are_refused(monkeypatch):
    monkeypatch.setattr(assets, "_pillow_check", lambda data: None)
    fake = bytearray(png_bytes())
    fake[16:24] = struct.pack(">II", 40_000, 40_000)
    tag_start, tag_end = 12, 16
    body = bytes(fake[tag_end:tag_end + 13])
    fake[29:33] = struct.pack(">I", zlib.crc32(bytes(fake[tag_start:tag_end]) + body) & 0xFFFFFFFF)
    with pytest.raises(ImageError) as info:
        inspect_image(bytes(fake), "x.png")
    assert info.value.code == "BAD_DIMENSIONS"


# --------------------------------------------------------------------------- source files and base64

def test_read_source_file_rejects_relative_and_missing_paths(tmp_path):
    with pytest.raises(ImageError) as info:
        read_source_file("relative/path.png")
    assert info.value.code == "BAD_SOURCE_PATH"
    with pytest.raises(ImageError) as info:
        read_source_file(str(tmp_path / "nope.png"))
    assert info.value.code == "SOURCE_NOT_FOUND"
    with pytest.raises(ImageError) as info:
        read_source_file(str(tmp_path))
    assert info.value.code in ("SOURCE_NOT_FOUND", "SOURCE_UNREADABLE")
    with pytest.raises(ImageError) as info:
        read_source_file("   ")
    assert info.value.code == "BAD_SOURCE_PATH"


def test_read_source_file_reads_bytes(tmp_path):
    path = tmp_path / "face.png"
    path.write_bytes(png_bytes())
    data, name = read_source_file(str(path))
    assert data == png_bytes() and name == "face.png"


def test_read_source_file_honours_the_size_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("FILM_HOUSE_MAX_ASSET_MB", "0.01")
    path = tmp_path / "big.png"
    path.write_bytes(png_bytes(200, 200, noisy=True))
    with pytest.raises(ImageError) as info:
        read_source_file(str(path))
    assert info.value.code == "FILE_TOO_LARGE"


def test_base64_accepts_data_urls_and_rejects_junk():
    import base64
    raw = png_bytes()
    plain = base64.b64encode(raw).decode()
    assert decode_base64(plain, "a.png") == raw
    assert decode_base64(f"data:image/png;base64,{plain}", "a.png") == raw
    assert decode_base64("\n".join([plain[:10], plain[10:]]), "a.png") == raw
    with pytest.raises(ImageError) as info:
        decode_base64("not base64 ***", "a.png")
    assert info.value.code == "BAD_ENCODING"
    with pytest.raises(ImageError) as info:
        decode_base64("", "a.png")
    assert info.value.code == "EMPTY_FILE"


# --------------------------------------------------------------------------- asset file storage

def test_asset_files_are_write_once_and_read_only(isolated_workspace):
    data = png_bytes()
    relative, digest = storage.write_asset_file(PROJECT_ID, "b" * 12, ".png", data)
    path = isolated_workspace / PROJECT_ID / relative
    assert relative == "assets/bbbbbbbbbbbb.png"
    assert path.read_bytes() == data
    assert digest == __import__("hashlib").sha256(data).hexdigest()
    assert not (path.stat().st_mode & 0o200)                       # read-only on disk
    with pytest.raises(storage.StorageError):
        storage.write_asset_file(PROJECT_ID, "b" * 12, ".png", b"other")
    assert path.read_bytes() == data                               # original preserved


def test_asset_file_status_detects_missing_and_modified(isolated_workspace):
    data = png_bytes()
    relative, digest = storage.write_asset_file(PROJECT_ID, "c" * 12, ".png", data)
    assert storage.asset_file_status(PROJECT_ID, relative, digest) == "ok"
    path = isolated_workspace / PROJECT_ID / relative
    path.chmod(0o666)
    path.write_bytes(png_bytes(9, 9))
    assert storage.asset_file_status(PROJECT_ID, relative, digest) == "modified"
    path.unlink()
    assert storage.asset_file_status(PROJECT_ID, relative, digest) == "missing"
    assert storage.asset_file_status(PROJECT_ID, "../outside.png", digest) == "missing"


def test_read_asset_file_verifies_the_hash(isolated_workspace):
    relative, digest = storage.write_asset_file(PROJECT_ID, "d" * 12, ".png", png_bytes())
    assert storage.read_asset_file(PROJECT_ID, relative, digest) == png_bytes()
    path = isolated_workspace / PROJECT_ID / relative
    path.chmod(0o666)
    path.write_bytes(png_bytes(3, 3))
    with pytest.raises(storage.StorageError) as info:
        storage.read_asset_file(PROJECT_ID, relative, digest)
    assert "integrity check" in str(info.value)
    path.unlink()
    with pytest.raises(storage.StorageError) as info:
        storage.read_asset_file(PROJECT_ID, relative, digest)
    assert "missing" in str(info.value)


def test_delete_asset_file_handles_read_only_and_missing(isolated_workspace):
    relative, _ = storage.write_asset_file(PROJECT_ID, "e" * 12, ".png", png_bytes())
    assert storage.delete_asset_file(PROJECT_ID, relative) is True
    assert not (isolated_workspace / PROJECT_ID / relative).exists()
    assert storage.delete_asset_file(PROJECT_ID, relative) is False
    assert storage.delete_asset_file(PROJECT_ID, "../../escape.png") is False


@pytest.mark.parametrize("bad_id", ["../evil", "x" * 40, "", "ABCDEF123456", "short"])
def test_asset_ids_are_validated_before_touching_the_disk(bad_id):
    with pytest.raises(ValueError):
        storage.asset_relative_path(bad_id, ".png")


def test_write_asset_file_refuses_non_image_suffixes():
    with pytest.raises(storage.StorageError):
        storage.write_asset_file(PROJECT_ID, "f" * 12, ".exe", b"MZ")


def test_asset_kind_appears_in_the_manifest(isolated_workspace):
    from app.models import Project
    project = Project(id=PROJECT_ID, title="m")
    storage.save_project(project)
    relative, _ = storage.write_asset_file(PROJECT_ID, "a" * 12, ".png", png_bytes())
    manifest = storage.write_manifest(PROJECT_ID)
    entry = next(f for f in manifest["files"] if f["path"] == relative)
    assert entry["kind"] == "reference_asset" and entry["read_only"] is True
