"""Local hardware and model-runtime diagnostics.

Every probe is optional and isolated: a missing tool, library or permission yields
``"unknown"`` plus a note, never an exception. Probes only read local information:
OS APIs, the Windows registry, ``/proc``, binaries already on PATH (run with an argument
list, no shell, and a timeout), and localhost-only ComfyUI/Ollama endpoints.
Nothing is downloaded or installed.
"""
from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from app import storage
from app.model_catalog import recommend
from app.services import _get_json, is_local_url, service_urls

UNKNOWN = "unknown"
GIB = 1024 ** 3
CMD_TIMEOUT = 6.0
CACHE_SECONDS = 30.0

_cache: dict[str, Any] = {"at": 0.0, "report": None}
_cache_lock = threading.Lock()

# Integrated/basic adapters that local diffusion/video models cannot realistically use.
_INTEGRATED_RE = re.compile(
    r"\b(UHD|Iris|HD Graphics|Microsoft Basic|Radeon\(TM\) Graphics|Radeon Graphics|Vega \d+ Graphics|"
    r"llvmpipe|SVGA|VirtualBox|Hyper-V|Parallels|Mali|Adreno)\b", re.IGNORECASE)


def _gb(num_bytes: float | int | None) -> float | None:
    return None if num_bytes is None else round(num_bytes / GIB, 1)


def run_tool(name: str, args: list[str], timeout: float = CMD_TIMEOUT) -> tuple[str | None, str | None]:
    """Run a binary from PATH with fixed arguments. Returns (stdout, error)."""
    exe = shutil.which(name)
    if not exe:
        return None, f"{name} not found on PATH"
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        done = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, check=False, **kwargs)
    except subprocess.TimeoutExpired:
        return None, f"{name} timed out after {timeout:g}s"
    except OSError as exc:
        return None, f"{name} could not run: {exc}"
    if done.returncode != 0:
        return None, f"{name} exited with code {done.returncode}: {(done.stderr or '').strip()[:200]}"
    return done.stdout, None


def _safe(probe: Callable[[], dict[str, Any]], fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        return probe()
    except Exception as exc:  # a diagnostic must never take the screen down
        return {**fallback, "error": f"{type(exc).__name__}: {exc}"[:300]}


# --------------------------------------------------------------------------- OS / Python / CPU / disk

def probe_os() -> dict[str, Any]:
    info: dict[str, Any] = {
        "system": platform.system() or UNKNOWN,
        "release": platform.release() or UNKNOWN,
        "version": platform.version() or UNKNOWN,
        "machine": platform.machine() or UNKNOWN,
        "platform": platform.platform() or UNKNOWN,
    }
    if info["system"] == "Darwin":
        info["release"] = platform.mac_ver()[0] or info["release"]
    elif info["system"] == "Windows":
        edition = getattr(platform, "win32_edition", lambda: None)()
        if edition:
            info["edition"] = edition
    return info


def probe_python() -> dict[str, Any]:
    v = sys.version_info
    supported = (3, 11) <= (v.major, v.minor) <= (3, 13)
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "bits": 64 if sys.maxsize > 2 ** 32 else 32,
        "note": None if supported else "README recommends Python 3.11 or 3.12; other versions are untested.",
    }


def probe_cpu() -> dict[str, Any]:
    name = platform.processor() or UNKNOWN
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return {"name": name, "logical_cores": os.cpu_count() or UNKNOWN}


def probe_disk() -> dict[str, Any]:
    path = storage.PROJECTS_DIR
    usage = shutil.disk_usage(path)
    return {"path": str(path), "total_gb": _gb(usage.total), "free_gb": _gb(usage.free),
            "used_percent": round(usage.used / usage.total * 100, 1) if usage.total else None}


# --------------------------------------------------------------------------- RAM

def _ram_windows() -> tuple[int, int] | None:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        return None
    return status.ullTotalPhys, status.ullAvailPhys


def _ram_linux() -> tuple[int, int | None] | None:
    values: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0]) * 1024
    if "MemTotal" not in values:
        return None
    return values["MemTotal"], values.get("MemAvailable")


def _ram_macos() -> tuple[int, None] | None:
    out, _ = run_tool("sysctl", ["-n", "hw.memsize"])
    return (int(out.strip()), None) if out and out.strip().isdigit() else None


def probe_ram() -> dict[str, Any]:
    system = platform.system()
    result: tuple[int, int | None] | None = None
    source = None
    try:
        import psutil  # optional; used only if already installed
        vm = psutil.virtual_memory()
        result, source = (vm.total, vm.available), "psutil"
    except Exception:
        pass
    if result is None:
        try:
            if system == "Windows":
                result, source = _ram_windows(), "GlobalMemoryStatusEx"
            elif system == "Linux":
                result, source = _ram_linux(), "/proc/meminfo"
            elif system == "Darwin":
                result, source = _ram_macos(), "sysctl hw.memsize"
        except Exception:
            result = None
    if not result:
        return {"total_gb": None, "available_gb": None, "source": None,
                "note": "RAM could not be detected safely on this system."}
    return {"total_gb": _gb(result[0]), "available_gb": _gb(result[1]), "source": source}


# --------------------------------------------------------------------------- GPU

def _gpu_entry(name: str, vendor: str, vram_gb: float | None, source: str, **extra: Any) -> dict[str, Any]:
    integrated = bool(_INTEGRATED_RE.search(name))
    return {"name": name.strip(), "vendor": vendor, "vram_gb": vram_gb, "source": source,
            "integrated": integrated, **extra}


def _vendor(name: str) -> str:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "quadro" in n or "rtx" in n or "tesla" in n:
        return "NVIDIA"
    if "amd" in n or "radeon" in n or "ati " in n:
        return "AMD"
    if "intel" in n:
        return "Intel"
    if "apple" in n:
        return "Apple"
    return UNKNOWN


def _gpus_nvidia_smi(notes: list[str]) -> list[dict[str, Any]]:
    out, err = run_tool("nvidia-smi", ["--query-gpu=name,memory.total,memory.free,driver_version",
                                       "--format=csv,noheader,nounits"])
    if out is None:
        notes.append(f"NVIDIA: {err}.")
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        def mib(value: str) -> float | None:
            return round(float(value) / 1024, 1) if re.fullmatch(r"\d+(\.\d+)?", value) else None
        gpus.append(_gpu_entry(parts[0], "NVIDIA", mib(parts[1]), "nvidia-smi", vram_free_gb=mib(parts[2]),
                               driver=parts[3], compute="cuda"))
    return gpus


def _nvidia_probe() -> tuple[list[dict[str, Any]], list[str]]:
    notes: list[str] = []
    try:
        return _gpus_nvidia_smi(notes), notes
    except Exception as exc:
        return [], [f"NVIDIA: probe failed ({type(exc).__name__})."]


def _gpus_windows_registry(notes: list[str]) -> list[dict[str, Any]]:
    """Display adapters from the registry. qwMemorySize is 64-bit, unlike WMI's AdapterRAM (capped at 4 GB)."""
    import winreg  # Windows only
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    gpus: list[dict[str, Any]] = []
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
        index = 0
        while True:
            try:
                sub = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(root, sub) as key:
                    name = str(winreg.QueryValueEx(key, "DriverDesc")[0])
                    vram = None
                    for value_name in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
                        try:
                            raw = winreg.QueryValueEx(key, value_name)[0]
                        except OSError:
                            continue
                        if isinstance(raw, bytes):
                            raw = int.from_bytes(raw[:8], "little")
                        if isinstance(raw, int) and raw > 0:
                            vram = _gb(raw)
                            break
                    try:
                        driver = str(winreg.QueryValueEx(key, "DriverVersion")[0])
                    except OSError:
                        driver = None
            except OSError:
                continue
            gpus.append(_gpu_entry(name, _vendor(name), vram, "Windows registry", driver=driver,
                                   vram_note="dedicated memory reported by the driver"))
    if not gpus:
        notes.append("Windows: no display adapters found in the registry.")
    return gpus


def _gpus_linux(notes: list[str]) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    out, err = run_tool("rocm-smi", ["--showproductname", "--showmeminfo", "vram", "--json"])
    if out:
        import json
        try:
            for card in json.loads(out).values():
                name = card.get("Card series") or card.get("Card model") or "AMD GPU"
                total = card.get("VRAM Total Memory (B)")
                gpus.append(_gpu_entry(str(name), "AMD", _gb(int(total)) if str(total).isdigit() else None,
                                       "rocm-smi", compute="rocm"))
        except (ValueError, AttributeError):
            notes.append("AMD: rocm-smi output could not be parsed.")
    else:
        notes.append(f"AMD: {err}.")
    if not gpus:
        out, err = run_tool("lspci", [])
        if out:
            for line in out.splitlines():
                if re.search(r"VGA compatible controller|3D controller|Display controller", line):
                    name = line.split(":", 2)[-1].strip()
                    gpus.append(_gpu_entry(name, _vendor(name), None, "lspci"))
        else:
            notes.append(f"lspci: {err}.")
    return gpus


def _gpus_macos(notes: list[str], ram_gb: float | None) -> list[dict[str, Any]]:
    import json
    out, err = run_tool("system_profiler", ["SPDisplaysDataType", "-json"], timeout=10)
    if not out:
        notes.append(f"macOS: {err}.")
        return []
    gpus = []
    for item in json.loads(out).get("SPDisplaysDataType", []):
        name = item.get("sppci_model") or item.get("_name") or "GPU"
        vram_text = item.get("spdisplays_vram") or item.get("spdisplays_vram_shared") or ""
        match = re.search(r"(\d+)\s*(GB|MB)", vram_text)
        vram = (float(match.group(1)) / (1 if match.group(2) == "GB" else 1024)) if match else None
        apple = "apple" in name.lower()
        if apple and vram is None and ram_gb:
            vram = round(ram_gb * 0.66, 1)  # unified memory; macOS lets the GPU use roughly two thirds by default
        gpus.append(_gpu_entry(name, "Apple" if apple else _vendor(name), vram, "system_profiler",
                               compute="mps" if apple else None,
                               vram_note="estimated share of unified memory" if apple else None))
    return gpus


def _gpus_from_comfy(stats: dict[str, Any] | None) -> list[dict[str, Any]]:
    gpus = []
    for dev in (stats or {}).get("devices", []) or []:
        kind = str(dev.get("type", "")).lower()
        if kind == "cpu":
            continue
        name = str(dev.get("name", "GPU"))
        name = re.sub(r"^cuda:\d+\s*", "", name).split(" : ")[0]
        compute = "cuda" if kind == "cuda" else ("mps" if kind == "mps" else kind or None)
        gpus.append(_gpu_entry(name, _vendor(name), _gb(dev.get("vram_total")), "ComfyUI /system_stats",
                               vram_free_gb=_gb(dev.get("vram_free")), compute=compute))
    return gpus


def probe_gpus(ram_gb: float | None, comfy_stats: dict[str, Any] | None,
               nvidia: tuple[list[dict[str, Any]], list[str]] | None = None) -> dict[str, Any]:
    notes: list[str] = []
    system = platform.system()
    if nvidia is None:
        found = _gpus_nvidia_smi(notes)
    else:
        found, nvidia_notes = list(nvidia[0]), nvidia[1]
        notes.extend(nvidia_notes)
    found += _gpus_from_comfy(comfy_stats)
    try:
        if system == "Windows":
            found += _gpus_windows_registry(notes)
        elif system == "Linux":
            found += _gpus_linux(notes)
        elif system == "Darwin":
            found += _gpus_macos(notes, ram_gb)
    except Exception as exc:
        notes.append(f"OS GPU listing failed: {type(exc).__name__}.")

    # Merge duplicates (same adapter from several sources); prefer entries with measured VRAM.
    priority = {"nvidia-smi": 0, "ComfyUI /system_stats": 1, "rocm-smi": 2, "system_profiler": 3,
                "Windows registry": 4, "lspci": 5}
    merged: dict[str, dict[str, Any]] = {}
    for gpu in sorted(found, key=lambda g: priority.get(g["source"], 9)):
        key = re.sub(r"[^a-z0-9]", "", gpu["name"].lower().replace("nvidia", "").replace("geforce", ""))
        if key in merged:
            merged[key].setdefault("also_seen_by", []).append(gpu["source"])
            if merged[key].get("vram_gb") is None:
                merged[key]["vram_gb"] = gpu.get("vram_gb")
            continue
        merged[key] = dict(gpu)
    gpus = list(merged.values())
    if not gpus:
        notes.append("No GPU information available; showing 'unknown'.")

    usable = [g for g in gpus if not g["integrated"] and g.get("vram_gb")]
    best = max(usable, key=lambda g: g["vram_gb"], default=None)
    if best is not None:
        if best.get("compute") or best["vendor"] == "NVIDIA":
            kind = best.get("compute") or "cuda"
        elif best["vendor"] == "AMD":
            kind = "rocm"
        else:
            kind = None
        if best["vendor"] == "NVIDIA" and best["source"] not in ("nvidia-smi", "ComfyUI /system_stats"):
            notes.append("NVIDIA GPU listed by the OS but nvidia-smi was not found; CUDA readiness is unverified.")
    elif gpus and all(g["integrated"] for g in gpus):
        kind = "none"
        notes.append("Only integrated/basic graphics found; GPU image/video models are not expected to work.")
    else:
        kind = None
    return {"devices": gpus, "best_vram_gb": best["vram_gb"] if best else None,
            "best_device": best["name"] if best else None, "compute": kind, "notes": notes}


# --------------------------------------------------------------------------- FFmpeg

def probe_ffmpeg() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tool in ("ffmpeg", "ffprobe"):
        out, err = run_tool(tool, ["-hide_banner", "-version"] if tool == "ffprobe" else ["-version"])
        if out is None:
            result[tool] = {"available": False, "version": None, "path": shutil.which(tool), "error": err}
            continue
        first = out.splitlines()[0] if out else ""
        match = re.search(r"version\s+(\S+)", first)
        entry: dict[str, Any] = {"available": True, "version": match.group(1) if match else first[:80],
                                 "path": shutil.which(tool)}
        if tool == "ffmpeg":
            config = next((l for l in out.splitlines() if l.startswith("configuration:")), "")
            entry["build_license"] = ("GPL (built with --enable-gpl)" if "--enable-gpl" in config
                                      else "LGPL (no --enable-gpl)" if config else UNKNOWN)
            entry["nonfree"] = "--enable-nonfree" in config
        result[tool] = entry
    result["available"] = bool(result["ffmpeg"]["available"])
    if not result["available"]:
        result["hint"] = ("Install FFmpeg yourself (https://ffmpeg.org/download.html) and make sure 'ffmpeg' is on "
                          "PATH. It is only needed for rendering in later phases.")
    result["license_info"] = "https://ffmpeg.org/legal.html"
    return result


# --------------------------------------------------------------------------- ComfyUI / Ollama

def probe_comfyui(url: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    info: dict[str, Any] = {"url": url, "local": is_local_url(url), "available": False}
    if not info["local"]:
        info["note"] = "COMFYUI_URL is not a localhost address, so it was not contacted."
        return info, None
    ok, stats = _get_json(f"{url}/system_stats", timeout=2.0)
    if not ok or not isinstance(stats, dict):
        info["note"] = "Not running (or not reachable). Start ComfyUI to enable image/video work in later phases."
        return info, None
    system = stats.get("system", {}) if isinstance(stats.get("system"), dict) else {}
    info.update({
        "available": True,
        "version": system.get("comfyui_version") or UNKNOWN,
        "python_version": (system.get("python_version") or UNKNOWN).split(" ")[0],
        "pytorch_version": system.get("pytorch_version") or UNKNOWN,
        "devices": [{"name": d.get("name"), "type": d.get("type"), "vram_total_gb": _gb(d.get("vram_total")),
                     "vram_free_gb": _gb(d.get("vram_free"))} for d in stats.get("devices", []) or []],
    })
    ok, checkpoints = _get_json(f"{url}/models/checkpoints", timeout=2.0)
    info["checkpoints"] = [str(c) for c in checkpoints][:200] if ok and isinstance(checkpoints, list) else None
    ok, diffusion = _get_json(f"{url}/models/diffusion_models", timeout=2.0)
    info["diffusion_models"] = [str(c) for c in diffusion][:200] if ok and isinstance(diffusion, list) else None
    return info, stats


def probe_ollama(url: str) -> dict[str, Any]:
    info: dict[str, Any] = {"url": url, "local": is_local_url(url), "available": False, "models": []}
    if not info["local"]:
        info["note"] = "OLLAMA_URL is not a localhost address, so it was not contacted."
        return info
    ok, version = _get_json(f"{url}/api/version", timeout=2.0)
    ok_tags, tags = _get_json(f"{url}/api/tags", timeout=2.0)
    if not (ok or ok_tags):
        info["note"] = "Not running (or not reachable). The app works fully without Ollama."
        return info
    info["available"] = True
    info["version"] = version.get("version", UNKNOWN) if isinstance(version, dict) else UNKNOWN
    if isinstance(tags, dict):
        for model in tags.get("models", []) or []:
            details = model.get("details") or {}
            info["models"].append({
                "name": model.get("name", UNKNOWN), "size_gb": _gb(model.get("size")),
                "parameter_size": details.get("parameter_size"), "quantization": details.get("quantization_level"),
                "family": details.get("family"),
            })
    if not info["models"]:
        info["note"] = "Ollama is running but has no local models."
    return info


# --------------------------------------------------------------------------- report

def build_report() -> dict[str, Any]:
    started = time.perf_counter()
    comfy_url, ollama_url = service_urls()

    def comfy_probe() -> tuple[dict[str, Any], dict[str, Any] | None]:
        try:
            return probe_comfyui(comfy_url)
        except Exception as exc:
            return {"url": comfy_url, "available": False, "error": f"{type(exc).__name__}: {exc}"[:300]}, None

    # Independent probes run concurrently; each has its own timeout.
    with ThreadPoolExecutor(max_workers=6) as pool:
        f_comfy = pool.submit(comfy_probe)
        f_ollama = pool.submit(_safe, lambda: probe_ollama(ollama_url),
                               {"url": ollama_url, "available": False, "models": []})
        f_ffmpeg = pool.submit(_safe, probe_ffmpeg, {"available": False, "ffmpeg": {"available": False}})
        f_ram = pool.submit(_safe, probe_ram, {"total_gb": None, "available_gb": None, "source": None})
        f_nvidia = pool.submit(_nvidia_probe)
        comfy, comfy_stats = f_comfy.result()
        ram = f_ram.result()
        nvidia = f_nvidia.result()
        report: dict[str, Any] = {
            "os": _safe(probe_os, {"system": UNKNOWN}),
            "python": _safe(probe_python, {"version": UNKNOWN}),
            "cpu": _safe(probe_cpu, {"name": UNKNOWN, "logical_cores": UNKNOWN}),
            "ram": ram,
            "disk": _safe(probe_disk, {"path": str(storage.PROJECTS_DIR), "total_gb": None, "free_gb": None}),
            "gpu": _safe(lambda: probe_gpus(ram.get("total_gb"), comfy_stats, nvidia),
                         {"devices": [], "best_vram_gb": None, "compute": None, "notes": []}),
            "ffmpeg": f_ffmpeg.result(),
            "comfyui": comfy,
            "ollama": f_ollama.result(),
        }
    hw = {"vram_gb": report["gpu"].get("best_vram_gb"), "gpu_kind": report["gpu"].get("compute"),
          "ram_gb": ram.get("total_gb"), "disk_free_gb": report["disk"].get("free_gb")}
    if hw["gpu_kind"] == "none":
        hw["vram_gb"] = 0.0
    comfy_files = (report["comfyui"].get("checkpoints") or []) + (report["comfyui"].get("diffusion_models") or [])
    report["recommendations"] = recommend(hw, [m["name"] for m in report["ollama"].get("models", [])], comfy_files)
    report["summary_inputs"] = hw
    report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
    report["privacy"] = "Collected locally. Nothing is sent anywhere; only localhost services are contacted."
    return report


def get_report(refresh: bool = False) -> dict[str, Any]:
    with _cache_lock:
        if not refresh and _cache["report"] is not None and time.monotonic() - _cache["at"] < CACHE_SECONDS:
            return {**_cache["report"], "cached": True}
        report = build_report()
        _cache.update(at=time.monotonic(), report=report)
        return {**report, "cached": False}
