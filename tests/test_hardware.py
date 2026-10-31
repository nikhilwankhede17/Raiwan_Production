"""Hardware & Models diagnostics. All external tools and services are mocked; nothing is downloaded."""
import sys
import urllib.request

import pytest
from fastapi.testclient import TestClient

from app import hardware, services
from app.main import app
from app.model_catalog import CATALOG, CATEGORY_LABELS, recommend

NVIDIA_24GB = "NVIDIA GeForce RTX 4090, 24564, 23000, 560.94\n"
FFMPEG_OUT = ("ffmpeg version 7.1-full_build Copyright (c) 2000-2024 the FFmpeg developers\n"
              "configuration: --enable-gpl --enable-version3 --enable-libx264\n")
COMFY_STATS = {
    "system": {"comfyui_version": "0.3.60", "python_version": "3.12.7 (main)", "pytorch_version": "2.7.0+cu128"},
    "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync", "type": "cuda",
                 "vram_total": 25_757_220_864, "vram_free": 24_000_000_000}],
}


@pytest.fixture(autouse=True)
def fresh_cache():
    hardware._cache.update(at=0.0, report=None)
    yield
    hardware._cache.update(at=0.0, report=None)


@pytest.fixture
def bare_machine(monkeypatch):
    """No tools on PATH, no psutil, no GPU listing, no local services."""
    monkeypatch.setattr(hardware, "run_tool", lambda name, args, timeout=6.0: (None, f"{name} not found on PATH"))
    monkeypatch.setattr(hardware, "_get_json", lambda url, timeout=1.5: (False, None))
    monkeypatch.setitem(sys.modules, "psutil", None)  # import psutil -> ImportError
    for fn in ("_ram_windows", "_ram_linux", "_ram_macos"):
        monkeypatch.setattr(hardware, fn, lambda: (_ for _ in ()).throw(OSError("denied")))
    for fn in ("_gpus_windows_registry", "_gpus_linux"):
        monkeypatch.setattr(hardware, fn, lambda notes: (_ for _ in ()).throw(OSError("no access")))
    monkeypatch.setattr(hardware, "_gpus_macos", lambda notes, ram: (_ for _ in ()).throw(OSError("no access")))


def fake_tools(outputs):
    def run(name, args, timeout=6.0):
        return (outputs[name], None) if name in outputs else (None, f"{name} not found on PATH")
    return run


# --------------------------------------------------------------------------- graceful degradation

def test_report_works_with_nothing_available(bare_machine):
    r = hardware.build_report()
    assert r["os"]["system"] and r["python"]["version"]
    assert r["ram"]["total_gb"] is None and "could not be detected" in r["ram"]["note"]
    assert r["gpu"]["devices"] == [] and r["gpu"]["best_vram_gb"] is None and r["gpu"]["compute"] is None
    assert any("nvidia-smi not found" in n for n in r["gpu"]["notes"])
    assert r["ffmpeg"]["available"] is False and "ffmpeg.org" in r["ffmpeg"]["hint"]
    assert r["comfyui"]["available"] is False and r["ollama"]["available"] is False
    assert r["disk"]["free_gb"] is not None


def test_unknown_gpu_never_recommends_gpu_models(bare_machine, monkeypatch):
    monkeypatch.setattr(hardware, "probe_disk", lambda: {"path": "x", "total_gb": 1000.0, "free_gb": 500.0})
    recs = hardware.build_report()["recommendations"]
    picked = {m["id"] for m in recs["recommended"]}
    assert not picked & {"sdxl-base-1.0", "flux1-schnell", "wan2.2-ti2v-5b", "qwen3-8b"}
    wan = next(m for m in recs["other_candidates"] if m["id"] == "wan2.2-ti2v-5b")
    assert wan["status"] == "unknown" and "could not be detected" in wan["reason"]
    # RAM unknown too, so CPU picks are flagged "possible", not "recommended".
    assert all(m["status"] == "possible" for m in recs["recommended"])


def test_probe_exception_is_contained(bare_machine, monkeypatch):
    monkeypatch.setattr(hardware, "probe_cpu", lambda: 1 / 0)
    monkeypatch.setattr(hardware, "probe_ffmpeg", lambda: 1 / 0)
    r = hardware.build_report()
    assert r["cpu"]["name"] == "unknown" and "ZeroDivisionError" in r["cpu"]["error"]
    assert r["ffmpeg"]["available"] is False and "error" in r["ffmpeg"]


def test_run_tool_timeout_and_missing_binary():
    out, err = hardware.run_tool(sys.executable, ["-c", "import time; time.sleep(5)"], timeout=0.5)
    assert out is None and "timed out" in err
    out, err = hardware.run_tool("definitely-not-a-real-tool-xyz", ["--version"])
    assert out is None and "not found" in err
    out, err = hardware.run_tool(sys.executable, ["-c", "print('hi')"])
    assert out.strip() == "hi" and err is None


# --------------------------------------------------------------------------- detection paths

def test_nvidia_smi_parsing_and_high_end_recommendations(bare_machine, monkeypatch):
    monkeypatch.setattr(hardware, "run_tool", fake_tools({"nvidia-smi": NVIDIA_24GB, "ffmpeg": FFMPEG_OUT,
                                                          "ffprobe": FFMPEG_OUT.replace("ffmpeg", "ffprobe", 1)}))
    monkeypatch.setattr(hardware, "probe_ram", lambda: {"total_gb": 64.0, "available_gb": 40.0, "source": "test"})
    monkeypatch.setattr(hardware, "probe_disk", lambda: {"path": "x", "total_gb": 2000.0, "free_gb": 900.0})
    r = hardware.build_report()
    gpu = r["gpu"]["devices"][0]
    assert gpu["name"] == "NVIDIA GeForce RTX 4090" and gpu["vram_gb"] == 24.0 and gpu["driver"] == "560.94"
    assert r["gpu"]["compute"] == "cuda" and r["gpu"]["best_vram_gb"] == 24.0
    assert r["ffmpeg"]["available"] and r["ffmpeg"]["ffmpeg"]["version"] == "7.1-full_build"
    assert r["ffmpeg"]["ffmpeg"]["build_license"].startswith("GPL")
    picked = {m["category"]: m["id"] for m in r["recommendations"]["recommended"]}
    assert picked == {"llm": "qwen3-8b", "image": "flux1-schnell", "video": "wan2.2-ti2v-5b",
                      "tts": "kokoro-82m", "asr": "whisper-large-v3-turbo"}


def test_comfyui_stats_supply_gpu_and_installed_models(bare_machine, monkeypatch):
    def fake_json(url, timeout=1.5):
        if url.endswith("/system_stats"):
            return True, COMFY_STATS
        if url.endswith("/models/checkpoints"):
            return True, ["sd_xl_base_1.0.safetensors"]
        if url.endswith("/models/diffusion_models"):
            return True, ["wan2.2_ti2v_5B_fp16.safetensors"]
        if url.endswith("/api/version"):
            return True, {"version": "0.12.0"}
        if url.endswith("/api/tags"):
            return True, {"models": [{"name": "qwen3:8b", "size": 5_200_000_000,
                                      "details": {"parameter_size": "8.2B", "quantization_level": "Q4_K_M"}}]}
        return False, None
    monkeypatch.setattr(hardware, "_get_json", fake_json)
    monkeypatch.setattr(hardware, "probe_ram", lambda: {"total_gb": 32.0, "available_gb": 20.0, "source": "test"})
    monkeypatch.setattr(hardware, "probe_disk", lambda: {"path": "x", "total_gb": 1000.0, "free_gb": 500.0})
    r = hardware.build_report()
    assert r["comfyui"]["available"] and r["comfyui"]["version"] == "0.3.60"
    assert r["comfyui"]["python_version"] == "3.12.7"
    assert r["gpu"]["devices"][0]["name"] == "NVIDIA GeForce RTX 4090"
    assert r["gpu"]["devices"][0]["source"] == "ComfyUI /system_stats" and r["gpu"]["compute"] == "cuda"
    assert r["ollama"]["version"] == "0.12.0" and r["ollama"]["models"][0]["parameter_size"] == "8.2B"
    everything = r["recommendations"]["recommended"] + r["recommendations"]["other_candidates"]
    installed = {m["id"] for m in everything if m["installed"]}
    assert installed == {"sdxl-base-1.0", "wan2.2-ti2v-5b", "qwen3-8b"}


def test_gpu_merge_prefers_measured_vram(bare_machine, monkeypatch):
    monkeypatch.setattr(hardware, "run_tool", fake_tools({"nvidia-smi": NVIDIA_24GB}))
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware, "_gpus_windows_registry", lambda notes: [
        hardware._gpu_entry("NVIDIA GeForce RTX 4090", "NVIDIA", 4.0, "Windows registry"),
        hardware._gpu_entry("Intel(R) UHD Graphics 770", "Intel", 0.1, "Windows registry")])
    gpu = hardware.probe_gpus(16.0, None)
    names = [g["name"] for g in gpu["devices"]]
    assert names.count("NVIDIA GeForce RTX 4090") == 1
    nv = gpu["devices"][0]
    assert nv["vram_gb"] == 24.0 and nv["also_seen_by"] == ["Windows registry"]
    intel = next(g for g in gpu["devices"] if "Intel" in g["name"])
    assert intel["integrated"] is True and gpu["best_device"] == "NVIDIA GeForce RTX 4090"


def test_integrated_only_means_no_gpu(bare_machine, monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware, "_gpus_windows_registry",
                        lambda notes: [hardware._gpu_entry("Intel(R) UHD Graphics", "Intel", 1.0, "Windows registry")])
    gpu = hardware.probe_gpus(8.0, None)
    assert gpu["compute"] == "none" and gpu["best_vram_gb"] is None
    assert any("integrated" in n for n in gpu["notes"])


def test_nvidia_listed_without_nvidia_smi_is_flagged(bare_machine, monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware, "_gpus_windows_registry",
                        lambda notes: [hardware._gpu_entry("NVIDIA GeForce RTX 3060", "NVIDIA", 12.0, "Windows registry")])
    gpu = hardware.probe_gpus(16.0, None)
    assert gpu["best_vram_gb"] == 12.0 and gpu["compute"] == "cuda"
    assert any("CUDA readiness is unverified" in n for n in gpu["notes"])


def test_ffmpeg_lgpl_build_detected(bare_machine, monkeypatch):
    lgpl = "ffmpeg version n6.1 Copyright\nconfiguration: --enable-shared\n"
    monkeypatch.setattr(hardware, "run_tool", fake_tools({"ffmpeg": lgpl}))
    ff = hardware.probe_ffmpeg()
    assert ff["available"] and ff["ffmpeg"]["build_license"].startswith("LGPL")
    assert ff["ffprobe"]["available"] is False


# --------------------------------------------------------------------------- localhost-only

@pytest.mark.parametrize("url,local", [
    ("http://127.0.0.1:8188", True), ("http://localhost:11434", True), ("http://[::1]:8188", True),
    ("http://127.5.5.5:1", True), ("http://192.168.1.10:8188", False), ("https://example.com", False),
    ("http://127.0.0.1.evil.com", False), ("file:///etc/passwd", False), ("ftp://127.0.0.1", False), ("", False),
])
def test_is_local_url(url, local):
    assert services.is_local_url(url) is local


def test_non_local_service_urls_are_never_contacted(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setenv("COMFYUI_URL", "https://comfy.example.com")
    monkeypatch.setenv("OLLAMA_URL", "http://10.0.0.5:11434")
    comfy, stats = hardware.probe_comfyui(services.service_urls()[0])
    assert comfy["available"] is False and comfy["local"] is False and stats is None
    oll = hardware.probe_ollama(services.service_urls()[1])
    assert oll["available"] is False and "not a localhost" in oll["note"]
    assert services.check_services()["comfyui"]["available"] is False


# --------------------------------------------------------------------------- recommendation rules

def test_cpu_only_laptop_gets_cpu_models_only():
    recs = recommend({"vram_gb": 0.0, "gpu_kind": "none", "ram_gb": 7.8, "disk_free_gb": 80.0})
    picked = {m["category"]: m["id"] for m in recs["recommended"]}
    assert picked == {"llm": "qwen3-1.7b", "tts": "kokoro-82m", "asr": "whisper-large-v3-turbo"}
    assert all("CPU" in m["reason"] for m in recs["recommended"] if m["id"] != "kokoro-82m")
    sdxl = next(m for m in recs["other_candidates"] if m["id"] == "sdxl-base-1.0")
    assert sdxl["status"] == "not_compatible" and "dedicated GPU" in sdxl["reason"]


def test_mid_range_gpu():
    recs = recommend({"vram_gb": 12.0, "gpu_kind": "cuda", "ram_gb": 32.0, "disk_free_gb": 500.0})
    picked = {m["category"]: m["id"] for m in recs["recommended"]}
    assert picked["image"] == "sdxl-base-1.0" and picked["llm"] == "qwen3-8b" and "video" not in picked
    wan = next(m for m in recs["other_candidates"] if m["id"] == "wan2.2-ti2v-5b")
    assert "24 GB+ VRAM" in wan["reason"]


def test_video_requires_cuda_and_low_disk_excludes():
    recs = recommend({"vram_gb": 32.0, "gpu_kind": "mps", "ram_gb": 48.0, "disk_free_gb": 500.0})
    wan = next(m for m in recs["recommended"] + recs["other_candidates"] if m["id"] == "wan2.2-ti2v-5b")
    assert wan["status"] == "not_compatible" and "CUDA" in wan["reason"]
    tiny_disk = recommend({"vram_gb": 24.0, "gpu_kind": "cuda", "ram_gb": 64.0, "disk_free_gb": 0.05})
    assert tiny_disk["recommended"] == []
    assert all("disk" in m["reason"] for m in tiny_disk["other_candidates"])


def test_catalog_entries_are_complete_and_few():
    assert len(CATALOG) <= 10
    for entry in CATALOG:
        assert entry["category"] in CATEGORY_LABELS
        assert entry["model_card"].startswith("https://") and entry["license_url"].startswith("https://")
        assert entry["license"] and entry["commercial_use"] and entry["install_hint"]
    recs = recommend({"vram_gb": 80.0, "gpu_kind": "cuda", "ram_gb": 256.0, "disk_free_gb": 5000.0})
    assert len(recs["recommended"]) == len(CATEGORY_LABELS)  # at most one per category


# --------------------------------------------------------------------------- API & UI

def test_api_endpoint_and_cache(bare_machine):
    client = TestClient(app)
    first = client.get("/api/diagnostics")
    assert first.status_code == 200 and first.json()["cached"] is False
    assert client.get("/api/diagnostics").json()["cached"] is True
    assert client.get("/api/diagnostics?refresh=true").json()["cached"] is False
    page = client.get("/diagnostics")
    assert page.status_code == 200 and "/api/diagnostics" in page.text and "Official model card" in page.text
    assert 'href="/diagnostics"' in client.get("/").text
