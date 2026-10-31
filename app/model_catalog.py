"""Curated local model candidates and hardware-matching rules.

Every entry links to the official model card and licence. Hardware and size figures are
approximate planning values taken from those pages (checked 2026-10-24); always confirm on the
model card before downloading. Nothing in this module downloads or installs anything.
"""
from __future__ import annotations

from typing import Any

CATALOG_CHECKED = "2026-10-24"
# Installed RAM is reported slightly below its nominal size (an "8 GB" machine shows ~7.8 GB).
RAM_TOLERANCE = 0.94

# min_vram_gb: None = runs on CPU. cpu_ok: usable (possibly slowly) without a supported GPU.
# needs: "cuda" means an NVIDIA CUDA GPU is assumed by the official instructions.
CATALOG: list[dict[str, Any]] = [
    # ------------------------------------------------------------------ local LLM (Ollama)
    {
        "id": "qwen3-1.7b", "category": "llm", "name": "Qwen3 1.7B", "runtime": "Ollama",
        "purpose": "Optional director assistant (breakdown/prompt drafting) on modest machines.",
        "min_vram_gb": None, "rec_vram_gb": 4, "min_ram_gb": 8, "disk_gb": 1.4, "cpu_ok": True, "needs": None,
        "rank": 3, "license": "Apache-2.0", "commercial_use": "Permitted under Apache-2.0.",
        "model_card": "https://huggingface.co/Qwen/Qwen3-1.7B",
        "license_url": "https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/LICENSE",
        "runtime_page": "https://ollama.com/library/qwen3",
        "install_hint": "ollama pull qwen3:1.7b", "ollama_name": "qwen3:1.7b",
        "notes": "Size is the Ollama q4_K_M download (1.4 GB).",
    },
    {
        "id": "qwen3-4b", "category": "llm", "name": "Qwen3 4B", "runtime": "Ollama",
        "purpose": "Better drafting quality; fine on CPU with 16 GB RAM, faster on a 6 GB+ GPU.",
        "min_vram_gb": None, "rec_vram_gb": 6, "min_ram_gb": 16, "disk_gb": 2.5, "cpu_ok": True, "needs": None,
        "rank": 2, "license": "Apache-2.0", "commercial_use": "Permitted under Apache-2.0.",
        "model_card": "https://huggingface.co/Qwen/Qwen3-4B",
        "license_url": "https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE",
        "runtime_page": "https://ollama.com/library/qwen3",
        "install_hint": "ollama pull qwen3:4b", "ollama_name": "qwen3:4b",
        "notes": "Size is the Ollama default 4b tag (2.5 GB).",
    },
    {
        "id": "qwen3-8b", "category": "llm", "name": "Qwen3 8B", "runtime": "Ollama",
        "purpose": "Strongest of these for structured JSON drafting; wants an 8 GB+ GPU.",
        "min_vram_gb": 8, "rec_vram_gb": 8, "min_ram_gb": 16, "disk_gb": 5.2, "cpu_ok": False, "needs": None,
        "rank": 1, "license": "Apache-2.0", "commercial_use": "Permitted under Apache-2.0.",
        "model_card": "https://huggingface.co/Qwen/Qwen3-8B",
        "license_url": "https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE",
        "runtime_page": "https://ollama.com/library/qwen3",
        "install_hint": "ollama pull qwen3:8b", "ollama_name": "qwen3:8b",
        "notes": "Runs on CPU too, but slowly; recommended only with a GPU. Size is the q4_K_M download (5.2 GB).",
    },
    # ------------------------------------------------------------------ image (ComfyUI)
    {
        "id": "sdxl-base-1.0", "category": "image", "name": "Stable Diffusion XL base 1.0", "runtime": "ComfyUI",
        "purpose": "Storyboard frames and character/location reference stills on 8 GB-class GPUs.",
        "min_vram_gb": 8, "rec_vram_gb": 12, "min_ram_gb": 16, "disk_gb": 7, "cpu_ok": False, "needs": None,
        "rank": 2, "license": "CreativeML Open RAIL++-M",
        "commercial_use": "Permitted, subject to the licence's use-based restrictions (Attachment A).",
        "model_card": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        "license_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md",
        "runtime_page": "https://github.com/comfyanonymous/ComfyUI",
        "install_hint": "Place the checkpoint in ComfyUI/models/checkpoints/ yourself.",
        "comfy_match": ["sd_xl_base", "sdxl"],
        "notes": "8 GB is a practical floor for 1024px in ComfyUI; lower VRAM needs --lowvram and is slow.",
    },
    {
        "id": "flux1-schnell", "category": "image", "name": "FLUX.1 [schnell]", "runtime": "ComfyUI",
        "purpose": "Higher prompt adherence in 1-4 steps; needs a 16 GB+ GPU (fp8 builds).",
        "min_vram_gb": 16, "rec_vram_gb": 24, "min_ram_gb": 32, "disk_gb": 34, "cpu_ok": False, "needs": None,
        "rank": 1, "license": "Apache-2.0", "commercial_use": "Permitted under Apache-2.0 (per model card).",
        "model_card": "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        "license_url": "https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-schnell",
        "runtime_page": "https://github.com/comfyanonymous/ComfyUI",
        "install_hint": "Download weights manually per the model card; map them in a ComfyUI workflow.",
        "comfy_match": ["flux1-schnell", "flux1_schnell"],
        "notes": "12B parameters. Disk is ~34 GB at full precision including text encoders; fp8 is roughly half. "
                 "Do not confuse with FLUX.1 [dev], which has a non-commercial licence.",
    },
    # ------------------------------------------------------------------ video (ComfyUI / Python)
    {
        "id": "wan2.2-ti2v-5b", "category": "video", "name": "Wan2.2 TI2V-5B", "runtime": "ComfyUI or Python",
        "purpose": "Short text/image-to-video shots at 720p/24fps from approved stills.",
        "min_vram_gb": 24, "rec_vram_gb": 24, "min_ram_gb": 32, "disk_gb": 35, "cpu_ok": False, "needs": "cuda",
        "rank": 1, "license": "Apache-2.0",
        "commercial_use": "Permitted under Apache-2.0; the authors claim no rights over generated content.",
        "model_card": "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B",
        "license_url": "https://github.com/Wan-Video/Wan2.2/blob/main/LICENSE.txt",
        "runtime_page": "https://github.com/Wan-Video/Wan2.2",
        "install_hint": "Follow the official repository; do not start until Phase 5.",
        "comfy_match": ["wan2.2_ti2v_5b", "wan2.2-ti2v-5b"],
        "notes": "Official README: needs >= 24 GB VRAM with offloading (e.g. RTX 4090); a 5 s 720p clip takes "
                 "under 9 minutes there. Community low-VRAM workflows exist but are not verified here. "
                 "Disk size is an estimate including the text encoder and VAE.",
    },
    # ------------------------------------------------------------------ TTS
    {
        "id": "kokoro-82m", "category": "tts", "name": "Kokoro-82M", "runtime": "Python (kokoro package)",
        "purpose": "Fast local dialogue TTS; runs on CPU.",
        "min_vram_gb": None, "rec_vram_gb": None, "min_ram_gb": 4, "disk_gb": 0.4, "cpu_ok": True, "needs": None,
        "rank": 1, "license": "Apache-2.0 (weights and code)", "commercial_use": "Permitted under Apache-2.0.",
        "model_card": "https://huggingface.co/hexgrad/Kokoro-82M",
        "license_url": "https://github.com/hexgrad/kokoro/blob/main/LICENSE",
        "runtime_page": "https://github.com/hexgrad/kokoro",
        "install_hint": "pip install kokoro (also needs espeak-ng); do not start until Phase 6.",
        "notes": "Some training data is CC BY; see the model card's attribution section. Fictional preset "
                 "voices only — no voice cloning.",
    },
    {
        "id": "piper", "category": "tts", "name": "Piper", "runtime": "Python (piper-tts)",
        "purpose": "Very light CPU TTS with many languages; useful as a fallback voice engine.",
        "min_vram_gb": None, "rec_vram_gb": None, "min_ram_gb": 2, "disk_gb": 0.1, "cpu_ok": True, "needs": None,
        "rank": 2, "license": "GPL-3.0 (engine); each voice has its own licence",
        "commercial_use": "Engine is GPL-3.0 (copyleft obligations if you distribute it). Check each voice's licence.",
        "model_card": "https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md",
        "license_url": "https://github.com/OHF-Voice/piper1-gpl/blob/main/COPYING",
        "runtime_page": "https://github.com/OHF-Voice/piper1-gpl",
        "install_hint": "pip install piper-tts; download one voice manually and record its licence.",
        "notes": "Development moved from rhasspy/piper (MIT, archived) to OHF-Voice/piper1-gpl (GPL-3.0).",
    },
    # ------------------------------------------------------------------ speech recognition (subtitle alignment)
    {
        "id": "whisper-large-v3-turbo", "category": "asr", "name": "Whisper large-v3-turbo", "runtime": "Python",
        "purpose": "Transcribe generated dialogue to check lines and align subtitles.",
        "min_vram_gb": None, "rec_vram_gb": 6, "min_ram_gb": 8, "disk_gb": 1.6, "cpu_ok": True, "needs": None,
        "rank": 1, "license": "MIT", "commercial_use": "Permitted under MIT.",
        "model_card": "https://huggingface.co/openai/whisper-large-v3-turbo",
        "license_url": "https://github.com/openai/whisper/blob/main/LICENSE",
        "runtime_page": "https://github.com/openai/whisper",
        "install_hint": "Install later (Phase 6/7) with the Whisper or transformers package.",
        "notes": "Works on CPU but is much faster with a GPU.",
    },
]

CATEGORY_LABELS = {
    "llm": "Local LLM assistant", "image": "Image / storyboard", "video": "Video generation",
    "tts": "Dialogue (TTS)", "asr": "Transcription / subtitles",
}


def _evaluate(entry: dict[str, Any], hw: dict[str, Any]) -> tuple[str, str]:
    """Return (status, reason). status: recommended | possible | not_compatible | unknown."""
    vram, ram, disk = hw.get("vram_gb"), hw.get("ram_gb"), hw.get("disk_free_gb")
    gpu_kind = hw.get("gpu_kind")  # "cuda" | "rocm" | "mps" | "none" | None (unknown)
    reasons: list[str] = []
    marginal = False

    # Checked in order of importance: GPU first, then RAM, then disk.
    needs_gpu = entry["min_vram_gb"] is not None
    if needs_gpu:
        if vram is None:
            return "unknown", ("GPU VRAM could not be detected. Install NVIDIA drivers (nvidia-smi) or start "
                               f"ComfyUI so its GPU can be read. Needs {entry['min_vram_gb']} GB+ VRAM.")
        if gpu_kind == "none":
            return "not_compatible", f"Needs a dedicated GPU with {entry['min_vram_gb']} GB+ VRAM; none detected."
        if entry["needs"] == "cuda" and gpu_kind not in ("cuda", None):
            return "not_compatible", "Official instructions assume an NVIDIA CUDA GPU; none detected."
        if vram < entry["min_vram_gb"]:
            return "not_compatible", f"Needs {entry['min_vram_gb']} GB+ VRAM; best GPU has {vram:g} GB."
        if gpu_kind in ("rocm", "mps"):
            reasons.append(f"{gpu_kind.upper()} support depends on the workflow; test before relying on it")
    else:
        rec = entry.get("rec_vram_gb")
        if rec and (gpu_kind == "none" or vram is None or vram < rec):
            detail = ("no usable GPU" if gpu_kind == "none" else "GPU unknown" if vram is None
                      else f"{vram:g} GB GPU is below the suggested {rec} GB")
            reasons.append(f"runs on CPU ({detail}); expect slower generation")

    if ram is not None and ram < entry["min_ram_gb"] * RAM_TOLERANCE:
        if not entry["cpu_ok"] or ram < entry["min_ram_gb"] * 0.75:
            return "not_compatible", f"Needs {entry['min_ram_gb']} GB RAM; detected {ram:.1f} GB."
        reasons.append(f"RAM ({ram:.1f} GB) is below the suggested {entry['min_ram_gb']} GB")
        marginal = True
    if disk is not None and disk < entry["disk_gb"] * 1.5:
        return "not_compatible", f"Needs ~{entry['disk_gb']:g} GB disk plus headroom; only {disk:.1f} GB free."

    if ram is None:
        reasons.append("RAM unknown")
        marginal = True
    if disk is None:
        reasons.append("free disk unknown")
        marginal = True
    status = "possible" if marginal else "recommended"
    if not reasons:
        return status, "Meets the listed requirements."
    text = "; ".join(reasons)
    return status, text[0].upper() + text[1:] + "."


def recommend(hw: dict[str, Any], installed_ollama: list[str] | None = None,
              comfy_models: list[str] | None = None) -> dict[str, Any]:
    """Pick at most one candidate per category: the most capable (lowest rank) that fits the hardware."""
    installed_ollama = [m.lower() for m in (installed_ollama or [])]
    comfy_models = [m.lower() for m in (comfy_models or [])]
    evaluated = []
    for entry in CATALOG:
        status, reason = _evaluate(entry, hw)
        found = False
        if entry.get("ollama_name"):
            found = any(m == entry["ollama_name"] or m.startswith(entry["ollama_name"] + "-") for m in installed_ollama)
        if entry.get("comfy_match"):
            found = any(key in m for m in comfy_models for key in entry["comfy_match"])
        evaluated.append({**entry, "status": status, "reason": reason, "installed": found})

    picks: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for category in CATEGORY_LABELS:
        group = [e for e in evaluated if e["category"] == category]
        usable = [e for e in group if e["status"] in ("recommended", "possible")]
        usable.sort(key=lambda e: (e["status"] != "recommended", e["rank"]))
        pick = usable[0] if usable else None
        if pick:
            picks.append({**pick, "category_label": CATEGORY_LABELS[category]})
        others.extend({**e, "category_label": CATEGORY_LABELS[category]} for e in group if e is not pick)
    return {
        "catalog_checked": CATALOG_CHECKED,
        "recommended": picks,
        "other_candidates": others,
        "disclaimer": ("Planning guidance only. Figures are approximate and come from each official model card; "
                       "confirm requirements and licence terms there before downloading. Nothing is downloaded "
                       "or installed by this app."),
    }
