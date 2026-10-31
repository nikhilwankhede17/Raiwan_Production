from __future__ import annotations

import ipaddress
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_LOCAL_HOSTNAMES = {"localhost", "localhost.localdomain"}


def is_local_url(url: str) -> bool:
    """True only for http(s) URLs that point at this machine (loopback)."""
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host in _LOCAL_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _get_json(url: str, timeout: float = 1.5) -> tuple[bool, Any]:
    """GET a localhost URL. Non-local URLs are refused, never contacted."""
    if not is_local_url(url):
        return False, None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read(2_000_000).decode("utf-8", errors="replace")
            try:
                return True, json.loads(payload)
            except json.JSONDecodeError:
                return True, payload[:500]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False, None


def service_urls() -> tuple[str, str]:
    comfy_url = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
    ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    return comfy_url, ollama_url


def check_services() -> dict[str, Any]:
    """Check optional local services only. No remote API calls are made."""
    comfy_url, ollama_url = service_urls()
    comfy_ok, _ = _get_json(f"{comfy_url}/system_stats")
    ollama_ok, ollama_data = _get_json(f"{ollama_url}/api/tags")
    models = []
    if ollama_ok and isinstance(ollama_data, dict):
        models = [m.get("name", "unknown") for m in ollama_data.get("models", [])]
    return {
        "mode": "local-first; no hosted model APIs configured",
        "comfyui": {"available": comfy_ok, "url": comfy_url, "local": is_local_url(comfy_url)},
        "ollama": {"available": ollama_ok, "url": ollama_url, "models": models, "local": is_local_url(ollama_url)},
        "notes": [
            "Model workflows and checkpoints must be installed separately.",
            "This starter never sends scripts or assets to external services.",
            "Only localhost service URLs are contacted; other COMFYUI_URL/OLLAMA_URL values are ignored.",
        ],
    }
