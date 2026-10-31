"""Local-only ComfyUI adapter.

Talks to a ComfyUI instance on this machine over its HTTP API. Every request has an explicit
timeout, every failure becomes a readable :class:`ComfyError`, and a non-localhost URL is refused
outright (there is no cloud fallback anywhere in this app).

Adapter contract (docs/ARCHITECTURE.md): :meth:`ComfyAdapter.capabilities`,
:meth:`validate_config`, :meth:`submit`, :meth:`status`, :meth:`cancel`, :meth:`collect_outputs`.
"""
from __future__ import annotations

import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from app.services import is_local_url, service_urls

DEFAULT_TIMEOUT = 10.0
UPLOAD_TIMEOUT = 60.0
DOWNLOAD_TIMEOUT = 60.0
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 128 * 1024 * 1024
USER_AGENT = "ai-film-house/local"


class ComfyError(Exception):
    """A failure talking to ComfyUI. `code` is stable, `message` is for the user."""

    def __init__(self, code: str, message: str, detail: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class ComfyClient:
    """Thin HTTP client for a localhost ComfyUI server."""

    def __init__(self, base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT,
                 client_id: str | None = None):
        url = (base_url if base_url is not None else service_urls()[0]).rstrip("/")
        if not is_local_url(url):
            raise ComfyError(
                "NOT_LOCAL",
                f"Refusing to contact {url or '(empty URL)'}: this app only talks to ComfyUI on this "
                "computer. Set COMFYUI_URL to a 127.0.0.1/localhost address.",
            )
        self.base_url = url
        self.timeout = timeout
        self.client_id = client_id or uuid.uuid4().hex

    # ------------------------------------------------------------------ plumbing

    def _open(self, request: urllib.request.Request, timeout: float | None) -> tuple[int, bytes, str]:
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                limit = MAX_DOWNLOAD_BYTES if request.get_method() == "GET" else MAX_JSON_BYTES
                return response.status, response.read(limit + 1), response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read(MAX_JSON_BYTES)
            except Exception:
                pass
            raise ComfyError(
                "HTTP_ERROR",
                f"ComfyUI returned HTTP {exc.code} for {request.get_method()} {request.full_url}.",
                _decode(body),
            ) from exc
        except TimeoutError as exc:
            raise ComfyError("TIMEOUT", f"ComfyUI did not answer within {timeout or self.timeout:g}s "
                                       f"({request.get_method()} {request.full_url}).") from exc
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                raise ComfyError("TIMEOUT", f"ComfyUI did not answer within {timeout or self.timeout:g}s.") from exc
            raise ComfyError("UNREACHABLE", f"Cannot reach ComfyUI at {self.base_url} ({reason}). "
                                            "Start ComfyUI, or check COMFYUI_URL.") from exc

    def request_json(self, method: str, path: str, body: Any = None, params: dict[str, str] | None = None,
                     timeout: float | None = None) -> Any:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        data = None
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        _status, payload, _ctype = self._open(request, timeout)
        if not payload:
            return None
        try:
            return json.loads(payload.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise ComfyError("BAD_RESPONSE", f"ComfyUI sent a non-JSON reply for {method} {path}.",
                             _decode(payload)[:400]) from exc

    def request_bytes(self, path: str, params: dict[str, str] | None = None,
                      timeout: float | None = None) -> bytes:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
        _status, payload, _ctype = self._open(request, timeout or DOWNLOAD_TIMEOUT)
        if len(payload) > MAX_DOWNLOAD_BYTES:
            raise ComfyError("OUTPUT_TOO_LARGE", f"A ComfyUI output file exceeds the "
                                                 f"{MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB limit.")
        return payload

    # ------------------------------------------------------------------ read-only endpoints

    def system_stats(self) -> dict[str, Any]:
        stats = self.request_json("GET", "/system_stats", timeout=5.0)
        if not isinstance(stats, dict):
            raise ComfyError("BAD_RESPONSE", "ComfyUI /system_stats did not return an object.")
        return stats

    def object_info(self) -> dict[str, Any]:
        """Node definitions of the running server (used to validate a workflow before queueing)."""
        info = self.request_json("GET", "/object_info", timeout=30.0)
        if not isinstance(info, dict):
            raise ComfyError("BAD_RESPONSE", "ComfyUI /object_info did not return an object.")
        return info

    def models(self, folder: str) -> list[str]:
        safe = urllib.parse.quote(folder, safe="")
        try:
            listing = self.request_json("GET", f"/models/{safe}", timeout=10.0)
        except ComfyError as exc:
            if exc.code == "HTTP_ERROR":
                return []
            raise
        return [str(item) for item in listing] if isinstance(listing, list) else []

    def queue(self) -> dict[str, Any]:
        data = self.request_json("GET", "/queue", timeout=5.0)
        return data if isinstance(data, dict) else {}

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        data = self.request_json("GET", f"/history/{urllib.parse.quote(prompt_id)}", timeout=10.0)
        if isinstance(data, dict):
            entry = data.get(prompt_id)
            if isinstance(entry, dict):
                return entry
        return None

    # ------------------------------------------------------------------ job control

    def submit(self, prompt: dict[str, Any], extra_data: dict[str, Any] | None = None) -> str:
        """Queue a patched API workflow. Returns ComfyUI's prompt_id."""
        body: dict[str, Any] = {"prompt": prompt, "client_id": self.client_id}
        if extra_data:
            body["extra_data"] = extra_data
        try:
            response = self.request_json("POST", "/prompt", body=body, timeout=30.0)
        except ComfyError as exc:
            if exc.code == "HTTP_ERROR":
                raise ComfyError("REJECTED", "ComfyUI rejected the workflow: " + _prompt_error_text(exc.detail),
                                 exc.detail) from exc
            raise
        if not isinstance(response, dict) or not response.get("prompt_id"):
            raise ComfyError("REJECTED", "ComfyUI did not return a prompt_id, so nothing was queued.", response)
        if response.get("node_errors"):
            raise ComfyError("REJECTED", "ComfyUI reported node errors: " + _prompt_error_text(response),
                             response["node_errors"])
        return str(response["prompt_id"])

    def job_state(self, prompt_id: str) -> dict[str, Any]:
        """Where a prompt is: pending (queued), running, done, or missing."""
        queue = self.queue()
        running = [_queue_id(item) for item in queue.get("queue_running", []) or []]
        pending = [_queue_id(item) for item in queue.get("queue_pending", []) or []]
        if prompt_id in running:
            return {"state": "running", "queue_position": 0, "queue_pending": len(pending)}
        if prompt_id in pending:
            return {"state": "pending", "queue_position": pending.index(prompt_id) + 1,
                    "queue_pending": len(pending)}
        entry = self.history(prompt_id)
        if entry is not None:
            status = entry.get("status") or {}
            return {"state": "done", "history": entry, "status_str": status.get("status_str"),
                    "completed": bool(status.get("completed", True)), "messages": status.get("messages")}
        return {"state": "missing"}

    def interrupt(self) -> None:
        """Stop the prompt ComfyUI is executing right now (it has no per-prompt interrupt)."""
        self.request_json("POST", "/interrupt", body={}, timeout=10.0)

    def delete_queued(self, prompt_id: str) -> None:
        self.request_json("POST", "/queue", body={"delete": [prompt_id]}, timeout=10.0)

    # ------------------------------------------------------------------ files

    def download_output(self, filename: str, subfolder: str = "", type_: str = "output") -> bytes:
        return self.request_bytes("/view", {"filename": filename, "subfolder": subfolder, "type": type_})

    def upload_image(self, path: Path, subfolder: str = "ai-film-house", overwrite: bool = True) -> dict[str, Any]:
        """Upload a local reference image into ComfyUI's input folder."""
        data = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        boundary = f"----filmhouse{uuid.uuid4().hex}"
        parts: list[bytes] = []

        def field(name: str, value: str) -> None:
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                         f"{value}\r\n".encode("utf-8"))

        field("type", "input")
        field("subfolder", subfolder)
        field("overwrite", "true" if overwrite else "false")
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8") + data + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        request = urllib.request.Request(
            f"{self.base_url}/upload/image", data=b"".join(parts), method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                     "Accept": "application/json", "User-Agent": USER_AGENT})
        _status, payload, _ctype = self._open(request, UPLOAD_TIMEOUT)
        try:
            result = json.loads(payload.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise ComfyError("UPLOAD_FAILED", "ComfyUI did not confirm the reference image upload.",
                             _decode(payload)[:400]) from exc
        if not isinstance(result, dict) or not result.get("name"):
            raise ComfyError("UPLOAD_FAILED", "ComfyUI did not return a filename for the uploaded image.", result)
        name = str(result["name"])
        folder = str(result.get("subfolder") or "")
        return {"name": f"{folder}/{name}" if folder else name, "filename": name, "subfolder": folder}


# --------------------------------------------------------------------------- helpers

def _decode(payload: bytes) -> Any:
    text = payload.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _queue_id(item: Any) -> str | None:
    """Queue entries are [number, prompt_id, prompt, extra_data, outputs]."""
    if isinstance(item, (list, tuple)) and len(item) > 1:
        return str(item[1])
    if isinstance(item, dict):
        return str(item.get("prompt_id")) if item.get("prompt_id") else None
    return None


def _prompt_error_text(detail: Any) -> str:
    """Turn ComfyUI's /prompt error payload into one readable sentence."""
    if not isinstance(detail, dict):
        return str(detail)[:300] if detail else "no details were returned."
    pieces: list[str] = []
    error = detail.get("error")
    if isinstance(error, dict):
        pieces.append(str(error.get("message") or error.get("type") or "unknown error"))
        if error.get("details"):
            pieces.append(str(error["details"])[:200])
    elif error:
        pieces.append(str(error)[:200])
    node_errors = detail.get("node_errors")
    if isinstance(node_errors, dict):
        for node_id, info in list(node_errors.items())[:5]:
            errors = (info or {}).get("errors") if isinstance(info, dict) else None
            if isinstance(errors, list) and errors:
                first = errors[0] or {}
                where = first.get("extra_info", {}).get("input_name") if isinstance(first, dict) else None
                pieces.append(f"node {node_id}"
                              + (f" input '{where}'" if where else "")
                              + f": {first.get('message', 'invalid') if isinstance(first, dict) else first}")
            else:
                pieces.append(f"node {node_id}: invalid")
    return "; ".join(p for p in pieces if p) or "no details were returned."


def client_from_env(timeout: float = DEFAULT_TIMEOUT) -> ComfyClient:
    return ComfyClient(timeout=timeout)


# --------------------------------------------------------------------------- adapter

class ComfyAdapter:
    """Adapter-contract wrapper. Owns no project state; the job manager drives it."""

    name = "comfyui"

    def __init__(self, client: ComfyClient | None = None):
        self._client = client

    @property
    def client(self) -> ComfyClient:
        if self._client is None:
            self._client = client_from_env()
        return self._client

    def capabilities(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "kinds": ["image"],
            "local_only": True,
            "requires": "A ComfyUI server running on this computer with your own models installed.",
            "features": {
                "queue": True, "cancel_running": True, "cancel_queued": True,
                "progress_percent": False, "reference_image_upload": True,
            },
            "notes": [
                "Workflows are not interchangeable: each imported workflow is pinned by SHA-256 and needs its "
                "own node mapping.",
                "Nothing is downloaded or installed; models and workflows come from your ComfyUI setup.",
                "Percent progress needs ComfyUI's WebSocket; this adapter polls queue/history instead.",
            ],
        }

    def validate_config(self) -> dict[str, Any]:
        """Is a usable local ComfyUI there? Never raises."""
        try:
            client = self.client
        except ComfyError as exc:
            return {"ok": False, "url": None, **exc.to_dict()}
        try:
            stats = client.system_stats()
        except ComfyError as exc:
            return {"ok": False, "url": client.base_url, **exc.to_dict()}
        system = stats.get("system") if isinstance(stats.get("system"), dict) else {}
        devices = [
            {"name": d.get("name"), "type": d.get("type"),
             "vram_total_gb": round((d.get("vram_total") or 0) / 1024 ** 3, 1) or None}
            for d in stats.get("devices") or [] if isinstance(d, dict)
        ]
        return {
            "ok": True, "url": client.base_url,
            "comfyui_version": system.get("comfyui_version", "unknown"),
            "pytorch_version": system.get("pytorch_version", "unknown"),
            "devices": devices,
            "gpu_available": any(d["type"] not in (None, "cpu") for d in devices),
        }

    def submit(self, prompt: dict[str, Any], extra_data: dict[str, Any] | None = None) -> str:
        return self.client.submit(prompt, extra_data)

    def status(self, prompt_id: str) -> dict[str, Any]:
        return self.client.job_state(prompt_id)

    def cancel(self, prompt_id: str) -> dict[str, Any]:
        """Delete it from the queue if pending; interrupt it if it is the running prompt."""
        state = self.client.job_state(prompt_id)
        if state["state"] == "pending":
            self.client.delete_queued(prompt_id)
            return {"cancelled": True, "how": "removed from the ComfyUI queue"}
        if state["state"] == "running":
            self.client.interrupt()
            return {"cancelled": True, "how": "interrupted the running ComfyUI prompt"}
        return {"cancelled": False, "how": f"nothing to cancel (ComfyUI state: {state['state']})"}

    def collect_outputs(self, prompt_id: str) -> list[dict[str, Any]]:
        """Image descriptors ComfyUI recorded for this prompt (no files are fetched here)."""
        entry = self.client.history(prompt_id)
        if entry is None:
            return []
        found: list[dict[str, Any]] = []
        for node_id, output in (entry.get("outputs") or {}).items():
            if not isinstance(output, dict):
                continue
            for key in ("images", "gifs", "videos"):
                for item in output.get(key) or []:
                    if isinstance(item, dict) and item.get("filename"):
                        found.append({"node_id": str(node_id), "filename": str(item["filename"]),
                                      "subfolder": str(item.get("subfolder") or ""),
                                      "type": str(item.get("type") or "output")})
        return found
