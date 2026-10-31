"""A tiny stand-in for ComfyUI's HTTP API, for the local smoke test. Not part of the app."""
import json
import re
import struct
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


def png(width=8, height=8, rgb=(210, 120, 40)):
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


IMAGE = png()
OBJECT_INFO = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["sd_xl_base_1.0.safetensors"]]}}},
    "EmptyLatentImage": {"input": {"required": {"width": ["INT"], "height": ["INT"], "batch_size": ["INT"]}}},
    "CLIPTextEncode": {"input": {"required": {"text": ["STRING"], "clip": ["CLIP"]}}},
    "KSampler": {"input": {"required": {"seed": ["INT"], "steps": ["INT"], "cfg": ["FLOAT"], "sampler_name": [[]],
                                        "scheduler": [[]], "denoise": ["FLOAT"], "model": ["MODEL"],
                                        "positive": ["CONDITIONING"], "negative": ["CONDITIONING"],
                                        "latent_image": ["LATENT"]}}},
    "VAEDecode": {"input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}}},
    "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING"]}}},
    "LoadImage": {"input": {"required": {"image": [["example.png"]]}}},
}

STATE = {"jobs": {}, "counter": 0, "uploads": [], "interrupts": 0}
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, payload, status=200, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)
        if path == "/system_stats":
            return self._send({"system": {"comfyui_version": "0.3.60-fake", "python_version": "3.12.7",
                                          "pytorch_version": "2.7.0+fake"},
                               "devices": [{"name": "cpu", "type": "cpu", "vram_total": 0, "vram_free": 0}]})
        if path == "/object_info":
            return self._send(OBJECT_INFO)
        if path.startswith("/models/"):
            folder = path.rsplit("/", 1)[-1]
            return self._send(["sd_xl_base_1.0.safetensors"] if folder == "checkpoints" else [])
        if path == "/queue":
            with LOCK:
                running, pending = [], []
                for prompt_id, job in STATE["jobs"].items():
                    entry = [job["number"], prompt_id, job["prompt"], {}, []]
                    elapsed = time.monotonic() - job["at"]
                    if job["done"]:
                        continue
                    if elapsed < 0.4:
                        pending.append(entry)
                    elif elapsed < 1.2:
                        running.append(entry)
                    else:
                        job["done"] = True
                return self._send({"queue_running": running, "queue_pending": pending})
        if path.startswith("/history/"):
            prompt_id = path.rsplit("/", 1)[-1]
            with LOCK:
                job = STATE["jobs"].get(prompt_id)
                if not job or not job["done"] or job.get("cancelled"):
                    return self._send({})
                save_nodes = [nid for nid, node in job["prompt"].items() if node["class_type"] == "SaveImage"]
                outputs = {nid: {"images": [{"filename": f"FilmHouse_{job['number']:05d}_.png",
                                             "subfolder": "", "type": "output"}]} for nid in save_nodes}
                return self._send({prompt_id: {"prompt": [job["number"], prompt_id, job["prompt"]],
                                               "status": {"status_str": "success", "completed": True,
                                                          "messages": []},
                                               "outputs": outputs}})
        if path == "/view":
            if not re.fullmatch(r"FilmHouse_\d+_\.png", (query.get("filename") or [""])[0]):
                return self._send({"error": "not found"}, status=404)
            return self._send(IMAGE, content_type="image/png")
        return self._send({"error": "unknown route"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        if parsed.path == "/prompt":
            body = json.loads(raw)
            prompt = body.get("prompt") or {}
            unknown = sorted({n.get("class_type") for n in prompt.values() if n.get("class_type") not in OBJECT_INFO})
            if unknown:
                return self._send({"error": {"type": "prompt_outputs_failed_validation",
                                             "message": f"Unknown node type {unknown[0]}"},
                                   "node_errors": {}}, status=400)
            with LOCK:
                STATE["counter"] += 1
                prompt_id = f"fake-{STATE['counter']:04d}"
                STATE["jobs"][prompt_id] = {"number": STATE["counter"], "prompt": prompt,
                                            "at": time.monotonic(), "done": False}
            return self._send({"prompt_id": prompt_id, "number": STATE["counter"], "node_errors": {}})
        if parsed.path == "/queue":
            body = json.loads(raw or b"{}")
            with LOCK:
                for prompt_id in body.get("delete", []):
                    STATE["jobs"].pop(prompt_id, None)
            return self._send({})
        if parsed.path == "/interrupt":
            with LOCK:
                STATE["interrupts"] += 1
                for job in STATE["jobs"].values():
                    if not job["done"]:
                        job["done"] = True
                        job["cancelled"] = True
            return self._send({})
        if parsed.path == "/upload/image":
            match = re.search(rb'filename="([^"]+)"', raw)
            name = match.group(1).decode() if match else "upload.png"
            with LOCK:
                STATE["uploads"].append(name)
            return self._send({"name": name, "subfolder": "ai-film-house", "type": "input"})
        return self._send({"error": "unknown route"}, status=404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8189
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"fake comfyui listening on 127.0.0.1:{port}", flush=True)
    server.serve_forever()
