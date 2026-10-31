"""Phase 4 smoke test over real HTTP: this app -> fake local ComfyUI. Not part of the test suite."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

PORT_API, PORT_COMFY = 8767, 8189
BASE = f"http://127.0.0.1:{PORT_API}"
WORKFLOW = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "placeholder", "clip": ["4", 1]},
          "_meta": {"title": "Positive"}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad", "clip": ["4", 1]},
          "_meta": {"title": "Negative"}},
    "3": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 7.0, "sampler_name": "euler",
                                               "scheduler": "normal", "denoise": 1.0, "model": ["4", 0],
                                               "positive": ["6", 0], "negative": ["7", 0],
                                               "latent_image": ["5", 0]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "FilmHouse", "images": ["8", 0]}},
    "12": {"class_type": "LoadImage", "inputs": {"image": "example.png"}},
}
MAPPING = {"prompt": {"node_id": "6", "input": "text"}, "negative_prompt": {"node_id": "7", "input": "text"},
           "seed": {"node_id": "3", "input": "seed"}, "width": {"node_id": "5", "input": "width"},
           "height": {"node_id": "5", "input": "height"},
           "reference_image": {"node_id": "12", "input": "image"}, "output_nodes": ["9"]}


def call(method, path, body=None, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{BASE}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
            return (response.status, payload) if raw else (response.status, json.loads(payload or b"null"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"null")


def wait_up(url, seconds=25):
    for _ in range(int(seconds * 5)):
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return True
        except Exception:
            time.sleep(0.2)
    return False


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    workspace = pathlib.Path(tempfile.mkdtemp(prefix="fh_p4_"))
    env = {**os.environ, "FILM_HOUSE_PROJECTS_DIR": str(workspace),
           "COMFYUI_URL": f"http://127.0.0.1:{PORT_COMFY}", "FILM_HOUSE_JOB_POLL_SECONDS": "0.3"}
    comfy = subprocess.Popen([sys.executable, "tools/fake_comfyui.py", str(PORT_COMFY)], env=env)
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT_API)], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
        if not condition:
            failures.append(label)

    try:
        if not (wait_up(f"http://127.0.0.1:{PORT_COMFY}/system_stats") and wait_up(f"{BASE}/api/health")):
            print("FAIL  servers did not start")
            return 1

        _, status = call("GET", "/api/comfyui/status")
        check("ComfyUI reported available", status["config"]["ok"] is True, status["config"].get("comfyui_version"))

        _, project = call("POST", "/api/projects", {"title": "Phase 4 smoke"})
        pid = project["id"]
        call("PUT", f"/api/projects/{pid}/script", {"script": "INT. CAFE - DAY\n\nMAYA\nHi.\n"})
        call("POST", f"/api/projects/{pid}/breakdown", {})

        code, imported = call("POST", f"/api/projects/{pid}/workflows",
                              {"name": "SDXL smoke", "workflow": WORKFLOW, "source_filename": "sdxl_api.json"})
        check("API workflow imported", code == 201 and imported["record"]["node_count"] == 8)
        check("import starts unmapped", imported["record"]["offline_validation"]["valid"] is False)
        check("suggestions trace the sampler", imported["suggestions"]["prompt"]["node_id"] == "6"
              and imported["suggestions"]["negative_prompt"]["node_id"] == "7")
        wid = imported["record"]["id"]

        code, ui = call("POST", f"/api/projects/{pid}/workflows",
                        {"name": "editor export", "workflow": {"nodes": [], "links": []}})
        check("editor-format workflow refused", code == 422 and ui["detail"]["code"] == "UI_FORMAT")

        code, mapped = call("PUT", f"/api/projects/{pid}/workflows/{wid}/mapping", {"mapping": MAPPING})
        check("mapping accepted", code == 200 and mapped["record"]["offline_validation"]["valid"] is True)

        code, live = call("POST", f"/api/projects/{pid}/workflows/{wid}/validate")
        check("live validation against ComfyUI passed",
              code == 200 and live["record"]["live_validation"]["valid"] is True,
              f"errors={len(live['record']['live_validation']['errors'])}")

        reference = workspace / pid / "references" / "input" / "face.png"
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_bytes(urllib.request.urlopen(
            f"http://127.0.0.1:{PORT_COMFY}/view?filename=FilmHouse_00001_.png").read())
        _, listing = call("GET", f"/api/projects/{pid}/reference-images")
        check("reference image listed", any(i["path"] == "references/input/face.png" for i in listing["images"]))

        code, job = call("POST", f"/api/projects/{pid}/jobs", {
            "workflow_id": wid, "prompt": "establishing wide of a rain-soaked cafe at dusk",
            "negative_prompt": "text, watermark", "seed": 4242, "width": 832, "height": 1216,
            "reference_path": "references/input/face.png", "target_type": "location", "label": "Cafe"})
        check("job queued", code == 201, job.get("status"))
        job_id = job["id"]
        for _ in range(100):
            _, job = call("GET", f"/api/projects/{pid}/jobs/{job_id}")
            if job["status"] not in ("queued", "submitted", "running"):
                break
            time.sleep(0.3)
        check("job succeeded", job["status"] == "succeeded", job.get("error") or f"{job['duration_seconds']}s")
        check("seed recorded", job["seed_used"] == 4242)
        check("reference uploaded to ComfyUI", job["reference_uploaded_as"] == "ai-film-house/face.png",
              job.get("reference_uploaded_as"))
        check("one output captured", len(job["outputs"]) == 1)
        check("queue states were observed",
              any("waiting in the ComfyUI queue" in line for line in job["log"])
              and any("ComfyUI is executing" in line for line in job["log"]))
        print("      log:")
        for line in job["log"]:
            print(f"        {line}")

        if job["outputs"]:
            output = job["outputs"][0]
            on_disk = workspace / pid / output["path"]
            code, payload = call("GET", f"/api/projects/{pid}/jobs/{job_id}/outputs/0", raw=True)
            check("output image served and on disk",
                  code == 200 and payload[:8] == b"\x89PNG\r\n\x1a\n" and on_disk.read_bytes() == payload,
                  f"{output['path']} {output['bytes']} bytes")
            metadata = json.loads((workspace / pid / job["metadata_path"]).read_text(encoding="utf-8"))
            check("metadata records seed/workflow/model/provider",
                  metadata["reproduce"]["seed"] == 4242
                  and metadata["workflow"]["sha256"] == job["workflow_sha256"]
                  and metadata["reproduce"]["models"][0]["name"] == "sd_xl_base_1.0.safetensors"
                  and metadata["provider"]["version"] == "0.3.60-fake")

        code, cancelled = call("POST", f"/api/projects/{pid}/jobs/{job_id}/cancel")
        check("finished job cannot be cancelled", code == 409 and cancelled["detail"]["code"] == "JOB_FINISHED")

        _, started = call("POST", f"/api/projects/{pid}/jobs", {"workflow_id": wid, "prompt": "cancel me",
                                                                "randomize_seed": True})
        for _ in range(100):                       # cancel only once ComfyUI is actually executing it
            _, result = call("GET", f"/api/projects/{pid}/jobs/{started['id']}")
            if result["status"] == "running":
                break
            time.sleep(0.05)
        check("second job reached running", result["status"] == "running", result["status"])
        call("POST", f"/api/projects/{pid}/jobs/{started['id']}/cancel")
        for _ in range(60):
            _, result = call("GET", f"/api/projects/{pid}/jobs/{started['id']}")
            if result["status"] not in ("queued", "submitted", "running"):
                break
            time.sleep(0.3)
        check("running job can be cancelled", result["status"] == "cancelled" and result["outputs"] == [],
              result["status"])

        print("\n--- with ComfyUI stopped ---")
        comfy.terminate()
        comfy.wait(timeout=10)
        time.sleep(0.5)
        _, status = call("GET", "/api/comfyui/status")
        check("status reports unavailable", status["config"]["ok"] is False, status["config"]["message"][:60])
        code, refused = call("POST", f"/api/projects/{pid}/jobs", {"workflow_id": wid, "prompt": "should not run"})
        check("job refused with 503", code == 503 and refused["detail"]["code"] == "COMFYUI_UNAVAILABLE")
        _, jobs = call("GET", f"/api/projects/{pid}/jobs")
        check("no extra job was created", len(jobs["jobs"]) == 2, f"{len(jobs['jobs'])} jobs")
        code, validate = call("POST", f"/api/projects/{pid}/workflows/{wid}/validate")
        check("live validation refuses without ComfyUI", code == 503)

        print("\nproject files:")
        root = workspace / pid
        for path in sorted(root.rglob("*")):
            if path.is_file():
                print(f"  {path.relative_to(root).as_posix()}  ({path.stat().st_size} bytes)")
    finally:
        for process in (api, comfy):
            process.terminate()
        for path in workspace.rglob("*"):
            if path.is_file():
                path.chmod(0o666)
        try:
            import shutil
            shutil.rmtree(workspace, ignore_errors=True)
        except Exception:
            pass
    print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
