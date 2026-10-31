"""Phase 3 smoke test over real HTTP: studio records, reference library, approvals, Phase 4 bridge.

Runs the real app in a subprocess against a temporary workspace and a local fake ComfyUI.
Not part of the pytest suite: `python tools/smoke_phase3.py`
"""
import base64
import hashlib
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib

PORT_API, PORT_COMFY = 8768, 8190
BASE = f"http://127.0.0.1:{PORT_API}"
SCRIPT = "INT. CAFE - DAY\n\nMAYA waits by the window.\n\nMAYA\nYou came.\n"
WORKFLOW = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "x", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "y", "clip": ["4", 1]}},
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


def png(width=24, height=18, rgb=(180, 90, 60)):
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def call(method, path, body=None, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{BASE}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
            return (response.status, payload) if raw else (response.status, json.loads(payload or b"null"))
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        return exc.code, (payload if raw else json.loads(payload or b"null"))


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
    workspace = pathlib.Path(tempfile.mkdtemp(prefix="fh_p3_"))
    env = {**os.environ, "FILM_HOUSE_PROJECTS_DIR": str(workspace),
           "COMFYUI_URL": f"http://127.0.0.1:{PORT_COMFY}", "FILM_HOUSE_JOB_POLL_SECONDS": "0.3"}
    comfy = subprocess.Popen([sys.executable, "tools/fake_comfyui.py", str(PORT_COMFY)], env=env,
                             stdout=subprocess.DEVNULL)
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT_API)], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    failures: list[str] = []

    def check(label, condition, detail=""):
        print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
        if not condition:
            failures.append(label)

    try:
        if not (wait_up(f"http://127.0.0.1:{PORT_COMFY}/system_stats") and wait_up(f"{BASE}/api/health")):
            print("FAIL  servers did not start")
            return 1
        _, project = call("POST", "/api/projects", {"title": "Phase 3 smoke"})
        pid = project["id"]
        call("PUT", f"/api/projects/{pid}/script", {"script": SCRIPT})
        call("POST", f"/api/projects/{pid}/breakdown", {})
        root = workspace / pid

        # ---- subjects
        _, character = call("POST", f"/api/projects/{pid}/characters", {
            "name": "Maya Rao", "age_range": "late 20s", "physical_description": "Tall, left-handed",
            "facial_traits": "Scar above right brow", "hairstyle": "Shoulder-length black"})
        _, location = call("POST", f"/api/projects/{pid}/locations", {
            "name": "Rooftop Cafe", "continuity_requirements": "Rain always on the glass"})
        _, prop = call("POST", f"/api/projects/{pid}/props", {"name": "Brass Key", "prop_type": "hand prop"})
        _, wardrobe = call("POST", f"/api/projects/{pid}/characters/{character['id']}/wardrobe",
                           {"name": "Rain coat", "description": "Olive parka"})
        check("character, location, prop and wardrobe created",
              all(x.get("id") for x in (character, location, prop, wardrobe)))

        # ---- import by upload and by path
        image = png(32, 24)
        code, face = call("POST", f"/api/projects/{pid}/assets", {
            "filename": "maya_face.png", "data_base64": base64.b64encode(image).decode(),
            "license": "my own photo",
            "association": {"subject_type": "character", "subject_id": character["id"], "role": "face"}})
        check("upload imported", code == 201 and face["sha256"] == hashlib.sha256(image).hexdigest(),
              f"{face.get('path')} {face.get('bytes')}B {face.get('integrity')}")
        stored = root / face["path"]
        check("stored file matches and is read-only",
              stored.read_bytes() == image and not (stored.stat().st_mode & 0o200))

        source = workspace / "outside_face.png"
        source.write_bytes(png(40, 30))
        code, from_path = call("POST", f"/api/projects/{pid}/assets", {
            "source_path": str(source),
            "association": {"subject_type": "location", "subject_id": location["id"], "role": "location_wide"}})
        check("import from an absolute path", code == 201 and source.exists(), "original left in place")

        code, duplicate = call("POST", f"/api/projects/{pid}/assets", {
            "filename": "again.png", "data_base64": base64.b64encode(image).decode()})
        check("duplicate refused", code == 409 and duplicate["detail"]["code"] == "DUPLICATE_ASSET")

        code, corrupt = call("POST", f"/api/projects/{pid}/assets", {
            "filename": "broken.png", "data_base64": base64.b64encode(png(64, 64)[:-9]).decode()})
        check("corrupt upload refused", code == 422 and corrupt["detail"]["code"] == "CORRUPT_IMAGE",
              corrupt["detail"]["message"][:60])
        code, traversal = call("POST", f"/api/projects/{pid}/assets", {"source_path": "../../../etc/passwd"})
        check("path traversal refused", code == 422, traversal["detail"]["code"])

        # ---- approval + canonical
        code, _ = call("POST", f"/api/projects/{pid}/characters/{character['id']}/canonical",
                       {"asset_id": face["id"]})
        check("draft cannot be canonical", code == 409)
        call("POST", f"/api/projects/{pid}/assets/{face['id']}/state", {"state": "approved"})
        code, canonical = call("POST", f"/api/projects/{pid}/characters/{character['id']}/canonical",
                               {"asset_id": face["id"]})
        check("approved image becomes canonical", code == 200 and canonical["canonical_reference_id"] == face["id"])

        second = png(33, 24)
        _, face2 = call("POST", f"/api/projects/{pid}/assets", {
            "filename": "maya_face_v2.png", "data_base64": base64.b64encode(second).decode(),
            "association": {"subject_type": "character", "subject_id": character["id"], "role": "face"}})
        call("POST", f"/api/projects/{pid}/assets/{face2['id']}/state", {"state": "approved"})
        check("version counts up per subject and role", face2["version"] == 2, f"v{face2['version']}")
        code, blocked = call("POST", f"/api/projects/{pid}/characters/{character['id']}/canonical",
                             {"asset_id": face2["id"]})
        check("replacing canonical needs confirmation", code == 409
              and blocked["detail"]["code"] == "CANONICAL_EXISTS")
        code, replaced = call("POST", f"/api/projects/{pid}/characters/{character['id']}/canonical",
                              {"asset_id": face2["id"], "confirm_replace": True, "note": "better light"})
        check("explicit replacement works", code == 200 and replaced["replaced_asset_id"] == face["id"])
        _, kept = call("GET", f"/api/projects/{pid}/assets/{face['id']}")
        check("replaced version is preserved", kept["state"] == "approved" and kept["file_status"] == "ok")

        code, refused = call("POST", f"/api/projects/{pid}/assets/{face2['id']}/state", {"state": "rejected"})
        check("canonical image cannot be rejected", code == 409
              and refused["detail"]["code"] == "REFERENCED_AS_CANONICAL")

        # ---- wardrobe keeps identity separate
        costume = png(28, 28)
        _, parka = call("POST", f"/api/projects/{pid}/assets", {
            "filename": "parka.png", "data_base64": base64.b64encode(costume).decode(),
            "association": {"subject_type": "character", "subject_id": character["id"],
                            "wardrobe_id": wardrobe["id"], "role": "wardrobe"}})
        call("POST", f"/api/projects/{pid}/assets/{parka['id']}/state", {"state": "approved"})
        code, _ = call("POST", f"/api/projects/{pid}/characters/{character['id']}/wardrobe/{wardrobe['id']}/canonical",
                       {"asset_id": parka["id"]})
        _, studio = call("GET", f"/api/projects/{pid}/studio")
        record = next(c for c in studio["characters"] if c["id"] == character["id"])
        check("identity and wardrobe canonicals are separate",
              code == 200 and record["canonical_reference_id"] == face2["id"]
              and record["wardrobe"][0]["canonical_reference_id"] == parka["id"])
        check("identity fields untouched by the costume variant",
              record["physical_description"] == "Tall, left-handed" and record["age_range"] == "late 20s")

        # ---- scene association
        _, full = call("GET", f"/api/projects/{pid}")
        scene = full["scenes"][0]
        call("POST", f"/api/projects/{pid}/assets/{from_path['id']}/state", {"state": "approved"})
        code, associated = call("PUT", f"/api/projects/{pid}/scenes/{scene['id']}/associations", {
            "character_ids": [character["id"]], "location_id": location["id"], "prop_ids": [prop["id"]],
            "location_reference_id": from_path["id"],
            "appearance_selections": [{"character_id": character["id"], "wardrobe_id": wardrobe["id"],
                                       "asset_id": parka["id"]}]})
        check("scene associations saved", code == 200
              and associated["scene"]["location_reference_id"] == from_path["id"]
              and associated["scene"]["appearance_selections"][0]["asset_id"] == parka["id"])
        check("screenplay untouched", associated["scene"]["body"] == scene["body"]
              and call("GET", f"/api/projects/{pid}/script/original", raw=True)[1].decode() == SCRIPT.strip())

        # ---- Phase 4 bridge
        _, imported = call("POST", f"/api/projects/{pid}/workflows", {"name": "wf", "workflow": WORKFLOW})
        wid = imported["record"]["id"]
        call("PUT", f"/api/projects/{pid}/workflows/{wid}/mapping", {"mapping": MAPPING})
        code, job = call("POST", f"/api/projects/{pid}/jobs", {
            "workflow_id": wid, "prompt": "medium close-up of Maya at the window",
            "reference_asset_id": face2["id"], "target_type": "character", "target_id": character["id"]})
        check("job accepted with a library reference", code == 201, job.get("status"))
        for _ in range(100):
            _, job = call("GET", f"/api/projects/{pid}/jobs/{job['id']}")
            if job["status"] not in ("queued", "submitted", "running"):
                break
            time.sleep(0.3)
        check("generation succeeded using the approved asset", job["status"] == "succeeded",
              job.get("error") or job["reference_uploaded_as"])
        check("the uploaded reference was the library file",
              (job.get("reference_uploaded_as") or "").endswith(f"{face2['id']}.png"))

        code, both = call("POST", f"/api/projects/{pid}/jobs", {
            "workflow_id": wid, "prompt": "p", "reference_asset_id": face2["id"],
            "reference_path": "references/input/x.png"})
        check("two reference sources refused", code == 422)

        no_reference = {k: v for k, v in WORKFLOW.items() if k != "12"}
        _, plain = call("POST", f"/api/projects/{pid}/workflows", {"name": "plain", "workflow": no_reference})
        plain_id = plain["record"]["id"]
        call("PUT", f"/api/projects/{pid}/workflows/{plain_id}/mapping",
             {"mapping": {k: v for k, v in MAPPING.items() if k != "reference_image"}})
        code, unmapped = call("POST", f"/api/projects/{pid}/jobs", {
            "workflow_id": plain_id, "prompt": "p", "reference_asset_id": face2["id"]})
        check("workflow without a reference node explains itself",
              code == 409 and unmapped["detail"]["code"] == "ROLE_UNMAPPED"
              and "LoadImage" in unmapped["detail"]["message"])

        code, generated = call("POST", f"/api/projects/{pid}/assets/from-job", {
            "job_id": job["id"], "output_index": 0,
            "association": {"subject_type": "location", "subject_id": location["id"], "role": "location_wide"}})
        check("generated output imported as a draft", code == 201 and generated["source"] == "generated"
              and generated["state"] == "draft", generated.get("path"))
        check("job output file still in place", (root / job["outputs"][0]["path"]).exists())

        # ---- safe deletion
        code, in_use = call("DELETE", f"/api/projects/{pid}/assets/{face2['id']}", {"confirm": True})
        check("canonical image cannot be deleted", code == 409 and in_use["detail"]["code"] == "ASSET_IN_USE")
        code, unconfirmed = call("DELETE", f"/api/projects/{pid}/assets/{generated['id']}", {})
        check("deletion needs confirmation", code == 400
              and unconfirmed["detail"]["code"] == "CONFIRMATION_REQUIRED")
        code, deleted = call("DELETE", f"/api/projects/{pid}/assets/{generated['id']}", {"confirm": True})
        check("confirmed deletion removes record and file",
              code == 200 and deleted["file_removed"] is True and not (root / generated["path"]).exists())

        _, removal = call("DELETE", f"/api/projects/{pid}/characters/{character['id']}", {})
        _, after = call("GET", f"/api/projects/{pid}/assets")
        check("deleting a character keeps its images",
              len(removal["assets_unassigned"]) == 3
              and all(a["subject_id"] is None for a in after["assets"] if a["id"] in removal["assets_unassigned"])
              and (root / face["path"]).exists())

        _, warnings = call("GET", f"/api/projects/{pid}/warnings")
        check("validation reports studio state", any(w["code"] in ("ASSET_ORPHANED", "NO_CANONICAL_REFERENCE")
                                                      for w in warnings["warnings"]),
              f"{warnings['summary']}")
        _, manifest = call("GET", f"/api/projects/{pid}/manifest")
        asset_files = [f for f in manifest["files"] if f["kind"] == "reference_asset"]
        check("manifest lists asset files with hashes",
              len(asset_files) == 4 and all(f["read_only"] for f in asset_files), f"{len(asset_files)} files")
        code, page = call("GET", "/studio", raw=True)
        check("studio page served", code == 200 and b"Characters" in page)

        print("\nproject files:")
        for path in sorted(root.rglob("*")):
            if path.is_file():
                print(f"  {path.relative_to(root).as_posix()}  ({path.stat().st_size} bytes)")
    finally:
        for process in (api, comfy):
            process.terminate()
        for path in workspace.rglob("*"):
            if path.is_file():
                try:
                    path.chmod(0o666)
                except OSError:
                    pass
        shutil.rmtree(workspace, ignore_errors=True)
    print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
