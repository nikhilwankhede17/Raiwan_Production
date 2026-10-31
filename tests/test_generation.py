"""Phase 4 end-to-end: workflow import/mapping/validation and generation jobs over a fake local ComfyUI."""
import json

import pytest
import time
from fastapi.testclient import TestClient

from app import jobs, storage
from app.comfy import ComfyClient, ComfyError
from app.main import app
from tests.test_comfy import MAPPING, OBJECT_INFO, WORKFLOW

def _real_png(width: int = 8, height: int = 6) -> bytes:
    """A genuine, decodable PNG so output handling is exercised with real image bytes."""
    import struct
    import zlib
    raw = b"".join(b"\x00" + bytes((40, 90, 160)) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


PNG = _real_png()


class FakeComfy(ComfyClient):
    """A scripted localhost ComfyUI. Overrides only the HTTP layer, so real client logic is exercised."""

    def __init__(self, states=("done",), outputs=None, status_str="success", messages=None,
                 reject=None, object_info=None, models=None, download_error=False):
        super().__init__("http://127.0.0.1:8188", client_id="test-client")
        self.states = list(states)
        self.outputs = {"9": {"images": [{"filename": "FilmHouse_00001_.png", "subfolder": "", "type": "output"}]}} \
            if outputs is None else outputs
        self.status_str = status_str
        self.messages = messages or []
        self.reject = reject
        self._object_info = object_info or OBJECT_INFO
        self.models_by_folder = models if models is not None else {"checkpoints": ["sd_xl_base_1.0.safetensors"]}
        self.download_error = download_error
        self.calls: list[tuple[str, str]] = []
        self.submitted: list[dict] = []
        self.uploads: list[str] = []
        self.polls = 0
        self.interrupted = 0
        self.deleted: list[str] = []
        self.prompt_id = "prompt-1"

    @property
    def state(self) -> str:
        return self.states[min(self.polls, len(self.states) - 1)]

    def request_json(self, method, path, body=None, params=None, timeout=None):
        self.calls.append((method, path))
        assert timeout, f"{method} {path} was sent without a timeout"
        if path == "/system_stats":
            return {"system": {"comfyui_version": "0.3.60-test", "pytorch_version": "2.7.0+cu128"},
                    "devices": [{"name": "cuda:0 Fake GPU", "type": "cuda", "vram_total": 12 * 1024 ** 3}]}
        if path == "/object_info":
            return self._object_info
        if path.startswith("/models/"):
            return self.models_by_folder.get(path.rsplit("/", 1)[-1], [])
        if path == "/prompt":
            if self.reject:
                raise self.reject
            self.submitted.append(body)
            return {"prompt_id": self.prompt_id, "number": 1, "node_errors": {}}
        if path == "/queue" and method == "GET":
            state = self.state
            self.polls += 1
            entry = [1, self.prompt_id, {}, {}, []]
            return {"queue_running": [entry] if state == "running" else [],
                    "queue_pending": [entry] if state == "pending" else []}
        if path == "/queue" and method == "POST":
            self.deleted.extend(body.get("delete", []))
            return {}
        if path == "/interrupt":
            self.interrupted += 1
            return {}
        if path.startswith("/history/"):
            if self.state != "done":
                return {}
            return {self.prompt_id: {"status": {"status_str": self.status_str, "completed": True,
                                                "messages": self.messages},
                                     "outputs": self.outputs}}
        return {}

    def request_bytes(self, path, params=None, timeout=None):
        self.calls.append(("GET", path))
        if self.download_error:
            raise ComfyError("HTTP_ERROR", "ComfyUI returned HTTP 404 for GET /view.")
        return PNG

    def upload_image(self, path, subfolder="ai-film-house", overwrite=True):
        self.uploads.append(str(path))
        return {"name": f"{subfolder}/{path.name}", "filename": path.name, "subfolder": subfolder}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def fast_jobs(monkeypatch):
    monkeypatch.setenv("FILM_HOUSE_JOB_POLL_SECONDS", "0")
    monkeypatch.setenv("FILM_HOUSE_JOB_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(jobs.MANAGER, "inline", True)
    monkeypatch.setattr(jobs.MANAGER, "client_factory", lambda: FakeComfy())
    drain()
    yield
    # The manager is a process-wide singleton with one worker thread: never let it outlive a test,
    # or it would write into the next test's (or the real) workspace.
    drain()
    deadline = time.monotonic() + 10.0
    while jobs.MANAGER._active and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not jobs.MANAGER._active, f"a job was still active at teardown: {jobs.MANAGER._active}"


def drain() -> None:
    while not jobs.MANAGER._queue.empty():
        jobs.MANAGER._queue.get_nowait()
        jobs.MANAGER._queue.task_done()


def use_comfy(monkeypatch, fake: FakeComfy) -> FakeComfy:
    monkeypatch.setattr(jobs.MANAGER, "client_factory", lambda: fake)
    return fake


def no_comfy(monkeypatch) -> None:
    def broken():
        raise ComfyError("UNREACHABLE", "Cannot reach ComfyUI at http://127.0.0.1:8188 (connection refused). "
                                        "Start ComfyUI, or check COMFYUI_URL.")
    monkeypatch.setattr(jobs.MANAGER, "client_factory", broken)


def make_project(client) -> str:
    pid = client.post("/api/projects", json={"title": "Phase 4"}).json()["id"]
    client.put(f"/api/projects/{pid}/script", json={"script": "INT. CAFE - DAY\n\nMAYA\nHi.\n"})
    client.post(f"/api/projects/{pid}/breakdown")
    return pid


def import_workflow(client, pid, workflow=None, name="SDXL txt2img") -> dict:
    response = client.post(f"/api/projects/{pid}/workflows",
                           json={"name": name, "workflow": workflow or WORKFLOW, "source_filename": "sdxl_api.json"})
    assert response.status_code == 201, response.text
    return response.json()


def map_workflow(client, pid, wid, mapping=None) -> dict:
    response = client.put(f"/api/projects/{pid}/workflows/{wid}/mapping",
                          json={"mapping": mapping or MAPPING})
    assert response.status_code == 200, response.text
    return response.json()


def ready_workflow(client) -> tuple[str, str]:
    pid = make_project(client)
    wid = import_workflow(client, pid)["record"]["id"]
    map_workflow(client, pid, wid)
    return pid, wid


# --------------------------------------------------------------------------- status endpoint

def test_comfyui_status_reports_availability(client, monkeypatch):
    use_comfy(monkeypatch, FakeComfy())
    body = client.get("/api/comfyui/status").json()
    assert body["capabilities"]["local_only"] is True
    assert body["config"]["ok"] is True and body["config"]["comfyui_version"] == "0.3.60-test"
    assert {r["role"] for r in body["roles"]} == {"prompt", "negative_prompt", "seed", "width", "height",
                                                  "reference_image"}
    no_comfy(monkeypatch)
    body = client.get("/api/comfyui/status").json()
    assert body["config"]["ok"] is False and "Start ComfyUI" in body["config"]["message"]


# --------------------------------------------------------------------------- import & mapping

def test_import_stores_workflow_write_once_with_hash(client, isolated_workspace):
    pid = make_project(client)
    detail = import_workflow(client, pid)
    record = detail["record"]
    path = isolated_workspace / pid / record["path"]
    assert path.exists() and record["path"] == f"workflows/{record['id']}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["9"]["class_type"] == "SaveImage"
    assert record["node_count"] == 7 and record["class_types"]["CLIPTextEncode"] == 2
    assert record["models"][0]["name"] == "sd_xl_base_1.0.safetensors"
    assert record["offline_validation"]["valid"] is False      # nothing mapped yet
    assert detail["suggestions"]["prompt"]["node_id"] == "6"
    assert detail["inspection"]["saving_output_nodes"] == ["9"]
    manifest = client.get(f"/api/projects/{pid}/manifest").json()
    entry = next(f for f in manifest["files"] if f["path"] == record["path"])
    assert entry["kind"] == "comfy_workflow" and entry["read_only"] is True


def test_importing_the_same_workflow_twice_is_refused(client):
    pid = make_project(client)
    import_workflow(client, pid)
    again = client.post(f"/api/projects/{pid}/workflows", json={"name": "copy", "workflow": WORKFLOW})
    assert again.status_code == 409 and again.json()["detail"]["code"] == "WORKFLOW_ALREADY_IMPORTED"


def test_importing_ui_format_is_rejected_with_advice(client):
    pid = make_project(client)
    response = client.post(f"/api/projects/{pid}/workflows",
                           json={"name": "editor file", "workflow": {"nodes": [], "links": []}})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "UI_FORMAT"
    assert client.get(f"/api/projects/{pid}/workflows").json()["workflows"] == []


def test_mapping_roundtrip_and_validation_state(client):
    pid = make_project(client)
    wid = import_workflow(client, pid)["record"]["id"]
    detail = map_workflow(client, pid, wid)
    assert detail["record"]["offline_validation"]["valid"] is True
    assert detail["record"]["mapping"]["prompt"] == {"node_id": "6", "input": "text"}
    assert detail["record"]["mapping_set_at"]
    assert client.get(f"/api/projects/{pid}/workflows").json()["workflows"][0]["ready"] is True
    bad = client.put(f"/api/projects/{pid}/workflows/{wid}/mapping",
                     json={"mapping": {**MAPPING, "prompt": {"node_id": "3", "input": "seed"}}})
    assert bad.status_code == 200
    assert bad.json()["record"]["offline_validation"]["valid"] is False
    assert bad.json()["record"]["offline_validation"]["errors"][0]["code"] == "TYPE_MISMATCH"


def test_live_validation_needs_comfyui(client, monkeypatch):
    pid, wid = ready_workflow(client)
    no_comfy(monkeypatch)
    response = client.post(f"/api/projects/{pid}/workflows/{wid}/validate")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "COMFYUI_UNAVAILABLE"
    assert response.json()["detail"]["offline_validation"]["valid"] is True


def test_live_validation_records_server_result(client, monkeypatch):
    pid, wid = ready_workflow(client)
    fake = use_comfy(monkeypatch, FakeComfy(models={"checkpoints": ["other.safetensors"]}))
    body = client.post(f"/api/projects/{pid}/workflows/{wid}/validate").json()
    live = body["record"]["live_validation"]
    assert live["valid"] is True and live["comfyui_version"] == "0.3.60-test"
    assert live["warnings"][0]["code"] == "MODEL_NOT_FOUND"
    assert ("GET", "/object_info") in fake.calls and ("GET", "/models/checkpoints") in fake.calls
    # Changing the mapping clears the live result so a stale pass is never trusted.
    after = map_workflow(client, pid, wid)
    assert after["record"]["live_validation"] is None


def test_live_validation_reports_missing_node_classes(client, monkeypatch):
    pid = make_project(client)
    wid = import_workflow(client, pid)["record"]["id"]
    map_workflow(client, pid, wid)
    use_comfy(monkeypatch, FakeComfy(object_info={k: v for k, v in OBJECT_INFO.items() if k != "KSampler"}))
    body = client.post(f"/api/projects/{pid}/workflows/{wid}/validate").json()
    assert body["record"]["live_validation"]["valid"] is False
    assert "KSampler" in body["record"]["live_validation"]["errors"][0]["message"]


def test_workflow_json_is_served_back_and_integrity_checked(client, isolated_workspace):
    pid = make_project(client)
    record = import_workflow(client, pid)["record"]
    assert client.get(f"/api/projects/{pid}/workflows/{record['id']}/json").json()["5"]["inputs"]["width"] == 1024
    path = isolated_workspace / pid / record["path"]
    path.chmod(0o666)
    path.write_text('{"1": {"class_type": "X", "inputs": {}}}', encoding="utf-8")
    response = client.get(f"/api/projects/{pid}/workflows/{record['id']}/json")
    assert response.status_code == 500 and response.json()["code"] == "STORAGE_ERROR"


# --------------------------------------------------------------------------- refusing to generate

def test_no_job_is_created_when_comfyui_is_unavailable(client, monkeypatch):
    pid, wid = ready_workflow(client)
    no_comfy(monkeypatch)
    response = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "a cafe at dusk"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "COMFYUI_UNAVAILABLE"
    assert client.get(f"/api/projects/{pid}/jobs").json()["jobs"] == []


def test_unvalidated_workflow_cannot_generate(client):
    pid = make_project(client)
    wid = import_workflow(client, pid)["record"]["id"]
    response = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "x"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "WORKFLOW_NOT_VALIDATED"


def test_values_without_a_mapped_role_are_refused(client):
    pid = make_project(client)
    wid = import_workflow(client, pid)["record"]["id"]
    map_workflow(client, pid, wid, {"prompt": {"node_id": "6", "input": "text"}, "output_nodes": ["9"]})
    response = client.post(f"/api/projects/{pid}/jobs",
                           json={"workflow_id": wid, "prompt": "x", "width": 512})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "ROLE_UNMAPPED" and "width" in response.json()["detail"]["message"]


def test_unknown_workflow_and_bad_payloads(client):
    pid = make_project(client)
    assert client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": "x" * 12, "prompt": "p"}).status_code == 404
    _pid, wid = ready_workflow(client)
    assert client.post(f"/api/projects/{_pid}/jobs", json={"workflow_id": wid, "prompt": ""}).status_code == 422
    assert client.post(f"/api/projects/{_pid}/jobs",
                       json={"workflow_id": wid, "prompt": "p", "seed": -5}).status_code == 422
    assert client.post(f"/api/projects/{_pid}/jobs",
                       json={"workflow_id": wid, "prompt": "p", "width": 10}).status_code == 422


def test_reference_outside_the_project_is_refused(client, monkeypatch):
    pid = make_project(client)
    wid = import_workflow(client, pid, {**WORKFLOW, "12": {"class_type": "LoadImage",
                                                           "inputs": {"image": "example.png"}}})["record"]["id"]
    map_workflow(client, pid, wid, {**MAPPING, "reference_image": {"node_id": "12", "input": "image"}})
    for bad in ("../../secret.png", "project.json", "references/input/missing.png"):
        response = client.post(f"/api/projects/{pid}/jobs",
                               json={"workflow_id": wid, "prompt": "p", "reference_path": bad})
        assert response.status_code == 409, bad
        assert response.json()["detail"]["code"] in ("BAD_REFERENCE", "LOCAL_ERROR")
    assert client.get(f"/api/projects/{pid}/jobs").json()["jobs"] == []


# --------------------------------------------------------------------------- the happy path

def test_generate_one_image_end_to_end(client, monkeypatch, isolated_workspace):
    pid, wid = ready_workflow(client)
    fake = use_comfy(monkeypatch, FakeComfy(states=["pending", "running", "done"]))
    response = client.post(f"/api/projects/{pid}/jobs", json={
        "workflow_id": wid, "prompt": "wide shot of a rain-soaked cafe at dusk", "negative_prompt": "text, watermark",
        "seed": 12345, "width": 832, "height": 1216, "target_type": "location", "target_id": "cafe",
        "label": "Cafe establishing"})
    assert response.status_code == 201, response.text
    job = response.json()
    assert job["status"] == "succeeded", job["error"]
    assert job["prompt_id"] == "prompt-1" and job["seed_used"] == 12345
    assert job["comfyui_version"] == "0.3.60-test" and job["provider"] == "comfyui"
    assert job["workflow_sha256"] and job["workflow_name"] == "SDXL txt2img"
    assert job["models"][0]["name"] == "sd_xl_base_1.0.safetensors"
    assert job["duration_seconds"] is not None

    # The patched workflow carried every mapped value, and only those.
    prompt = fake.submitted[0]["prompt"]
    assert prompt["6"]["inputs"]["text"] == "wide shot of a rain-soaked cafe at dusk"
    assert prompt["7"]["inputs"]["text"] == "text, watermark"
    assert prompt["3"]["inputs"]["seed"] == 12345 and prompt["3"]["inputs"]["steps"] == 20
    assert prompt["5"]["inputs"]["width"] == 832 and prompt["5"]["inputs"]["height"] == 1216
    assert fake.submitted[0]["client_id"] == "test-client"

    output = job["outputs"][0]
    assert output["path"] == f"references/{job['id']}/00_FilmHouse_00001_.png"
    saved = isolated_workspace / pid / output["path"]
    assert saved.read_bytes() == PNG and output["bytes"] == len(PNG)
    assert output["sha256"] == __import__("hashlib").sha256(PNG).hexdigest()

    metadata = json.loads((isolated_workspace / pid / job["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["reproduce"]["seed"] == 12345
    assert metadata["workflow"]["sha256"] == job["workflow_sha256"]
    assert metadata["workflow"]["mapping"]["prompt"] == {"node_id": "6", "input": "text"}
    assert metadata["provider"]["version"] == "0.3.60-test"
    assert any("queued in ComfyUI" in line for line in job["log"])
    assert any("ComfyUI is executing" in line for line in job["log"])

    served = client.get(f"/api/projects/{pid}/jobs/{job['id']}/outputs/0")
    assert served.status_code == 200 and served.content == PNG
    assert client.get(f"/api/projects/{pid}/jobs/{job['id']}/outputs/5").status_code == 404

    manifest = client.get(f"/api/projects/{pid}/manifest").json()
    kinds = {f["path"]: f["kind"] for f in manifest["files"]}
    assert kinds[output["path"]] == "generated_image"
    assert kinds[job["metadata_path"]] == "generation_metadata"


def test_shot_target_writes_into_storyboards(client, monkeypatch, isolated_workspace):
    pid, wid = ready_workflow(client)
    use_comfy(monkeypatch, FakeComfy())
    shot_id = client.get(f"/api/projects/{pid}").json()["shots"][0]["id"]
    job = client.post(f"/api/projects/{pid}/jobs", json={
        "workflow_id": wid, "prompt": "storyboard frame", "target_type": "shot", "target_id": shot_id}).json()
    assert job["outputs"][0]["path"].startswith(f"storyboards/{job['id']}/")


def test_randomized_seed_is_recorded(client, monkeypatch):
    pid, wid = ready_workflow(client)
    use_comfy(monkeypatch, FakeComfy())
    first = client.post(f"/api/projects/{pid}/jobs",
                        json={"workflow_id": wid, "prompt": "p", "randomize_seed": True}).json()
    second = client.post(f"/api/projects/{pid}/jobs",
                         json={"workflow_id": wid, "prompt": "p", "randomize_seed": True}).json()
    assert first["seed_used"] != second["seed_used"]
    assert all(0 <= job["seed_used"] < 2 ** 53 for job in (first, second))


def test_reference_image_is_uploaded_and_mapped(client, monkeypatch, isolated_workspace):
    pid = make_project(client)
    wid = import_workflow(client, pid, {**WORKFLOW, "12": {"class_type": "LoadImage",
                                                           "inputs": {"image": "example.png"}}})["record"]["id"]
    map_workflow(client, pid, wid, {**MAPPING, "reference_image": {"node_id": "12", "input": "image"}})
    reference = isolated_workspace / pid / "references" / "input" / "face.png"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(PNG)
    listing = client.get(f"/api/projects/{pid}/reference-images").json()
    assert listing["images"][0]["path"] == "references/input/face.png"
    fake = use_comfy(monkeypatch, FakeComfy())
    job = client.post(f"/api/projects/{pid}/jobs", json={
        "workflow_id": wid, "prompt": "portrait", "reference_path": "references/input/face.png",
        "target_type": "character"}).json()
    assert job["status"] == "succeeded", job["error"]
    assert fake.uploads and fake.uploads[0].endswith("face.png")
    assert job["reference_uploaded_as"] == "ai-film-house/face.png"
    assert fake.submitted[0]["prompt"]["12"]["inputs"]["image"] == "ai-film-house/face.png"


def test_outputs_are_limited_to_mapped_output_nodes(client, monkeypatch):
    pid, wid = ready_workflow(client)
    use_comfy(monkeypatch, FakeComfy(outputs={
        "9": {"images": [{"filename": "wanted.png", "subfolder": "", "type": "output"}]},
        "13": {"images": [{"filename": "preview.png", "subfolder": "", "type": "temp"}]}}))
    job = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "p"}).json()
    assert [o["filename"] for o in job["outputs"]] == ["00_wanted.png"]


def test_retry_reuses_the_request(client, monkeypatch):
    pid, wid = ready_workflow(client)
    use_comfy(monkeypatch, FakeComfy())
    first = client.post(f"/api/projects/{pid}/jobs",
                        json={"workflow_id": wid, "prompt": "a cafe", "seed": 7}).json()
    same = client.post(f"/api/projects/{pid}/jobs/{first['id']}/retry", json={}).json()
    assert same["request"]["prompt"] == "a cafe" and same["seed_used"] == 7 and same["id"] != first["id"]
    fresh = client.post(f"/api/projects/{pid}/jobs/{first['id']}/retry", json={"randomize_seed": True}).json()
    assert fresh["seed_used"] != 7
    assert len(client.get(f"/api/projects/{pid}/jobs").json()["jobs"]) == 3


# --------------------------------------------------------------------------- failures

def test_comfy_rejection_fails_the_job_with_the_server_message(client, monkeypatch):
    pid, wid = ready_workflow(client)
    use_comfy(monkeypatch, FakeComfy(reject=ComfyError(
        "REJECTED", "ComfyUI rejected the workflow: Value not in list: ckpt_name")))
    job = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "p"}).json()
    assert job["status"] == "failed" and job["error"]["code"] == "REJECTED"
    assert "ckpt_name" in job["error"]["message"] and job["outputs"] == []


def test_execution_error_in_history_is_reported(client, monkeypatch):
    messages = [["execution_error", {"node_id": "3", "node_type": "KSampler",
                                     "exception_type": "torch.OutOfMemoryError",
                                     "exception_message": "CUDA out of memory"}]]
    use_comfy(monkeypatch, FakeComfy(status_str="error", messages=messages))
    pid, wid = ready_workflow(client)
    job = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "p"}).json()
    assert job["status"] == "failed" and job["error"]["code"] == "EXECUTION_ERROR"
    assert "CUDA out of memory" in job["error"]["message"] and "KSampler" in job["error"]["message"]
    assert job["outputs"] == []


def test_success_without_images_is_not_claimed_as_output(client, monkeypatch):
    pid, wid = ready_workflow(client)
    use_comfy(monkeypatch, FakeComfy(outputs={}))
    job = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "p"}).json()
    assert job["status"] == "failed" and job["error"]["code"] == "NO_OUTPUT"
    assert job["outputs"] == [] and job["metadata_path"] is None


def test_failed_download_is_not_claimed_as_output(client, monkeypatch):
    pid, wid = ready_workflow(client)
    use_comfy(monkeypatch, FakeComfy(download_error=True))
    job = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "p"}).json()
    assert job["status"] == "failed" and job["error"]["code"] == "DOWNLOAD_FAILED"
    assert job["outputs"] == []


def test_prompt_lost_from_comfyui(client, monkeypatch):
    pid, wid = ready_workflow(client)
    use_comfy(monkeypatch, FakeComfy(states=["missing"]))
    job = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "p"}).json()
    assert job["status"] == "failed" and job["error"]["code"] == "PROMPT_LOST"


def test_job_timeout_interrupts_and_fails(client, monkeypatch):
    monkeypatch.setenv("FILM_HOUSE_JOB_TIMEOUT_SECONDS", "0")
    pid, wid = ready_workflow(client)
    fake = use_comfy(monkeypatch, FakeComfy(states=["running"]))
    job = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "p"}).json()
    assert job["status"] == "failed" and job["error"]["code"] == "TIMEOUT"
    assert "FILM_HOUSE_JOB_TIMEOUT_SECONDS" in job["error"]["message"]
    assert fake.interrupted == 1 and job["outputs"] == []


def test_workflow_file_tampering_fails_the_job(client, monkeypatch, isolated_workspace):
    pid, wid = ready_workflow(client)
    use_comfy(monkeypatch, FakeComfy())
    record = client.get(f"/api/projects/{pid}/workflows/{wid}").json()["record"]
    path = isolated_workspace / pid / record["path"]
    path.chmod(0o666)
    path.write_text(json.dumps(WORKFLOW).replace("sd_xl_base_1.0", "swapped"), encoding="utf-8")
    response = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "p"})
    assert response.status_code == 500 and response.json()["code"] == "STORAGE_ERROR"
    assert client.get(f"/api/projects/{pid}/jobs").json()["jobs"] == []


# --------------------------------------------------------------------------- cancellation

def test_cancel_before_submission_never_queues(client, monkeypatch):
    pid, wid = ready_workflow(client)
    fake = use_comfy(monkeypatch, FakeComfy())
    monkeypatch.setattr(jobs.MANAGER, "inline", False)      # keep it in the local queue
    monkeypatch.setattr(jobs.MANAGER, "_ensure_worker", lambda: None)
    job = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "p"}).json()
    assert job["status"] == "queued"
    cancelled = client.post(f"/api/projects/{pid}/jobs/{job['id']}/cancel").json()
    assert cancelled["status"] == "cancelled" and cancelled["error"]["code"] == "CANCELLED"
    assert fake.submitted == [] and ("POST", "/prompt") not in fake.calls
    again = client.post(f"/api/projects/{pid}/jobs/{job['id']}/cancel")
    assert again.status_code == 409 and again.json()["detail"]["code"] == "JOB_FINISHED"


def test_cancel_running_job_interrupts_comfyui(client, monkeypatch):
    pid, wid = ready_workflow(client)
    fake = FakeComfy(states=["running"])
    use_comfy(monkeypatch, fake)
    monkeypatch.setattr(jobs.MANAGER, "inline", False)
    response = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "p"})
    job_id = response.json()["id"]
    for _ in range(200):                                     # wait until ComfyUI is executing it
        current = client.get(f"/api/projects/{pid}/jobs/{job_id}").json()
        if current["status"] == "running":
            break
    assert current["status"] == "running" and current["prompt_id"] == "prompt-1"
    client.post(f"/api/projects/{pid}/jobs/{job_id}/cancel")
    final = jobs.MANAGER.wait(pid, job_id, timeout=10.0)
    assert final.status == "cancelled" and final.outputs == []
    assert fake.interrupted >= 1
    assert any("cancel requested" in line for line in final.log)


def test_cancel_unknown_job(client):
    pid = make_project(client)
    assert client.post(f"/api/projects/{pid}/jobs/{'a' * 12}/cancel").status_code == 404


# --------------------------------------------------------------------------- persistence

def test_jobs_survive_a_restart_but_are_never_left_active(client, monkeypatch):
    pid, wid = ready_workflow(client)
    use_comfy(monkeypatch, FakeComfy())
    job_id = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "p"}).json()["id"]
    # Simulate a crash: the record is active on disk but no worker owns it.
    stored = storage.load_job(pid, job_id)
    stored.status = "running"
    stored.outputs = []
    storage.save_job(stored)
    reloaded = client.get(f"/api/projects/{pid}/jobs/{job_id}").json()
    assert reloaded["status"] == "failed" and reloaded["error"]["code"] == "INTERRUPTED"
    assert "no output was collected" in reloaded["error"]["message"]


def test_job_records_live_outside_project_json(client, monkeypatch, isolated_workspace):
    pid, wid = ready_workflow(client)
    use_comfy(monkeypatch, FakeComfy())
    job = client.post(f"/api/projects/{pid}/jobs", json={"workflow_id": wid, "prompt": "p"}).json()
    assert (isolated_workspace / pid / "jobs" / f"{job['id']}.json").exists()
    project = client.get(f"/api/projects/{pid}").json()
    assert "jobs" not in project and len(project["workflows"]) == 1
    # Editing the project while a job exists does not conflict.
    scene_id = project["scenes"][0]["id"]
    edit = client.patch(f"/api/projects/{pid}/scenes/{scene_id}",
                        json={"base_revision": project["revision"], "notes": "while generating"})
    assert edit.status_code == 200


def test_workflows_are_snapshotted_with_the_project(client, monkeypatch):
    pid, wid = ready_workflow(client)
    snapshot = client.post(f"/api/projects/{pid}/snapshots", json={"label": "with workflow"}).json()
    project = client.get(f"/api/projects/{pid}").json()
    restored = client.post(f"/api/projects/{pid}/snapshots/{snapshot['id']}/restore",
                           json={"base_revision": project["revision"]}).json()
    assert [w["id"] for w in restored["workflows"]] == [wid]
    assert restored["workflows"][0]["offline_validation"]["valid"] is True


# --------------------------------------------------------------------------- threaded worker

def test_threaded_worker_runs_jobs_serially(client, monkeypatch):
    pid, wid = ready_workflow(client)
    use_comfy(monkeypatch, FakeComfy())
    monkeypatch.setattr(jobs.MANAGER, "inline", False)
    ids = [client.post(f"/api/projects/{pid}/jobs",
                       json={"workflow_id": wid, "prompt": f"shot {i}"}).json()["id"] for i in range(3)]
    finished = [jobs.MANAGER.wait(pid, job_id, timeout=20.0) for job_id in ids]
    assert [job.status for job in finished] == ["succeeded"] * 3
    assert all(len(job.outputs) == 1 for job in finished)
    spans = sorted((job.started_at, job.finished_at) for job in finished)
    for earlier, later in zip(spans, spans[1:]):
        assert earlier[1] <= later[1]            # one at a time


def test_ui_page_is_served(client):
    page = client.get("/images")
    assert page.status_code == 200
    for marker in ("/api/comfyui/status", "workflow", "reference", "Cancel"):
        assert marker.lower() in page.text.lower()
    assert 'href="/images"' in client.get("/").text
