"""Serial generation-job manager for the local ComfyUI adapter.

One job runs at a time (ComfyUI holds the GPU; parallel jobs invite VRAM exhaustion). Jobs are
persisted as ``jobs/<job-id>.json`` inside the project, outside ``project.json``, so generation
never collides with the editor's revision checks.

A job only ever reports outputs that were actually downloaded, hashed and written to disk. If
ComfyUI is unavailable nothing is queued and nothing is claimed.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import random
import threading
import time
from datetime import datetime
from typing import Any, Callable

from app import comfy, storage, studio, workflows
from app.models import GenerationJob, GenerationRequest, JobOutput, Project, WorkflowRecord, utc_now

ACTIVE_STATUSES = ("queued", "submitted", "running")
MAX_LOG_LINES = 200


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def job_timeout_seconds() -> float:
    return _env_float("FILM_HOUSE_JOB_TIMEOUT_SECONDS", 900.0)


def poll_seconds() -> float:
    return _env_float("FILM_HOUSE_JOB_POLL_SECONDS", 1.5)


class JobError(Exception):
    def __init__(self, code: str, message: str, detail: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class JobManager:
    """Queues and runs image-generation jobs one at a time."""

    def __init__(self, client_factory: Callable[[], comfy.ComfyClient] | None = None, inline: bool = False):
        self.client_factory = client_factory or comfy.client_from_env
        self.inline = inline                      # run jobs synchronously (used by tests)
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.RLock()
        self._active: set[tuple[str, str]] = set()
        self._done = threading.Condition()

    # ------------------------------------------------------------------ public API

    def adapter(self) -> comfy.ComfyAdapter:
        return comfy.ComfyAdapter(self.client_factory())

    def preflight(self) -> dict[str, Any]:
        """Check a local ComfyUI is reachable. Never raises."""
        try:
            return comfy.ComfyAdapter(self.client_factory()).validate_config()
        except comfy.ComfyError as exc:
            return {"ok": False, **exc.to_dict()}
        except Exception as exc:  # a broken client must not take the API down
            return {"ok": False, "code": "CLIENT_ERROR", "message": f"{type(exc).__name__}: {exc}"[:200]}

    def submit(self, project: Project, record: WorkflowRecord, request: GenerationRequest) -> GenerationJob:
        """Validate everything, then queue one job. Raises JobError before any job is created."""
        if not record.offline_validation or not record.offline_validation.valid:
            raise JobError("WORKFLOW_NOT_VALIDATED",
                           "Map this workflow's nodes and validate it before generating.")
        mapping = record.mapping
        if mapping.prompt is None:
            raise JobError("WORKFLOW_NOT_VALIDATED", "This workflow has no prompt node mapped.")

        text = storage.read_workflow_file(project.id, record.path, record.sha256)
        workflow = workflows.load_api_workflow(text)
        problems = workflows.validate_mapping(workflow, mapping)
        if not problems["valid"]:
            raise JobError("MAPPING_STALE", "The stored mapping no longer fits this workflow. Re-map it.",
                           problems["errors"])

        for role, value in (("negative_prompt", request.negative_prompt), ("width", request.width),
                            ("height", request.height)):
            if value is not None and getattr(mapping, role) is None:
                raise JobError("ROLE_UNMAPPED",
                               f"This workflow has no {workflows.ROLE_SPECS[role]['label'].lower()} node mapped, "
                               f"so that value cannot be applied. Clear the field or map the node.")
        reference_path = None
        if request.reference_path or request.reference_asset_id:
            if mapping.reference_image is None:
                raise JobError(
                    "ROLE_UNMAPPED",
                    "This workflow has no reference image node mapped, so a reference cannot be used with it. "
                    "Not every workflow supports one: an image-to-image or IPAdapter workflow has a LoadImage "
                    "node you can map, a plain text-to-image workflow does not. Either map that node and "
                    "validate again, or generate without a reference.")
            if request.reference_asset_id:
                try:
                    resolved = studio.resolve_reference_asset(project, request.reference_asset_id,
                                                              request.allow_draft_reference)
                except studio.StudioError as exc:
                    raise JobError(exc.code, exc.message, exc.detail) from exc
                reference_path = storage.safe_path(project.id, resolved["path"])
            else:
                try:
                    reference_path = storage.resolve_project_image(project.id, request.reference_path or "")
                except ValueError as exc:
                    raise JobError("BAD_REFERENCE", str(exc)) from exc

        status = self.preflight()
        if not status.get("ok"):
            raise JobError("COMFYUI_UNAVAILABLE",
                           status.get("message", "ComfyUI is not available on this computer, so nothing was "
                                                 "queued and no image was generated."),
                           {"url": status.get("url")})

        seed_used: int | None = None
        if mapping.seed is not None:
            seed_used = (random.randrange(0, 2 ** 53) if request.randomize_seed or request.seed is None
                         else request.seed)

        job = GenerationJob(
            id=storage.short_id(), project_id=project.id, status="queued",
            comfyui_url=status.get("url"), comfyui_version=status.get("comfyui_version"),
            workflow_id=record.id, workflow_name=record.name, workflow_sha256=record.sha256,
            request=request, seed_used=seed_used, models=record.models,
        )
        job.log.append(f"{utc_now()} queued using workflow '{record.name}' (sha256 {record.sha256[:10]})")
        if seed_used is not None:
            job.log.append(f"{utc_now()} seed {seed_used}")
        storage.save_job(job)

        with self._lock:
            self._active.add((job.project_id, job.id))
        if self.inline:
            self._process(job.project_id, job.id, reference_path)
        else:
            self._queue.put((job.project_id, job.id))
            self._ensure_worker()
        return storage.load_job(job.project_id, job.id) or job

    def get(self, project_id: str, job_id: str) -> GenerationJob | None:
        job = storage.load_job(project_id, job_id)
        if job is not None:
            self._reconcile_one(job)
        return job

    def list(self, project_id: str) -> list[GenerationJob]:
        jobs = storage.list_jobs(project_id)
        for job in jobs:
            self._reconcile_one(job)
        return jobs

    def cancel(self, project_id: str, job_id: str) -> GenerationJob:
        job = storage.load_job(project_id, job_id)
        if job is None:
            raise JobError("JOB_NOT_FOUND", "Job not found")
        if job.status not in ACTIVE_STATUSES:
            raise JobError("JOB_FINISHED", f"This job is already {job.status}.")
        job.cancel_requested = True
        job.log.append(f"{utc_now()} cancel requested")
        if job.prompt_id:
            try:
                result = self.adapter().cancel(job.prompt_id)
                job.log.append(f"{utc_now()} {result['how']}")
            except comfy.ComfyError as exc:
                job.log.append(f"{utc_now()} cancel could not reach ComfyUI: {exc.message}")
        if job.status == "queued" and not job.prompt_id:
            # Not sent to ComfyUI yet: finish it here so nothing is ever queued.
            self._finish(job, "cancelled", error={"code": "CANCELLED", "message": "Cancelled before it was queued."})
        else:
            self._persist(job)
        return job

    def wait(self, project_id: str, job_id: str, timeout: float = 30.0) -> GenerationJob | None:
        """Block until a job leaves the active states (used by tests and the live smoke test)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = storage.load_job(project_id, job_id)
            if job is None or job.status not in ACTIVE_STATUSES:
                return job
            with self._done:
                self._done.wait(0.05)
        return storage.load_job(project_id, job_id)

    # ------------------------------------------------------------------ worker

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="filmhouse-jobs", daemon=True)
                self._worker.start()

    def _run(self) -> None:
        while True:
            try:
                project_id, job_id = self._queue.get(timeout=30.0)
            except queue.Empty:
                return
            try:
                self._process(project_id, job_id, None)
            except Exception as exc:  # never let the worker die
                job = storage.load_job(project_id, job_id)
                if job is not None and job.status in ACTIVE_STATUSES:
                    self._finish(job, "failed", error={"code": "WORKER_ERROR",
                                                       "message": f"{type(exc).__name__}: {exc}"[:300]})
            finally:
                self._queue.task_done()

    def _notify(self) -> None:
        with self._done:
            self._done.notify_all()

    def _persist(self, job: GenerationJob) -> GenerationJob:
        """Save worker progress without losing a cancel request (or log line) written elsewhere."""
        with storage.project_lock(job.project_id):
            disk = storage.load_job(job.project_id, job.id)
            if disk is not None:
                job.cancel_requested = job.cancel_requested or disk.cancel_requested
                if set(disk.log) - set(job.log):
                    # Log lines start with an ISO timestamp, so sorting keeps them chronological.
                    job.log = sorted(set(job.log) | set(disk.log))
            del job.log[:-MAX_LOG_LINES]
            storage.save_job(job)
        return job

    def _log(self, job: GenerationJob, message: str) -> None:
        job.log.append(f"{utc_now()} {message}")
        self._persist(job)

    def _finish(self, job: GenerationJob, status: str, error: dict[str, Any] | None = None) -> GenerationJob:
        job.status = status                                     # type: ignore[assignment]
        job.error = error
        job.finished_at = utc_now()
        if job.started_at:
            try:
                started = datetime.fromisoformat(job.started_at)
                job.duration_seconds = round(max(0.0, (datetime.fromisoformat(job.finished_at)
                                                       - started).total_seconds()), 1)
            except ValueError:
                job.duration_seconds = None
        job.log.append(f"{utc_now()} {status}" + (f": {error.get('message', '')}" if error else ""))
        self._persist(job)
        with self._lock:
            self._active.discard((job.project_id, job.id))
        self._notify()
        return job

    def _reconcile_one(self, job: GenerationJob) -> None:
        """A job left active by a crashed/restarted server is not running anywhere: mark it failed."""
        if job.status not in ACTIVE_STATUSES:
            return
        with self._lock:
            if (job.project_id, job.id) in self._active:
                return
        self._finish(job, "failed", error={
            "code": "INTERRUPTED",
            "message": "The server stopped while this job was active, so no output was collected. "
                       "Check ComfyUI's own queue/history, then retry.",
        })

    # ------------------------------------------------------------------ one job

    def _process(self, project_id: str, job_id: str, reference_path: Any = None) -> None:
        job = storage.load_job(project_id, job_id)
        if job is None:
            return
        if job.cancel_requested:
            self._finish(job, "cancelled", error={"code": "CANCELLED", "message": "Cancelled before it was queued."})
            return
        job.started_at = utc_now()
        job.status = "submitted"
        self._persist(job)
        try:
            client = self.client_factory()
            adapter = comfy.ComfyAdapter(client)
            prompt = self._build_prompt(job, client, reference_path)
            job.prompt_id = adapter.submit(prompt)
            self._log(job, f"queued in ComfyUI as prompt {job.prompt_id}")
            self._poll(job, adapter)
        except comfy.ComfyError as exc:
            self._finish(job, "failed", error=exc.to_dict())
        except (workflows.WorkflowError, JobError) as exc:
            self._finish(job, "failed", error=exc.to_dict())
        except (OSError, ValueError) as exc:
            self._finish(job, "failed", error={"code": "LOCAL_ERROR", "message": f"{type(exc).__name__}: {exc}"[:300]})
        finally:
            with self._lock:
                self._active.discard((project_id, job_id))
            self._notify()

    def _build_prompt(self, job: GenerationJob, client: comfy.ComfyClient, reference_path: Any) -> dict[str, Any]:
        project = storage.load_project(job.project_id)
        if project is None:
            raise JobError("PROJECT_MISSING", "The project no longer exists.")
        record = next((w for w in project.workflows if w.id == job.workflow_id), None)
        if record is None or record.sha256 != job.workflow_sha256:
            raise JobError("WORKFLOW_CHANGED", "The workflow this job was created from is no longer available "
                                               "under the same SHA-256.")
        workflow = workflows.load_api_workflow(
            storage.read_workflow_file(job.project_id, record.path, record.sha256))
        values: dict[str, Any] = {"prompt": job.request.prompt}
        if job.request.negative_prompt is not None:
            values["negative_prompt"] = job.request.negative_prompt
        if job.seed_used is not None:
            values["seed"] = job.seed_used
        if job.request.width is not None:
            values["width"] = job.request.width
        if job.request.height is not None:
            values["height"] = job.request.height
        if job.request.reference_path or job.request.reference_asset_id:
            path = reference_path
            if path is None:
                if job.request.reference_asset_id:
                    resolved = studio.resolve_reference_asset(project, job.request.reference_asset_id,
                                                              job.request.allow_draft_reference)
                    path = storage.safe_path(job.project_id, resolved["path"])
                else:
                    path = storage.resolve_project_image(job.project_id, job.request.reference_path or "")
            uploaded = client.upload_image(path)
            job.reference_uploaded_as = uploaded["name"]
            values["reference_image"] = uploaded["name"]
            self._log(job, f"uploaded reference image as '{uploaded['name']}'"
                           + (f" (library asset {job.request.reference_asset_id})"
                              if job.request.reference_asset_id else ""))
        return workflows.apply_mapping(workflow, record.mapping, values)

    def _poll(self, job: GenerationJob, adapter: comfy.ComfyAdapter) -> None:
        deadline = time.monotonic() + job_timeout_seconds()
        interval = poll_seconds()
        seen_running = False
        while True:
            fresh = storage.load_job(job.project_id, job.id)
            if fresh is not None and fresh.cancel_requested:
                job.cancel_requested = True
            if job.cancel_requested:
                try:
                    result = adapter.cancel(job.prompt_id or "")
                    self._finish(job, "cancelled", error={"code": "CANCELLED",
                                                          "message": f"Cancelled: {result['how']}."})
                except comfy.ComfyError as exc:
                    self._finish(job, "cancelled", error={"code": "CANCELLED",
                                                          "message": f"Cancel requested; ComfyUI said: "
                                                                     f"{exc.message}"})
                return
            if time.monotonic() > deadline:
                try:
                    adapter.cancel(job.prompt_id or "")
                except comfy.ComfyError:
                    pass
                self._finish(job, "failed", error={
                    "code": "TIMEOUT",
                    "message": f"ComfyUI did not finish within {job_timeout_seconds():g}s. The prompt was "
                               "interrupted; nothing was collected. Raise FILM_HOUSE_JOB_TIMEOUT_SECONDS for slow "
                               "models or simplify the workflow."})
                return

            state = adapter.status(job.prompt_id or "")
            if state["state"] == "pending":
                if job.queue_position != state.get("queue_position"):
                    job.queue_position = state.get("queue_position")
                    self._log(job, f"waiting in the ComfyUI queue (position {job.queue_position})")
            elif state["state"] == "running":
                if not seen_running:
                    seen_running = True
                    job.status = "running"
                    job.queue_position = 0
                    self._log(job, "ComfyUI is executing the workflow")
            elif state["state"] == "done":
                self._collect(job, adapter, state)
                return
            elif state["state"] == "missing":
                self._finish(job, "failed", error={
                    "code": "PROMPT_LOST",
                    "message": "ComfyUI no longer knows this prompt (its queue or history was cleared, or it "
                               "restarted). No output was collected."})
                return
            time.sleep(interval)

    def _collect(self, job: GenerationJob, adapter: comfy.ComfyAdapter, state: dict[str, Any]) -> None:
        status_str = state.get("status_str")
        if status_str == "error":
            self._finish(job, "failed", error={
                "code": "EXECUTION_ERROR",
                "message": "ComfyUI failed while running the workflow: " + _history_error_text(state),
                "detail": state.get("messages")})
            return
        descriptors = adapter.collect_outputs(job.prompt_id or "")
        allowed = set(job_output_nodes(job))
        if allowed:
            descriptors = [d for d in descriptors if d["node_id"] in allowed] or descriptors
        if not descriptors:
            self._finish(job, "failed", error={
                "code": "NO_OUTPUT",
                "message": "ComfyUI finished but recorded no image files for this prompt, so there is nothing to "
                           "show. Make sure the workflow has a SaveImage node."})
            return

        directory = storage.output_dir(job.project_id, job)
        directory.mkdir(parents=True, exist_ok=True)
        outputs: list[JobOutput] = []
        for index, descriptor in enumerate(descriptors):
            try:
                data = adapter.client.download_output(descriptor["filename"], descriptor["subfolder"],
                                                     descriptor["type"])
            except comfy.ComfyError as exc:
                self._log(job, f"could not download '{descriptor['filename']}': {exc.message}")
                continue
            name = f"{index:02d}_{_safe_name(descriptor['filename'])}"
            target = directory / name
            target.write_bytes(data)
            outputs.append(JobOutput(
                index=index, path=target.relative_to(storage.project_dir(job.project_id)).as_posix(),
                filename=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                node_id=descriptor["node_id"], comfy_subfolder=descriptor["subfolder"],
                comfy_type=descriptor["type"]))
        if not outputs:
            self._finish(job, "failed", error={
                "code": "DOWNLOAD_FAILED",
                "message": "ComfyUI listed output images but none could be downloaded, so nothing was saved."})
            return
        job.outputs = outputs
        job.metadata_path = self._write_metadata(job, directory)
        self._log(job, f"saved {len(outputs)} image file(s) to {directory.name}/")
        try:
            storage.write_manifest(job.project_id)
        except (OSError, ValueError):
            pass
        # Finish last: once the status leaves the active set, callers may assume all writes are done.
        self._finish(job, "succeeded")

    def _write_metadata(self, job: GenerationJob, directory: Any) -> str:
        project = storage.load_project(job.project_id)
        record = next((w for w in (project.workflows if project else []) if w.id == job.workflow_id), None)
        payload = {
            "job": json.loads(job.model_dump_json()),
            "workflow": {"id": job.workflow_id, "name": job.workflow_name, "sha256": job.workflow_sha256,
                         "path": record.path if record else None,
                         "mapping": json.loads(record.mapping.model_dump_json()) if record else None},
            "provider": {"name": job.provider, "url": job.comfyui_url, "version": job.comfyui_version},
            "reproduce": {"seed": job.seed_used, "prompt": job.request.prompt,
                          "negative_prompt": job.request.negative_prompt,
                          "width": job.request.width, "height": job.request.height,
                          "reference_path": job.request.reference_path,
                          "reference_asset_id": job.request.reference_asset_id,
                          "models": job.models},
            "note": "Written locally by AI Film House. Model licences are your responsibility; see "
                    "docs/MODEL_SELECTION.md.",
        }
        path = directory / "metadata.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path.relative_to(storage.project_dir(job.project_id)).as_posix()


def job_output_nodes(job: GenerationJob) -> list[str]:
    project = storage.load_project(job.project_id)
    record = next((w for w in (project.workflows if project else []) if w.id == job.workflow_id), None)
    return list(record.mapping.output_nodes) if record else []


def _safe_name(filename: str) -> str:
    name = os.path.basename(str(filename))[-120:]
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name).lstrip(".")
    return cleaned or "output.png"


def _history_error_text(state: dict[str, Any]) -> str:
    messages = state.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, (list, tuple)) and len(message) > 1 and "error" in str(message[0]):
            detail = message[1] if isinstance(message[1], dict) else {}
            parts = [str(detail.get("exception_message") or detail.get("exception_type") or message[0])]
            if detail.get("node_type"):
                parts.append(f"(node {detail.get('node_id')} {detail['node_type']})")
            return " ".join(parts)[:400]
    return "no details were recorded in ComfyUI's history."


MANAGER = JobManager()
