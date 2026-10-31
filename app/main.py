from __future__ import annotations

import csv
import hashlib
import io
import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.comfy import ComfyAdapter, ComfyError
from app.assets import ALLOWED_FORMATS, ImageError, decode_base64, inspect_image, max_bytes, read_source_file
from app.jobs import ACTIVE_STATUSES, MANAGER, JobError, job_timeout_seconds
from app.models import (
    AssetAssociation, AssetDelete, AssetEdit, AssetFromJob, AssetImport, AssetStateChange, BreakdownRequest,
    CanonicalSelect, Character, CharacterCreate, CharacterEdit, GenerationRequest, JobAction, LocationCreate,
    LocationEdit, MappingUpdate, Project, ProjectCreate, PropCreate, PropEdit, ReferenceAsset, RevisionOnly,
    SceneAssociations, SceneEdit, ScriptUpdate, ShotEdit, SnapshotCreate, ValidationProblem, WarningAck,
    WardrobeCreate, WardrobeEdit, WorkflowImport, WorkflowMapping, WorkflowRecord, WorkflowValidation, utc_now,
)
from app.screenplay import build_draft_shot_plan, parse_screenplay_document
from app.services import check_services
from app.hardware import get_report
from app.storage import (
    ROOT, RevisionConflict, StorageError, add_script_version, add_workflow_file, check_revision, create_snapshot,
    delete_asset_file, list_projects, list_reference_images, list_snapshots, load_project, load_snapshot,
    maybe_autosnapshot, project_lock, read_asset_file, read_script_version, read_workflow_file, safe_path,
    save_project, short_id, write_asset_file, write_manifest,
)
from app.studio import (
    StudioError, apply_edit, asset_view, assert_deletable, change_state, clear_canonical, create_character,
    create_location, create_prop, create_wardrobe, delete_asset, delete_subject, delete_wardrobe, duplicate_of,
    find_wardrobe, next_version, record_asset, references_to, require_asset, require_subject, set_canonical,
    set_scene_associations, studio_overview, validate_association,
)
from app.validation import summarize, validate_project
from app.workflows import (
    WorkflowError, inspect_workflow, load_api_workflow, role_specs, suggest_mapping, validate_against_server,
    validate_mapping,
)

VERSION = "0.4.0"

app = FastAPI(
    title="AI Film House — Local Production Engine",
    description="A local-first foundation for screenplay-to-film production workflows.",
    version=VERSION,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(RevisionConflict)
def _revision_conflict(_request: Request, exc: RevisionConflict) -> JSONResponse:
    return JSONResponse(status_code=409, content={
        "detail": str(exc), "code": "REVISION_CONFLICT", "current_revision": exc.current,
    })


@app.exception_handler(StorageError)
def _storage_error(_request: Request, exc: StorageError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc), "code": "STORAGE_ERROR"})


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": "ai-film-house", "version": VERSION}


@app.get("/api/services")
def services() -> dict:
    return check_services()


@app.get("/api/diagnostics")
def diagnostics(refresh: bool = False) -> dict:
    """Local hardware & model-runtime report. Read-only; nothing is downloaded or installed."""
    return get_report(refresh=refresh)


@app.get("/diagnostics", include_in_schema=False)
def diagnostics_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "diagnostics.html")


@app.get("/api/projects")
def projects() -> list[dict]:
    return list_projects()


@app.post("/api/projects", status_code=201)
def create_project(data: ProjectCreate) -> Project:
    project = Project(id=uuid.uuid4().hex, title=data.title.strip())
    save_project(project)
    return project


# --------------------------------------------------------------------------- helpers

def _get_or_404(project_id: str) -> Project:
    project = load_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.script.strip() and not project.script_versions:
        # Phase 0 project: preserve its script as the immutable original before anything else.
        with project_lock(project_id):
            project = load_project(project_id) or project
            if not project.script_versions:
                add_script_version(project, project.script)
                project.breakdown_script_version = 1 if project.scenes else None
                save_project(project)
                write_manifest(project_id)
    return project


@contextmanager
def _editing(project_id: str, base_revision: int | None, autosnapshot: bool = True) -> Iterator[Project]:
    """Load → check revision → mutate → save atomically, all under the project lock."""
    with project_lock(project_id):
        project = _get_or_404(project_id)
        check_revision(project, base_revision)
        yield project
        save_project(project)
        if autosnapshot:
            maybe_autosnapshot(project)


def _find(items: list, item_id: str, label: str):
    for item in items:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail=f"{label} not found")


def _clean_names(names: list[str]) -> list[str]:
    result: list[str] = []
    for name in names:
        value = " ".join(str(name).split())[:80]
        if value and value not in result:
            result.append(value)
    return result


# --------------------------------------------------------------------------- project & script

@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> Project:
    return _get_or_404(project_id)


@app.put("/api/projects/{project_id}/script")
def update_script(project_id: str, data: ScriptUpdate) -> Project:
    """Import or edit the screenplay. Never overwrites: every change becomes a new script version."""
    text = data.script.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise HTTPException(status_code=422, detail="The screenplay is empty")
    with project_lock(project_id):
        project = _get_or_404(project_id)
        check_revision(project, data.base_revision)
        if project.scenes and project.script != text:
            create_snapshot(project, "pre-script-edit")
        version = add_script_version(project, text, data.note.strip())
        project.aspect_ratio = data.aspect_ratio
        project.visual_style = data.visual_style.strip() or "cinematic live-action realism"
        project.language = data.language.strip() or "English"
        if not project.scenes:
            project.status = "script_ready"
        if version is not None:
            project.production_notes = [
                f"Saved screenplay as version {version.number}"
                + (" (original import, read-only)." if version.number == 1 else "; earlier versions are kept."),
                "The screenplay parser uses common screenplay conventions and may need manual correction.",
                "All suggested shots are drafts. Approve the shot list before generating media.",
            ]
        save_project(project)
        write_manifest(project_id)
        return project


@app.get("/api/projects/{project_id}/script/versions")
def script_versions(project_id: str) -> list[dict]:
    project = _get_or_404(project_id)
    return [v.model_dump() | {"current": v.number == project.current_script_version,
                              "used_by_breakdown": v.number == project.breakdown_script_version}
            for v in project.script_versions]


@app.get("/api/projects/{project_id}/script/versions/{number}", response_class=PlainTextResponse)
def script_version_text(project_id: str, number: int) -> str:
    project = _get_or_404(project_id)
    try:
        return read_script_version(project, number)
    except KeyError:
        raise HTTPException(status_code=404, detail="Script version not found")


@app.get("/api/projects/{project_id}/script/original", response_class=PlainTextResponse)
def script_original(project_id: str) -> str:
    project = _get_or_404(project_id)
    if not project.script_versions:
        raise HTTPException(status_code=404, detail="No screenplay imported yet")
    return read_script_version(project, project.script_versions[0].number)


# --------------------------------------------------------------------------- breakdown

def _merge_characters(old: list[Character], new: list[Character]) -> list[Character]:
    """Keep bible fields the user wrote when the same cue appears in a re-parse."""
    by_key = {(c.cue_name or c.name.upper()): c for c in old}
    merged: list[Character] = []
    for fresh in new:
        previous = by_key.pop(fresh.cue_name, None)
        if previous is None:
            merged.append(fresh)
            continue
        merged.append(fresh.model_copy(update={
            "id": previous.id, "name": previous.name, "description": previous.description,
            "visual_signature": previous.visual_signature, "voice_direction": previous.voice_direction,
            "notes": previous.notes, "reference_images": previous.reference_images,
            "manually_edited": previous.manually_edited,
        }))
    # Characters the user edited but who no longer have cues are kept, not silently dropped.
    merged.extend(c.model_copy(update={"scene_numbers": [], "dialogue_count": 0})
                  for c in by_key.values() if c.manually_edited)
    return merged


@app.post("/api/projects/{project_id}/breakdown")
def breakdown(project_id: str, data: BreakdownRequest | None = None) -> Project:
    data = data or BreakdownRequest()
    with project_lock(project_id):
        project = _get_or_404(project_id)
        check_revision(project, data.base_revision)
        if not project.script.strip():
            raise HTTPException(status_code=400, detail="Add a screenplay before building the production plan")
        edited = sum(s.manually_edited for s in project.scenes) + sum(s.manually_edited for s in project.shots)
        if edited and not data.force:
            raise HTTPException(status_code=409, detail={
                "code": "BREAKDOWN_WOULD_REPLACE_EDITS",
                "message": f"{edited} manually edited scene/shot record(s) would be replaced. A snapshot will be "
                           "saved first; resend with force=true to continue. Character bible edits are kept.",
            })
        parsed = parse_screenplay_document(project.script)
        if not parsed.scenes:
            raise HTTPException(status_code=400, detail={
                "code": "NOTHING_PARSED", "message": "No screenplay content could be parsed",
                "warnings": [w.model_dump() for w in parsed.warnings],
            })
        if project.scenes:
            create_snapshot(project, "pre-breakdown")
        project.scenes = parsed.scenes
        project.characters = _merge_characters(project.characters, parsed.characters)
        project.shots = build_draft_shot_plan(parsed.scenes, project.visual_style, project.aspect_ratio)
        project.parse_warnings = parsed.warnings
        project.breakdown_script_version = project.current_script_version
        if parsed.title_page:
            project.settings["title_page"] = parsed.title_page
        project.status = "plan_ready"
        project.production_notes = [
            f"Parsed script v{project.current_script_version}: {len(parsed.scenes)} scene(s), "
            f"{len(project.characters)} character(s), {len(parsed.warnings)} parser warning(s).",
            f"Created {len(project.shots)} editable draft shots. These are not generated video clips.",
            "Confirm character appearance, location continuity, scene geography, shot order and duration before generation.",
        ]
        save_project(project)
        create_snapshot(project, "breakdown")
        write_manifest(project_id)
        return project


# --------------------------------------------------------------------------- record editing (autosave targets)

@app.patch("/api/projects/{project_id}/scenes/{scene_id}")
def edit_scene(project_id: str, scene_id: str, data: SceneEdit) -> Project:
    changes = data.model_dump(exclude_unset=True, exclude={"base_revision"})
    with _editing(project_id, data.base_revision) as project:
        scene = _find(project.scenes, scene_id, "Scene")
        if "characters" in changes:
            changes["characters"] = _clean_names(changes["characters"] or [])
        for key, value in changes.items():
            setattr(scene, key, value.strip() if isinstance(value, str) else value)
        scene.manually_edited = True
    return project


@app.patch("/api/projects/{project_id}/characters/{character_id}")
def edit_character(project_id: str, character_id: str, data: CharacterEdit) -> Project:
    changes = data.model_dump(exclude_unset=True, exclude={"base_revision"})
    with _editing(project_id, data.base_revision) as project:
        character = _find(project.characters, character_id, "Character")
        new_name = " ".join((changes.pop("name", None) or character.name).split())
        if new_name != character.name:
            if any(c.name.upper() == new_name.upper() and c.id != character.id for c in project.characters):
                raise HTTPException(status_code=400, detail=f"Another character is already named '{new_name}'")
            old_name = character.name
            # Keep scene casts, shots and parsed dialogue pointing at the renamed record.
            for scene in project.scenes:
                scene.characters = [new_name if n == old_name else n for n in scene.characters]
                for element in scene.elements:
                    if element.character == old_name:
                        element.character = new_name
            for shot in project.shots:
                shot.characters = [new_name if n == old_name else n for n in shot.characters]
            character.name = new_name
        for key, value in changes.items():
            setattr(character, key, value.strip() if isinstance(value, str) else value)
        character.manually_edited = True
    return project


@app.patch("/api/projects/{project_id}/shots/{shot_id}")
def edit_shot(project_id: str, shot_id: str, data: ShotEdit) -> Project:
    changes = data.model_dump(exclude_unset=True, exclude={"base_revision"})
    with _editing(project_id, data.base_revision) as project:
        shot = _find(project.shots, shot_id, "Shot")
        if "characters" in changes:
            changes["characters"] = _clean_names(changes["characters"] or [])
        if changes.get("duration_seconds") is not None:
            changes["duration_seconds"] = round(float(changes["duration_seconds"]), 2)
        for key, value in changes.items():
            if value is None:
                continue
            setattr(shot, key, value.strip() if isinstance(value, str) else value)
        shot.manually_edited = True
        shot.updated_at = utc_now()
    return project


# --------------------------------------------------------------------------- warnings

@app.get("/api/projects/{project_id}/warnings")
def warnings(project_id: str) -> dict:
    project = _get_or_404(project_id)
    items = validate_project(project)
    return {"revision": project.revision, "summary": summarize(items), "warnings": [w.model_dump() for w in items]}


@app.post("/api/projects/{project_id}/warnings/{warning_id}/ack")
def acknowledge_warning(project_id: str, warning_id: str, data: WarningAck) -> dict:
    with _editing(project_id, data.base_revision) as project:
        if warning_id not in {w.id for w in validate_project(project)}:
            raise HTTPException(status_code=404, detail="Warning not found")
        acked = [w for w in project.acknowledged_warnings if w != warning_id]
        if data.acknowledged:
            acked.append(warning_id)
        project.acknowledged_warnings = acked
    items = validate_project(project)
    return {"revision": project.revision, "summary": summarize(items), "warnings": [w.model_dump() for w in items]}


# --------------------------------------------------------------------------- snapshots & manifest

@app.get("/api/projects/{project_id}/snapshots")
def snapshots(project_id: str) -> list[dict]:
    _get_or_404(project_id)
    return [s.model_dump() for s in reversed(list_snapshots(project_id))]


@app.post("/api/projects/{project_id}/snapshots", status_code=201)
def make_snapshot(project_id: str, data: SnapshotCreate) -> dict:
    with project_lock(project_id):
        project = _get_or_404(project_id)
        info = create_snapshot(project, "manual", data.label)
        write_manifest(project_id)
        return info.model_dump()


@app.post("/api/projects/{project_id}/snapshots/{snapshot_id}/restore")
def restore_snapshot(project_id: str, snapshot_id: str, data: RevisionOnly) -> Project:
    """Restore production data from a snapshot. The current state is snapshotted first."""
    with project_lock(project_id):
        project = _get_or_404(project_id)
        check_revision(project, data.base_revision)
        try:
            snap = load_snapshot(project_id, snapshot_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        create_snapshot(project, "pre-restore")
        restored = snap.model_copy(update={
            "revision": project.revision,
            # Script versions are append-only files; restoring never forgets a version.
            "script_versions": project.script_versions,
            "production_notes": [f"Restored production data from snapshot {snapshot_id} "
                                 f"(revision {snap.revision})."] + snap.production_notes[:3],
        })
        if restored.current_script_version:
            restored.script = read_script_version(project, restored.current_script_version)
        save_project(restored)
        write_manifest(project_id)
        return restored


@app.get("/api/projects/{project_id}/manifest")
def manifest(project_id: str) -> dict:
    _get_or_404(project_id)
    return write_manifest(project_id)


# --------------------------------------------------------------------------- Phase 4: ComfyUI workflows

@app.get("/api/comfyui/status")
def comfyui_status() -> dict:
    """Is a local ComfyUI usable? Also reports what this adapter can and cannot do."""
    return {"capabilities": ComfyAdapter().capabilities(), "config": MANAGER.preflight(),
            "roles": role_specs(),
            "job_timeout_seconds": job_timeout_seconds()}


def _workflow_or_404(project: Project, workflow_id: str) -> WorkflowRecord:
    record = next((w for w in project.workflows if w.id == workflow_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return record


def _workflow_detail(project: Project, record: WorkflowRecord) -> dict:
    text = read_workflow_file(project.id, record.path, record.sha256)
    workflow = load_api_workflow(text)
    offline = validate_mapping(workflow, record.mapping)
    return {
        "record": record.model_dump(),
        "inspection": inspect_workflow(workflow),
        "suggestions": suggest_mapping(workflow),
        "current_validation": offline,
        "roles": role_specs(),
    }


@app.post("/api/projects/{project_id}/workflows", status_code=201)
def import_workflow(project_id: str, data: WorkflowImport) -> dict:
    """Import a ComfyUI API-format workflow. Stored write-once and pinned by SHA-256."""
    try:
        workflow = load_api_workflow(data.workflow)
    except WorkflowError as exc:
        raise HTTPException(status_code=422, detail=exc.to_dict())
    inspection = inspect_workflow(workflow)
    with project_lock(project_id):
        project = _get_or_404(project_id)
        check_revision(project, data.base_revision)
        text = json.dumps(workflow, indent=2, sort_keys=True)
        sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = next((w for w in project.workflows if w.sha256 == sha256), None)
        if existing is not None:
            raise HTTPException(status_code=409, detail={
                "code": "WORKFLOW_ALREADY_IMPORTED",
                "message": f"This exact workflow is already imported as '{existing.name}'.",
                "workflow_id": existing.id})
        workflow_id = short_id()
        relative, stored_sha = add_workflow_file(project_id, workflow_id, text)
        record = WorkflowRecord(
            id=workflow_id, name=data.name.strip(), path=relative, sha256=stored_sha,
            source_filename=data.source_filename.strip()[:260], node_count=inspection["node_count"],
            class_types=inspection["class_types"], models=inspection["models"],
            mapping=WorkflowMapping(output_nodes=inspection["saving_output_nodes"] or inspection["output_nodes"]),
        )
        record.offline_validation = WorkflowValidation(
            scope="offline", **_validation_fields(validate_mapping(workflow, record.mapping)))
        project.workflows.append(record)
        save_project(project)
        write_manifest(project_id)
        return _workflow_detail(project, record) | {"revision": project.revision}


def _validation_fields(result: dict) -> dict:
    return {"valid": bool(result["valid"]),
            "errors": [ValidationProblem(**problem) for problem in result["errors"]],
            "warnings": [ValidationProblem(**problem) for problem in result["warnings"]]}


@app.get("/api/projects/{project_id}/workflows")
def list_workflows(project_id: str) -> dict:
    project = _get_or_404(project_id)
    return {"revision": project.revision, "workflows": [
        w.model_dump() | {"ready": w.ready} for w in project.workflows]}


@app.get("/api/projects/{project_id}/workflows/{workflow_id}")
def get_workflow(project_id: str, workflow_id: str) -> dict:
    project = _get_or_404(project_id)
    record = _workflow_or_404(project, workflow_id)
    try:
        return _workflow_detail(project, record) | {"revision": project.revision}
    except WorkflowError as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict())


@app.get("/api/projects/{project_id}/workflows/{workflow_id}/json")
def get_workflow_json(project_id: str, workflow_id: str) -> dict:
    project = _get_or_404(project_id)
    record = _workflow_or_404(project, workflow_id)
    return json.loads(read_workflow_file(project_id, record.path, record.sha256))


@app.put("/api/projects/{project_id}/workflows/{workflow_id}/mapping")
def set_workflow_mapping(project_id: str, workflow_id: str, data: MappingUpdate) -> dict:
    """Set the explicit node mapping and run offline validation."""
    with project_lock(project_id):
        project = _get_or_404(project_id)
        check_revision(project, data.base_revision)
        record = _workflow_or_404(project, workflow_id)
        workflow = load_api_workflow(read_workflow_file(project_id, record.path, record.sha256))
        result = validate_mapping(workflow, data.mapping)
        record.mapping = data.mapping
        record.mapping_set_at = utc_now()
        record.offline_validation = WorkflowValidation(scope="offline", **_validation_fields(result))
        record.live_validation = None       # a changed mapping invalidates the live check
        save_project(project)
        return _workflow_detail(project, record) | {"revision": project.revision}


@app.post("/api/projects/{project_id}/workflows/{workflow_id}/validate")
def validate_workflow_live(project_id: str, workflow_id: str) -> dict:
    """Check the mapping against the running ComfyUI: node classes, inputs and model files."""
    project = _get_or_404(project_id)
    record = _workflow_or_404(project, workflow_id)
    workflow = load_api_workflow(read_workflow_file(project_id, record.path, record.sha256))
    offline = validate_mapping(workflow, record.mapping)
    config = MANAGER.preflight()
    if not config.get("ok"):
        raise HTTPException(status_code=503, detail={
            "code": "COMFYUI_UNAVAILABLE",
            "message": config.get("message", "ComfyUI is not reachable on this computer."),
            "offline_validation": offline})
    client = MANAGER.client_factory()
    try:
        object_info = client.object_info()
        installed = {folder: client.models(folder)
                     for folder in sorted({m["folder"] for m in record.models if m["folder"]})}
    except ComfyError as exc:
        raise HTTPException(status_code=503, detail=exc.to_dict())
    live = validate_against_server(workflow, record.mapping, object_info, installed)
    with project_lock(project_id):
        project = _get_or_404(project_id)
        record = _workflow_or_404(project, workflow_id)
        record.offline_validation = WorkflowValidation(scope="offline", **_validation_fields(offline))
        record.live_validation = WorkflowValidation(
            scope="live", comfyui_version=config.get("comfyui_version"), comfyui_url=config.get("url"),
            **_validation_fields(live))
        save_project(project)
        return _workflow_detail(project, record) | {"revision": project.revision}


# --------------------------------------------------------------------------- Phase 4: generation jobs

@app.get("/api/projects/{project_id}/reference-images")
def reference_images(project_id: str) -> dict:
    """Images in this project that can be used as a workflow reference."""
    project = _get_or_404(project_id)
    assets = [asset_view(project, asset) for asset in project.assets]
    return {"images": list_reference_images(project_id),
            "assets": [a for a in assets if a["file_status"] == "ok" and a["state"] != "rejected"],
            "hint": "Pick an approved image from the Studio library, or put files in references/input/ "
                    "inside the project folder."}


@app.post("/api/projects/{project_id}/jobs", status_code=201)
def create_job(project_id: str, data: GenerationRequest) -> dict:
    """Queue one image generation. Nothing is queued unless a local ComfyUI answers first."""
    project = _get_or_404(project_id)
    record = _workflow_or_404(project, data.workflow_id)
    try:
        job = MANAGER.submit(project, record, data)
    except JobError as exc:
        status = 503 if exc.code == "COMFYUI_UNAVAILABLE" else 409
        raise HTTPException(status_code=status, detail=exc.to_dict())
    except WorkflowError as exc:
        raise HTTPException(status_code=422, detail=exc.to_dict())
    except ComfyError as exc:
        raise HTTPException(status_code=503, detail=exc.to_dict())
    return job.model_dump()


@app.get("/api/projects/{project_id}/jobs")
def list_generation_jobs(project_id: str) -> dict:
    _get_or_404(project_id)
    jobs = MANAGER.list(project_id)
    return {"jobs": [job.model_dump() for job in jobs],
            "active": sum(job.status in ACTIVE_STATUSES for job in jobs)}


@app.get("/api/projects/{project_id}/jobs/{job_id}")
def get_generation_job(project_id: str, job_id: str) -> dict:
    _get_or_404(project_id)
    job = MANAGER.get(project_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.model_dump()


@app.post("/api/projects/{project_id}/jobs/{job_id}/cancel")
def cancel_generation_job(project_id: str, job_id: str) -> dict:
    _get_or_404(project_id)
    try:
        return MANAGER.cancel(project_id, job_id).model_dump()
    except JobError as exc:
        raise HTTPException(status_code=404 if exc.code == "JOB_NOT_FOUND" else 409, detail=exc.to_dict())


@app.post("/api/projects/{project_id}/jobs/{job_id}/retry", status_code=201)
def retry_generation_job(project_id: str, job_id: str, data: JobAction) -> dict:
    """Run the same request again (same workflow and prompt; optionally a new seed)."""
    project = _get_or_404(project_id)
    previous = MANAGER.get(project_id, job_id)
    if previous is None:
        raise HTTPException(status_code=404, detail="Job not found")
    record = _workflow_or_404(project, previous.workflow_id)
    request = previous.request.model_copy(update={"randomize_seed": data.randomize_seed,
                                                  "seed": None if data.randomize_seed else previous.seed_used})
    try:
        return MANAGER.submit(project, record, request).model_dump()
    except JobError as exc:
        raise HTTPException(status_code=503 if exc.code == "COMFYUI_UNAVAILABLE" else 409, detail=exc.to_dict())


@app.get("/api/projects/{project_id}/jobs/{job_id}/outputs/{index}")
def get_job_output(project_id: str, job_id: str, index: int) -> FileResponse:
    """Serve a generated image that was downloaded into this project."""
    _get_or_404(project_id)
    job = MANAGER.get(project_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    output = next((o for o in job.outputs if o.index == index), None)
    if output is None:
        raise HTTPException(status_code=404, detail="Output not found")
    try:
        path = safe_path(project_id, output.path)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid output path")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="The output file is no longer on disk")
    return FileResponse(path, filename=output.filename)


@app.get("/images", include_in_schema=False)
def images_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "images.html")


# --------------------------------------------------------------------------- Phase 3: studio

def _studio_error(exc: StudioError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=exc.to_dict())


@contextmanager
def _studio_edit(project_id: str, base_revision: int | None, snapshot_reason: str | None = None):
    """Revision-checked edit of studio data, with an optional safety snapshot first."""
    with project_lock(project_id):
        project = _get_or_404(project_id)
        check_revision(project, base_revision)
        if snapshot_reason:
            create_snapshot(project, snapshot_reason)
        try:
            yield project
        except StudioError as exc:
            raise _studio_error(exc)
        save_project(project)
        maybe_autosnapshot(project)


@app.get("/api/projects/{project_id}/studio")
def studio_overview_endpoint(project_id: str) -> dict:
    """Characters, locations, props, assets and scene associations in one payload."""
    return studio_overview(_get_or_404(project_id))


@app.get("/studio", include_in_schema=False)
def studio_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "studio.html")


# ---- characters & wardrobe ----

@app.post("/api/projects/{project_id}/characters", status_code=201)
def create_character_endpoint(project_id: str, data: CharacterCreate) -> dict:
    with _studio_edit(project_id, data.base_revision) as project:
        character = create_character(project, data)
    return character.model_dump() | {"revision": project.revision}


@app.delete("/api/projects/{project_id}/characters/{character_id}")
def delete_character_endpoint(project_id: str, character_id: str, data: RevisionOnly | None = None) -> dict:
    data = data or RevisionOnly()
    with _studio_edit(project_id, data.base_revision, "pre-delete") as project:
        result = delete_subject(project, "character", character_id)
    return result | {"revision": project.revision}


@app.post("/api/projects/{project_id}/characters/{character_id}/wardrobe", status_code=201)
def create_wardrobe_endpoint(project_id: str, character_id: str, data: WardrobeCreate) -> dict:
    with _studio_edit(project_id, data.base_revision) as project:
        character = require_subject(project, "character", character_id)
        variant = create_wardrobe(project, character, data)
    return variant.model_dump() | {"character_id": character_id, "revision": project.revision}


@app.patch("/api/projects/{project_id}/characters/{character_id}/wardrobe/{wardrobe_id}")
def edit_wardrobe_endpoint(project_id: str, character_id: str, wardrobe_id: str, data: WardrobeEdit) -> dict:
    with _studio_edit(project_id, data.base_revision) as project:
        character = require_subject(project, "character", character_id)
        variant = find_wardrobe(character, wardrobe_id)
        if variant is None:
            raise HTTPException(status_code=404, detail="Wardrobe variant not found")
        apply_edit(variant, data)
    return variant.model_dump() | {"character_id": character_id, "revision": project.revision}


@app.delete("/api/projects/{project_id}/characters/{character_id}/wardrobe/{wardrobe_id}")
def delete_wardrobe_endpoint(project_id: str, character_id: str, wardrobe_id: str,
                             data: RevisionOnly | None = None) -> dict:
    data = data or RevisionOnly()
    with _studio_edit(project_id, data.base_revision, "pre-delete") as project:
        character = require_subject(project, "character", character_id)
        result = delete_wardrobe(project, character, wardrobe_id)
    return result | {"revision": project.revision}


# ---- locations & props ----

@app.post("/api/projects/{project_id}/locations", status_code=201)
def create_location_endpoint(project_id: str, data: LocationCreate) -> dict:
    with _studio_edit(project_id, data.base_revision) as project:
        location = create_location(project, data)
    return location.model_dump() | {"revision": project.revision}


@app.patch("/api/projects/{project_id}/locations/{location_id}")
def edit_location_endpoint(project_id: str, location_id: str, data: LocationEdit) -> dict:
    with _studio_edit(project_id, data.base_revision) as project:
        location = require_subject(project, "location", location_id)
        if data.scene_ids is not None:
            _check_scene_ids(project, data.scene_ids)
        apply_edit(location, data)
    return location.model_dump() | {"revision": project.revision}


@app.delete("/api/projects/{project_id}/locations/{location_id}")
def delete_location_endpoint(project_id: str, location_id: str, data: RevisionOnly | None = None) -> dict:
    data = data or RevisionOnly()
    with _studio_edit(project_id, data.base_revision, "pre-delete") as project:
        result = delete_subject(project, "location", location_id)
    return result | {"revision": project.revision}


@app.post("/api/projects/{project_id}/props", status_code=201)
def create_prop_endpoint(project_id: str, data: PropCreate) -> dict:
    with _studio_edit(project_id, data.base_revision) as project:
        prop = create_prop(project, data)
    return prop.model_dump() | {"revision": project.revision}


@app.patch("/api/projects/{project_id}/props/{prop_id}")
def edit_prop_endpoint(project_id: str, prop_id: str, data: PropEdit) -> dict:
    with _studio_edit(project_id, data.base_revision) as project:
        prop = require_subject(project, "prop", prop_id)
        if data.scene_ids is not None:
            _check_scene_ids(project, data.scene_ids)
        if data.owner_character_id:
            require_subject(project, "character", data.owner_character_id)
        apply_edit(prop, data)
    return prop.model_dump() | {"revision": project.revision}


@app.delete("/api/projects/{project_id}/props/{prop_id}")
def delete_prop_endpoint(project_id: str, prop_id: str, data: RevisionOnly | None = None) -> dict:
    data = data or RevisionOnly()
    with _studio_edit(project_id, data.base_revision, "pre-delete") as project:
        result = delete_subject(project, "prop", prop_id)
    return result | {"revision": project.revision}


def _check_scene_ids(project: Project, scene_ids: list[str]) -> None:
    known = {s.id for s in project.scenes}
    unknown = [s for s in scene_ids if s not in known]
    if unknown:
        raise HTTPException(status_code=400, detail={
            "code": "SCENE_NOT_FOUND",
            "message": f"These scene ids are not in this project: {', '.join(unknown[:5])}"})


# ---- canonical reference selection ----

def _set_canonical(project_id: str, subject_type: str, subject_id: str, data: CanonicalSelect,
                   wardrobe_id: str | None = None) -> dict:
    with _studio_edit(project_id, data.base_revision) as project:
        result = set_canonical(project, subject_type, subject_id, data.asset_id, wardrobe_id,
                               data.confirm_replace, data.note)
    return {"subject_type": subject_type, "subject_id": subject_id, "wardrobe_id": wardrobe_id,
            "canonical_reference_id": result["holder"].canonical_reference_id,
            "replaced_asset_id": result["replaced"],
            "canonical_history": [c.model_dump() for c in result["holder"].canonical_history],
            "note": "The replaced image is kept in the library." if result["replaced"] else "",
            "revision": project.revision}


def _clear_canonical(project_id: str, subject_type: str, subject_id: str, data: RevisionOnly,
                     wardrobe_id: str | None = None) -> dict:
    with _studio_edit(project_id, data.base_revision) as project:
        result = clear_canonical(project, subject_type, subject_id, wardrobe_id)
    return {"subject_type": subject_type, "subject_id": subject_id, "wardrobe_id": wardrobe_id,
            "cleared_asset_id": result["cleared"], "revision": project.revision}


@app.post("/api/projects/{project_id}/characters/{character_id}/canonical")
def set_character_canonical(project_id: str, character_id: str, data: CanonicalSelect) -> dict:
    return _set_canonical(project_id, "character", character_id, data)


@app.delete("/api/projects/{project_id}/characters/{character_id}/canonical")
def clear_character_canonical(project_id: str, character_id: str, data: RevisionOnly | None = None) -> dict:
    return _clear_canonical(project_id, "character", character_id, data or RevisionOnly())


@app.post("/api/projects/{project_id}/characters/{character_id}/wardrobe/{wardrobe_id}/canonical")
def set_wardrobe_canonical(project_id: str, character_id: str, wardrobe_id: str, data: CanonicalSelect) -> dict:
    return _set_canonical(project_id, "character", character_id, data, wardrobe_id)


@app.delete("/api/projects/{project_id}/characters/{character_id}/wardrobe/{wardrobe_id}/canonical")
def clear_wardrobe_canonical(project_id: str, character_id: str, wardrobe_id: str,
                             data: RevisionOnly | None = None) -> dict:
    return _clear_canonical(project_id, "character", character_id, data or RevisionOnly(), wardrobe_id)


@app.post("/api/projects/{project_id}/locations/{location_id}/canonical")
def set_location_canonical(project_id: str, location_id: str, data: CanonicalSelect) -> dict:
    return _set_canonical(project_id, "location", location_id, data)


@app.delete("/api/projects/{project_id}/locations/{location_id}/canonical")
def clear_location_canonical(project_id: str, location_id: str, data: RevisionOnly | None = None) -> dict:
    return _clear_canonical(project_id, "location", location_id, data or RevisionOnly())


@app.post("/api/projects/{project_id}/props/{prop_id}/canonical")
def set_prop_canonical(project_id: str, prop_id: str, data: CanonicalSelect) -> dict:
    return _set_canonical(project_id, "prop", prop_id, data)


@app.delete("/api/projects/{project_id}/props/{prop_id}/canonical")
def clear_prop_canonical(project_id: str, prop_id: str, data: RevisionOnly | None = None) -> dict:
    return _clear_canonical(project_id, "prop", prop_id, data or RevisionOnly())


# ---- reference image library ----

@app.post("/api/projects/{project_id}/assets", status_code=201)
def import_asset(project_id: str, data: AssetImport) -> dict:
    """Import one image from this computer into project-managed storage."""
    has_data = data.data_base64 is not None          # an empty string still means "a file was sent"
    has_path = bool(data.source_path and data.source_path.strip())
    if has_data == has_path:
        raise HTTPException(status_code=422, detail={
            "code": "ONE_SOURCE_REQUIRED",
            "message": "Send exactly one of data_base64 (browser upload) or source_path (file on this computer)."})
    try:
        if has_path:
            raw, filename = read_source_file(data.source_path)
            source_detail = data.source_path.strip()[:1000]
        else:
            filename = (data.filename or "upload").strip()[:260]
            raw = decode_base64(data.data_base64 or "", filename)
            source_detail = f"uploaded as {filename}"
        info = inspect_image(raw, filename)
    except ImageError as exc:
        raise HTTPException(status_code=422, detail=exc.to_dict())

    digest = hashlib.sha256(raw).hexdigest()
    with project_lock(project_id):
        project = _get_or_404(project_id)
        check_revision(project, data.base_revision)
        existing = duplicate_of(project, digest)
        if existing is not None and not data.allow_duplicate:
            raise HTTPException(status_code=409, detail={
                "code": "DUPLICATE_ASSET",
                "message": f"This exact image is already in the library as '{existing.original_filename or existing.id}'"
                           f" ({existing.state}). Send allow_duplicate=true to import a second copy.",
                "asset_id": existing.id})
        try:
            association = validate_association(project, data.association)
        except StudioError as exc:
            raise _studio_error(exc)

        asset_id = short_id()
        relative, stored_digest = write_asset_file(project_id, asset_id, info["suffix"], raw)
        try:
            asset = ReferenceAsset(
                id=asset_id, path=relative, original_filename=filename, sha256=stored_digest,
                bytes=info["bytes"], width=info["width"], height=info["height"],
                image_format=info["format"], source="import", source_detail=source_detail,
                license=data.license.strip(), license_url=data.license_url.strip(), notes=data.notes.strip(),
                integrity=info["integrity"], **association)
            record_asset(project, asset)
            save_project(project)
        except Exception:
            delete_asset_file(project_id, relative)       # never leave a file without a record
            raise
        write_manifest(project_id)
        return asset_view(project, asset) | {
            "revision": project.revision,
            "duplicate_of": existing.id if existing else None,
            "extension_mismatch": info["extension_mismatch"],
            "stored_as": relative,
        }


@app.post("/api/projects/{project_id}/assets/from-job", status_code=201)
def import_asset_from_job(project_id: str, data: AssetFromJob) -> dict:
    """Copy a generated image into the reference library as a draft (the job output stays untouched)."""
    project = _get_or_404(project_id)
    job = MANAGER.get(project_id, data.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    output = next((o for o in job.outputs if o.index == data.output_index), None)
    if output is None:
        raise HTTPException(status_code=404, detail={
            "code": "OUTPUT_NOT_FOUND",
            "message": f"Job {data.job_id} has no output #{data.output_index}."})
    try:
        raw = read_asset_file(project_id, output.path, output.sha256)
        info = inspect_image(raw, output.filename)
    except StorageError as exc:
        raise HTTPException(status_code=409, detail={"code": "OUTPUT_UNAVAILABLE", "message": str(exc)})
    except ImageError as exc:
        raise HTTPException(status_code=422, detail=exc.to_dict())

    with project_lock(project_id):
        project = _get_or_404(project_id)
        check_revision(project, data.base_revision)
        existing = duplicate_of(project, output.sha256)
        if existing is not None and not data.allow_duplicate:
            raise HTTPException(status_code=409, detail={
                "code": "DUPLICATE_ASSET",
                "message": f"That generated image is already in the library ({existing.id}).",
                "asset_id": existing.id})
        try:
            association = validate_association(project, data.association)
        except StudioError as exc:
            raise _studio_error(exc)
        asset_id = short_id()
        relative, stored_digest = write_asset_file(project_id, asset_id, info["suffix"], raw)
        try:
            asset = ReferenceAsset(
                id=asset_id, path=relative, original_filename=output.filename, sha256=stored_digest,
                bytes=info["bytes"], width=info["width"], height=info["height"], image_format=info["format"],
                source="generated", source_detail=output.path, job_id=job.id,
                license=data.license.strip(), notes=data.notes.strip(), integrity=info["integrity"],
                **association)
            record_asset(project, asset)
            save_project(project)
        except Exception:
            delete_asset_file(project_id, relative)
            raise
        write_manifest(project_id)
        return asset_view(project, asset) | {"revision": project.revision, "from_job": job.id}


@app.get("/api/projects/{project_id}/assets")
def list_assets(project_id: str, subject_type: str | None = None, subject_id: str | None = None,
                state: str | None = None, unassigned: bool = False) -> dict:
    project = _get_or_404(project_id)
    views = [asset_view(project, asset) for asset in project.assets]
    if subject_type:
        views = [v for v in views if v["subject_type"] == subject_type]
    if subject_id:
        views = [v for v in views if v["subject_id"] == subject_id]
    if state:
        views = [v for v in views if v["state"] == state]
    if unassigned:
        views = [v for v in views if not v["subject_id"]]
    return {"revision": project.revision, "assets": views, "total": len(project.assets),
            "limits": {"max_bytes": max_bytes(), "formats": list(ALLOWED_FORMATS)}}


@app.get("/api/projects/{project_id}/assets/{asset_id}")
def get_asset(project_id: str, asset_id: str) -> dict:
    project = _get_or_404(project_id)
    try:
        asset = require_asset(project, asset_id)
    except StudioError as exc:
        raise _studio_error(exc)
    return asset_view(project, asset) | {"references": references_to(project, asset_id, include_jobs=True),
                                          "revision": project.revision}


@app.get("/api/projects/{project_id}/assets/{asset_id}/file")
def get_asset_file(project_id: str, asset_id: str, download: bool = False) -> FileResponse:
    project = _get_or_404(project_id)
    try:
        asset = require_asset(project, asset_id)
    except StudioError as exc:
        raise _studio_error(exc)
    try:
        path = safe_path(project_id, asset.path)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid asset path")
    if not path.is_file():
        raise HTTPException(status_code=404, detail={
            "code": "ASSET_FILE_MISSING",
            "message": f"The stored file for this asset is missing ({asset.path})."})
    media_type = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp", "GIF": "image/gif",
                  "BMP": "image/bmp", "TIFF": "image/tiff"}.get(asset.image_format, "application/octet-stream")
    return FileResponse(path, media_type=media_type,
                        filename=(asset.original_filename or path.name) if download else None)


@app.patch("/api/projects/{project_id}/assets/{asset_id}")
def edit_asset(project_id: str, asset_id: str, data: AssetEdit) -> dict:
    """Change notes, licence, role or which subject an image belongs to."""
    with _studio_edit(project_id, data.base_revision) as project:
        asset = require_asset(project, asset_id)
        changes = data.model_dump(exclude_unset=True, exclude={"base_revision", "clear_association"})
        if data.clear_association:
            blocking = [r for r in references_to(project, asset_id) if r["blocking"]]
            if blocking:
                raise StudioError("ASSET_IN_USE",
                                  "This image is still used by " + ", ".join(r["label"] for r in blocking)
                                  + ". Unlink it there before detaching it from its subject.", detail=blocking)
            asset.subject_type = None
            asset.subject_id = None
            asset.wardrobe_id = None
        elif {"subject_type", "subject_id", "wardrobe_id", "role"} & set(changes):
            association = validate_association(project, AssetAssociation(
                subject_type=changes.get("subject_type", asset.subject_type),
                subject_id=changes.get("subject_id", asset.subject_id),
                wardrobe_id=changes.get("wardrobe_id", asset.wardrobe_id),
                role=changes.get("role", asset.role)))
            moved = (association["subject_type"], association["subject_id"], association["wardrobe_id"],
                     association["role"]) != (asset.subject_type, asset.subject_id, asset.wardrobe_id, asset.role)
            blocking = [r for r in references_to(project, asset_id) if r["blocking"]] if moved else []
            if blocking:
                raise StudioError("ASSET_IN_USE",
                                  "This image is used by " + ", ".join(r["label"] for r in blocking)
                                  + ". Unlink it there before re-assigning it.", detail=blocking)
            for key, value in association.items():
                setattr(asset, key, value)
            if moved:
                asset.version = next_version(project, asset.subject_type, asset.subject_id,
                                             asset.wardrobe_id, asset.role)
        for key in ("license", "license_url", "notes"):
            if key in changes and changes[key] is not None:
                setattr(asset, key, str(changes[key]).strip())
        asset.updated_at = utc_now()
    return asset_view(project, asset) | {"revision": project.revision}


@app.post("/api/projects/{project_id}/assets/{asset_id}/state")
def set_asset_state(project_id: str, asset_id: str, data: AssetStateChange) -> dict:
    """draft / approved / rejected. Demoting a canonical reference needs an explicit confirmation."""
    with _studio_edit(project_id, data.base_revision) as project:
        asset = require_asset(project, asset_id)
        result = change_state(project, asset, data.state, data.note, data.confirm_demote_canonical)
    return asset_view(project, asset) | {"revision": project.revision, "changed": result["changed"],
                                          "unlinked": result["unlinked"]}


@app.get("/api/projects/{project_id}/assets/{asset_id}/references")
def get_asset_references(project_id: str, asset_id: str) -> dict:
    project = _get_or_404(project_id)
    try:
        require_asset(project, asset_id)
    except StudioError as exc:
        raise _studio_error(exc)
    references = references_to(project, asset_id, include_jobs=True)
    return {"asset_id": asset_id, "references": references,
            "deletable": not any(r["blocking"] for r in references), "revision": project.revision}


@app.delete("/api/projects/{project_id}/assets/{asset_id}")
def delete_asset_endpoint(project_id: str, asset_id: str, data: AssetDelete | None = None) -> dict:
    """Delete a reference image and its stored file, once nothing points at it."""
    data = data or AssetDelete()
    with project_lock(project_id):
        project = _get_or_404(project_id)
        check_revision(project, data.base_revision)
        try:
            require_asset(project, asset_id)
            assert_deletable(project, asset_id, data.confirm)   # refuse before touching anything
            create_snapshot(project, "pre-asset-delete")
            result = delete_asset(project, asset_id, data.confirm)
        except StudioError as exc:
            raise _studio_error(exc)
        save_project(project)
        removed = delete_asset_file(project_id, result["path"])
        write_manifest(project_id)
        return {"deleted": asset_id, "file_removed": removed, "path": result["path"],
                "revision": project.revision,
                "note": "A snapshot was taken first, but the image file itself is gone." if removed
                        else "The record was removed; the file could not be deleted and is now unreferenced."}


# ---- scene associations ----

@app.put("/api/projects/{project_id}/scenes/{scene_id}/associations")
def set_scene_associations_endpoint(project_id: str, scene_id: str, data: SceneAssociations) -> dict:
    """Attach characters/location/props to a scene and pick approved looks. The screenplay is untouched."""
    with _studio_edit(project_id, data.base_revision) as project:
        scene = _find(project.scenes, scene_id, "Scene")
        set_scene_associations(project, scene, data)
    return {"scene": scene.model_dump(), "revision": project.revision}

@app.get("/api/projects/{project_id}/shot-list.csv")
def shot_list_csv(project_id: str) -> StreamingResponse:
    project = _get_or_404(project_id)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=[
        "scene_number", "scene_heading", "shot_number", "shot_type", "duration_seconds",
        "characters", "visual_prompt", "motion_prompt", "audio_notes", "status", "media_path", "notes"
    ])
    writer.writeheader()
    scene_by_id = {scene.id: scene for scene in project.scenes}
    for shot in project.shots:
        scene = scene_by_id.get(shot.scene_id)
        writer.writerow({
            "scene_number": shot.scene_number,
            "scene_heading": scene.heading if scene else "Unknown",
            "shot_number": shot.number,
            "shot_type": shot.shot_type,
            "duration_seconds": shot.duration_seconds,
            "characters": ", ".join(shot.characters),
            "visual_prompt": shot.visual_prompt,
            "motion_prompt": shot.motion_prompt,
            "audio_notes": shot.audio_notes,
            "status": shot.status,
            "media_path": shot.media_path or "",
            "notes": shot.notes,
        })
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{project.id}-shot-list.csv"'},
    )
