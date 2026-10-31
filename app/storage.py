"""Local project persistence.

Layout per project (see docs/ARCHITECTURE.md):

    <project-id>/project.json              current editable state (atomic writes, revision counter)
    <project-id>/screenplay/original.txt   first imported screenplay, read-only, never overwritten
    <project-id>/screenplay/versions/vNNNN.txt   later script versions (write-once)
    <project-id>/snapshots/<id>.json       full project-state snapshots + index.json
    <project-id>/manifests/assets.json     file list with sizes and SHA-256 hashes
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models import Project, ScriptVersion, SnapshotInfo, utc_now

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIR = Path(os.environ.get("FILM_HOUSE_PROJECTS_DIR", ROOT / "workspace" / "projects")).resolve()
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_SHORT_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_SNAPSHOT_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}\d{0,6}Z-r\d+-[a-z\-]{1,24}-[a-f0-9]{6}$")
ORIGINAL_SCRIPT = "screenplay/original.txt"
AUTOSAVE_KEEP = 30
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


class RevisionConflict(Exception):
    def __init__(self, current: int, base: int):
        super().__init__(f"Project changed elsewhere (revision {current}, you edited revision {base}).")
        self.current = current
        self.base = base


class StorageError(Exception):
    pass


# --------------------------------------------------------------------------- paths & locks

def project_lock(project_id: str) -> threading.RLock:
    with _locks_guard:
        return _locks.setdefault(project_id, threading.RLock())


def project_dir(project_id: str) -> Path:
    if not _ID_RE.fullmatch(project_id or ""):
        raise ValueError("Invalid project id")
    return PROJECTS_DIR / project_id


def project_path(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"


def safe_path(project_id: str, relative: str) -> Path:
    """Resolve a project-relative path and refuse anything outside the project directory."""
    base = project_dir(project_id).resolve()
    target = (base / relative).resolve()
    if target != base and base not in target.parents:
        raise ValueError("Path escapes the project workspace")
    return target


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bytes_retry(path: Path, attempts: int = 12) -> bytes:
    """Read a file that another thread may be replacing right now.

    `os.replace` is atomic, but on Windows a reader that opens the file during the swap gets
    PermissionError/FileNotFoundError instead of either version. Retry briefly before giving up.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return path.read_bytes()
        except (PermissionError, FileNotFoundError) as exc:
            last = exc
            if attempt == attempts - 1:
                break
            time.sleep(0.02)
    raise last if last else OSError(f"could not read {path}")


def _read_text_retry(path: Path) -> str:
    return _read_bytes_retry(path).decode("utf-8")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(10):  # Windows may briefly lock a file that is being read
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 9:
                temp.unlink(missing_ok=True)
                raise
            time.sleep(0.05)


def _write_once(path: Path, text: str) -> None:
    """Create a file that must never be overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def _write_once_bytes(path: Path, data: bytes) -> None:
    """Create a binary file that must never be overwritten (used for imported reference images)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


# --------------------------------------------------------------------------- project state

def check_revision(project: Project, base_revision: int | None) -> None:
    if base_revision is not None and base_revision != project.revision:
        raise RevisionConflict(project.revision, base_revision)


def save_project(project: Project) -> Project:
    """Persist atomically and bump the revision counter."""
    with project_lock(project.id):
        project.revision += 1
        project.updated_at = utc_now()
        _atomic_write_text(project_path(project.id), project.model_dump_json(indent=2))
    return project


def load_project(project_id: str) -> Project | None:
    try:
        path = project_path(project_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    return Project.model_validate_json(_read_text_retry(path))


def list_projects() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for file_path in PROJECTS_DIR.glob("*/project.json"):
        try:
            project = Project.model_validate_json(_read_text_retry(file_path))
            results.append({
                "id": project.id,
                "title": project.title,
                "status": project.status,
                "updated_at": project.updated_at,
                "revision": project.revision,
                "scene_count": len(project.scenes),
                "shot_count": len(project.shots),
                "script_versions": len(project.script_versions),
            })
        except Exception:
            # Ignore corrupted project files in listing; the original is not deleted.
            continue
    return sorted(results, key=lambda item: item["updated_at"], reverse=True)


# --------------------------------------------------------------------------- script versions

def _add_version(project: Project, text: str, source: str, note: str) -> ScriptVersion:
    number = (max((v.number for v in project.script_versions), default=0)) + 1
    relative = ORIGINAL_SCRIPT if number == 1 else f"screenplay/versions/v{number:04d}.txt"
    path = safe_path(project.id, relative)
    if path.exists():
        raise StorageError(f"Refusing to overwrite existing script file {relative}")
    _write_once(path, text)
    version = ScriptVersion(number=number, path=relative, sha256=_sha256_bytes(text.encode("utf-8")),
                            source=source, note=note, characters=len(text))
    project.script_versions.append(version)
    project.current_script_version = number
    project.script = text
    return version


def add_script_version(project: Project, text: str, note: str = "") -> ScriptVersion | None:
    """Store `text` as a new script version. Returns None if it equals the current version.

    The first version is the immutable original. A legacy project (Phase 0) whose script was only
    stored inside project.json gets that text preserved as the original first.
    """
    with project_lock(project.id):
        if not project.script_versions and project.script.strip():
            _add_version(project, project.script, "legacy_import", "Script saved before versioning existed")
        if project.script_versions and project.script == text:
            return None
        source = "import" if not project.script_versions else "edit"
        return _add_version(project, text, source, note or ("Original import" if source == "import" else ""))


def read_script_version(project: Project, number: int) -> str:
    version = next((v for v in project.script_versions if v.number == number), None)
    if version is None:
        raise KeyError(number)
    path = safe_path(project.id, version.path)
    data = _read_bytes_retry(path)
    if _sha256_bytes(data) != version.sha256:
        raise StorageError(f"Script version {number} failed its integrity check")
    return data.decode("utf-8")


# --------------------------------------------------------------------------- snapshots

def _snapshot_dir(project_id: str) -> Path:
    return safe_path(project_id, "snapshots")


def list_snapshots(project_id: str) -> list[SnapshotInfo]:
    index = _snapshot_dir(project_id) / "index.json"
    if not index.exists():
        return []
    try:
        return [SnapshotInfo.model_validate(item) for item in json.loads(index.read_text(encoding="utf-8"))]
    except (ValueError, OSError):
        return []


def _write_snapshot_index(project_id: str, items: list[SnapshotInfo]) -> None:
    _atomic_write_text(_snapshot_dir(project_id) / "index.json",
                       json.dumps([i.model_dump() for i in items], indent=2))


def create_snapshot(project: Project, reason: str, label: str = "") -> SnapshotInfo:
    reason = re.sub(r"[^a-z\-]", "", reason.lower())[:24] or "manual"
    with project_lock(project.id):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        snap_id = f"{stamp}-r{project.revision}-{reason}-{uuid.uuid4().hex[:6]}"
        relative = f"snapshots/{snap_id}.json"
        _write_once(safe_path(project.id, relative), project.model_dump_json(indent=2))
        info = SnapshotInfo(id=snap_id, created_at=utc_now(), revision=project.revision, reason=reason,
                            label=label.strip()[:120], path=relative)
        items = list_snapshots(project.id) + [info]
        # Only automatic autosave snapshots are pruned; manual and safety snapshots are kept.
        autos = [i for i in items if i.reason == "autosave"]
        for old in autos[:-AUTOSAVE_KEEP]:
            path = safe_path(project.id, old.path)
            if path.exists():
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                path.unlink()
            items.remove(old)
        _write_snapshot_index(project.id, items)
        return info


def maybe_autosnapshot(project: Project) -> SnapshotInfo | None:
    """Snapshot after edits, at most once per FILM_HOUSE_AUTOSNAPSHOT_SECONDS (default 120)."""
    interval = float(os.environ.get("FILM_HOUSE_AUTOSNAPSHOT_SECONDS", "120"))
    autos = [i for i in list_snapshots(project.id) if i.reason == "autosave"]
    if autos:
        last = datetime.fromisoformat(autos[-1].created_at)
        if (datetime.now(timezone.utc) - last).total_seconds() < interval:
            return None
    return create_snapshot(project, "autosave")


def load_snapshot(project_id: str, snapshot_id: str) -> Project:
    if not _SNAPSHOT_ID_RE.fullmatch(snapshot_id or ""):
        raise KeyError(snapshot_id)
    info = next((i for i in list_snapshots(project_id) if i.id == snapshot_id), None)
    if info is None:
        raise KeyError(snapshot_id)
    path = safe_path(project_id, info.path)
    snap = Project.model_validate_json(path.read_text(encoding="utf-8"))
    if snap.id != project_id:
        raise StorageError("Snapshot belongs to a different project")
    return snap


# --------------------------------------------------------------------------- comfy workflows & jobs

def short_id() -> str:
    return uuid.uuid4().hex[:12]


def _check_short_id(value: str) -> str:
    if not _SHORT_ID_RE.fullmatch(value or ""):
        raise ValueError("Invalid id")
    return value


def add_workflow_file(project_id: str, workflow_id: str, text: str) -> tuple[str, str]:
    """Store an imported workflow write-once. Returns (relative path, sha256)."""
    _check_short_id(workflow_id)
    relative = f"workflows/{workflow_id}.json"
    path = safe_path(project_id, relative)
    if path.exists():
        raise StorageError(f"Refusing to overwrite existing workflow file {relative}")
    _write_once(path, text)
    return relative, _sha256_bytes(text.encode("utf-8"))


def read_workflow_file(project_id: str, relative: str, sha256: str | None = None) -> str:
    data = _read_bytes_retry(safe_path(project_id, relative))
    if sha256 and _sha256_bytes(data) != sha256:
        raise StorageError(f"Workflow file {relative} failed its integrity check; it was modified on disk.")
    return data.decode("utf-8")


def job_path(project_id: str, job_id: str) -> Path:
    return safe_path(project_id, f"jobs/{_check_short_id(job_id)}.json")


def save_job(job: Any) -> Any:
    """Persist a generation job. Jobs live outside project.json, so they never cause edit conflicts."""
    job.updated_at = utc_now()
    with project_lock(job.project_id):
        _atomic_write_text(job_path(job.project_id, job.id), job.model_dump_json(indent=2))
    return job


def load_job(project_id: str, job_id: str) -> Any | None:
    from app.models import GenerationJob
    try:
        path = job_path(project_id, job_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        return GenerationJob.model_validate_json(_read_text_retry(path))
    except ValueError:
        return None


def list_jobs(project_id: str) -> list[Any]:
    from app.models import GenerationJob
    jobs = []
    directory = safe_path(project_id, "jobs")
    if not directory.exists():
        return jobs
    for path in directory.glob("*.json"):
        try:
            jobs.append(GenerationJob.model_validate_json(_read_text_retry(path)))
        except (ValueError, OSError):
            continue
    return sorted(jobs, key=lambda j: j.created_at, reverse=True)


def output_dir(project_id: str, job: Any) -> Path:
    base = "storyboards" if job.request.target_type == "shot" else "references"
    return safe_path(project_id, f"{base}/{_check_short_id(job.id)}")


def list_reference_images(project_id: str) -> list[dict[str, Any]]:
    """Images usable as a reference: generated outputs plus anything in references/input/."""
    base = project_dir(project_id).resolve()
    found: list[dict[str, Any]] = []
    for folder in ("references", "storyboards"):
        root = base / folder
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                stat_result = path.stat()
                found.append({"path": path.relative_to(base).as_posix(), "bytes": stat_result.st_size,
                              "modified": datetime.fromtimestamp(stat_result.st_mtime, timezone.utc).isoformat(),
                              "source": "upload" if folder == "references" and "input/" in
                                        path.relative_to(base).as_posix() else "generated"})
    return sorted(found, key=lambda item: item["modified"], reverse=True)


def resolve_project_image(project_id: str, relative: str) -> Path:
    """Validate that a project-relative path is an existing image inside this project."""
    path = safe_path(project_id, relative)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"'{relative}' is not an image file ({', '.join(IMAGE_SUFFIXES)}).")
    if not path.is_file():
        raise ValueError(f"'{relative}' does not exist in this project.")
    return path


# --------------------------------------------------------------------------- Phase 3 reference assets

ASSETS_DIR = "assets"


def asset_relative_path(asset_id: str, suffix: str) -> str:
    return f"{ASSETS_DIR}/{_check_short_id(asset_id)}{suffix}"


def write_asset_file(project_id: str, asset_id: str, suffix: str, data: bytes) -> tuple[str, str]:
    """Copy an imported image into project-managed storage, write-once. Returns (relative, sha256)."""
    if suffix.lower() not in IMAGE_SUFFIXES:
        raise StorageError(f"Refusing to store an asset with suffix '{suffix}'.")
    relative = asset_relative_path(asset_id, suffix.lower())
    path = safe_path(project_id, relative)
    if path.exists():
        raise StorageError(f"Refusing to overwrite existing asset file {relative}")
    _write_once_bytes(path, data)
    stored = path.read_bytes()
    digest = _sha256_bytes(stored)
    if stored != data:                      # a short write must never produce a half-recorded asset
        delete_asset_file(project_id, relative)
        raise StorageError(f"Asset file {relative} was not written completely; nothing was recorded.")
    return relative, digest


def read_asset_file(project_id: str, relative: str, sha256: str | None = None) -> bytes:
    path = safe_path(project_id, relative)
    if not path.is_file():
        raise StorageError(f"The asset file {relative} is missing from this project.")
    data = _read_bytes_retry(path)
    if sha256 and _sha256_bytes(data) != sha256:
        raise StorageError(f"Asset file {relative} failed its integrity check; it was modified on disk.")
    return data


def asset_file_status(project_id: str, relative: str, sha256: str) -> str:
    """'ok' | 'missing' | 'modified' — used by validation and the UI, never raises."""
    try:
        path = safe_path(project_id, relative)
    except ValueError:
        return "missing"
    if not path.is_file():
        return "missing"
    try:
        return "ok" if _sha256_file(path) == sha256 else "modified"
    except OSError:
        return "missing"


def delete_asset_file(project_id: str, relative: str) -> bool:
    """Remove a write-once asset file (read-only on disk, so clear the flag first)."""
    try:
        path = safe_path(project_id, relative)
    except ValueError:
        return False
    if not path.is_file():
        return False
    for attempt in range(5):
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
            path.unlink()
            return True
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05)
        except OSError:
            return False
    return False


# --------------------------------------------------------------------------- manifest

def _asset_kind(relative: str) -> str:
    if relative == "project.json":
        return "project_state"
    if relative == ORIGINAL_SCRIPT:
        return "script_original"
    if relative.startswith("screenplay/versions/"):
        return "script_version"
    if relative.startswith("snapshots/"):
        return "snapshot_index" if relative.endswith("index.json") else "snapshot"
    if relative.startswith("workflows/"):
        return "comfy_workflow"
    if relative.startswith("jobs/"):
        return "generation_job"
    if relative.startswith("assets/"):
        return "reference_asset"
    if relative.startswith("references/input/"):
        return "reference_input"
    if relative.startswith(("references/", "storyboards/")):
        return "generation_metadata" if relative.endswith("metadata.json") else "generated_image"
    return "asset"


def write_manifest(project_id: str) -> dict[str, Any]:
    with project_lock(project_id):
        base = project_dir(project_id).resolve()
        entries = []
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            relative = path.relative_to(base).as_posix()
            if relative.startswith("manifests/"):
                continue
            stat_result = path.stat()
            entries.append({
                "path": relative,
                "kind": _asset_kind(relative),
                "bytes": stat_result.st_size,
                "sha256": _sha256_file(path),
                "modified": datetime.fromtimestamp(stat_result.st_mtime, timezone.utc).isoformat(),
                "read_only": not os.access(path, os.W_OK),
            })
        manifest = {"project_id": project_id, "generated_at": utc_now(), "files": entries,
                    "note": "Hashes let you verify that originals and snapshots were not modified."}
        _atomic_write_text(safe_path(project_id, "manifests/assets.json"), json.dumps(manifest, indent=2))
        return manifest
