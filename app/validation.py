"""Project validation warnings.

Combines the parser's warnings (stored at breakdown time) with checks on the current,
possibly hand-edited production data. Warnings never block saving; they ask for an explicit
manual correction or an explicit acknowledgement ("this is intentional").
"""
from __future__ import annotations

import hashlib

from app.models import ParseWarning, Project

MAX_RECOMMENDED_SHOT_SECONDS = 10.0


def _wid(*parts: object) -> str:
    return hashlib.sha1("|".join(str(p) for p in ("validate",) + parts).encode("utf-8")).hexdigest()[:12]


def _warn(code: str, message: str, severity: str = "warning", entity_type: str | None = None,
          entity_id: str | None = None, scene_id: str | None = None, line: int | None = None) -> ParseWarning:
    return ParseWarning(id=_wid(code, entity_id, message if entity_id is None else ""), code=code,
                        severity=severity, message=message, entity_type=entity_type, entity_id=entity_id,
                        scene_id=scene_id, line=line)


def validate_project(project: Project) -> list[ParseWarning]:
    found: list[ParseWarning] = list(project.parse_warnings)

    if not project.script_versions and not project.script.strip():
        found.append(_warn("NO_SCRIPT", "No screenplay has been imported yet.", "info"))
    if project.current_script_version and project.scenes and \
            project.breakdown_script_version != project.current_script_version:
        found.append(_warn(
            "BREAKDOWN_OUT_OF_DATE",
            f"The scene breakdown was built from script v{project.breakdown_script_version or '?'} but the current "
            f"script is v{project.current_script_version}. Rebuild the breakdown to pick up script changes "
            "(a snapshot is taken first).",
        ))

    scene_ids = {s.id for s in project.scenes}
    character_names = {c.name for c in project.characters}

    seen_names: dict[str, str] = {}
    for character in project.characters:
        key = character.name.strip().upper()
        if key in seen_names:
            found.append(_warn("DUPLICATE_CHARACTER_NAME",
                               f"Two character records are named '{character.name}'.",
                               entity_type="character", entity_id=character.id))
        seen_names.setdefault(key, character.id)
        if character.description.strip() in ("", "To be defined in the character bible.",
                                              "Define age, appearance, personality and backstory in the character bible."):
            found.append(_warn("CHARACTER_UNDEFINED",
                               f"Character '{character.name}' has no bible description yet.", "info",
                               entity_type="character", entity_id=character.id))

    for scene in project.scenes:
        for name in scene.characters:
            if name not in character_names:
                found.append(_warn("SCENE_UNKNOWN_CHARACTER",
                                   f"Scene {scene.number} lists '{name}', who has no character record.",
                                   entity_type="scene", entity_id=scene.id, scene_id=scene.id))
        if not scene.heading.strip():
            found.append(_warn("SCENE_NO_HEADING", f"Scene {scene.number} has an empty heading.", "error",
                               entity_type="scene", entity_id=scene.id, scene_id=scene.id))
        if project.shots and not any(s.scene_id == scene.id for s in project.shots):
            found.append(_warn("SCENE_NO_SHOTS", f"Scene {scene.number} has no planned shots.",
                               entity_type="scene", entity_id=scene.id, scene_id=scene.id))

    for shot in project.shots:
        label = f"Shot {shot.scene_number}.{shot.number}"
        if shot.scene_id not in scene_ids:
            found.append(_warn("SHOT_ORPHANED", f"{label} points to a scene that no longer exists.", "error",
                               entity_type="shot", entity_id=shot.id))
        if not shot.visual_prompt.strip():
            found.append(_warn("SHOT_EMPTY_PROMPT", f"{label} has an empty visual prompt.", "error",
                               entity_type="shot", entity_id=shot.id, scene_id=shot.scene_id))
        if shot.duration_seconds > MAX_RECOMMENDED_SHOT_SECONDS:
            found.append(_warn("SHOT_LONG_DURATION",
                               f"{label} is {shot.duration_seconds:g}s; local video models usually produce "
                               f"clips of a few seconds. Consider splitting it.", "info",
                               entity_type="shot", entity_id=shot.id, scene_id=shot.scene_id))
        for name in shot.characters:
            if name not in character_names:
                found.append(_warn("SHOT_UNKNOWN_CHARACTER",
                                   f"{label} lists '{name}', who has no character record.",
                                   entity_type="shot", entity_id=shot.id, scene_id=shot.scene_id))
        if shot.status in ("generated", "approved") and not shot.media_path:
            found.append(_warn("SHOT_STATUS_WITHOUT_MEDIA",
                               f"{label} is marked '{shot.status}' but has no media attached.",
                               entity_type="shot", entity_id=shot.id, scene_id=shot.scene_id))

    found.extend(_studio_warnings(project))

    acknowledged = set(project.acknowledged_warnings)
    unique: dict[str, ParseWarning] = {}
    for w in found:
        if w.id not in unique:
            unique[w.id] = w.model_copy(update={"acknowledged": w.id in acknowledged})
    order = {"error": 0, "warning": 1, "info": 2}
    return sorted(unique.values(), key=lambda w: (w.acknowledged, order[w.severity], w.line or 0))


def _studio_warnings(project: Project) -> list[ParseWarning]:
    """Phase 3 checks: asset files on disk, canonical pointers and scene selections."""
    from app import storage

    found: list[ParseWarning] = []
    assets = {asset.id: asset for asset in project.assets}
    for asset in project.assets:
        status = storage.asset_file_status(project.id, asset.path, asset.sha256)
        if status == "missing":
            found.append(_warn("ASSET_FILE_MISSING",
                               f"Reference image '{asset.original_filename or asset.id}' is recorded but its file "
                               f"({asset.path}) is gone. Re-import it or delete the record.", "error",
                               entity_type="asset", entity_id=asset.id))
        elif status == "modified":
            found.append(_warn("ASSET_FILE_MODIFIED",
                               f"Reference image '{asset.original_filename or asset.id}' no longer matches its "
                               "recorded SHA-256; the file was changed outside the app.", "error",
                               entity_type="asset", entity_id=asset.id))
        if asset.subject_id and not _subject_exists(project, asset.subject_type, asset.subject_id):
            found.append(_warn("ASSET_ORPHANED",
                               f"Reference image '{asset.original_filename or asset.id}' points at a "
                               f"{asset.subject_type} that no longer exists.",
                               entity_type="asset", entity_id=asset.id))

    for subject_type, field, label in (("character", "characters", "Character"),
                                       ("location", "locations", "Location"), ("prop", "props", "Prop")):
        for subject in getattr(project, field):
            holders = [(subject, f"{label} '{subject.name}'")]
            if subject_type == "character":
                holders += [(v, f"{subject.name} / wardrobe '{v.name}'") for v in subject.wardrobe]
            for holder, holder_label in holders:
                canonical_id = holder.canonical_reference_id
                if not canonical_id:
                    if subject_type != "prop" and holder is subject:
                        found.append(_warn("NO_CANONICAL_REFERENCE",
                                           f"{holder_label} has no canonical reference image yet.", "info",
                                           entity_type=subject_type, entity_id=subject.id))
                    continue
                asset = assets.get(canonical_id)
                if asset is None:
                    found.append(_warn("CANONICAL_MISSING",
                                       f"{holder_label} points at reference image '{canonical_id}', which is not "
                                       "in the library.", "error", entity_type=subject_type, entity_id=subject.id))
                elif asset.state != "approved":
                    found.append(_warn("CANONICAL_NOT_APPROVED",
                                       f"{holder_label} uses a '{asset.state}' image as its canonical reference.",
                                       "error", entity_type=subject_type, entity_id=subject.id))

    for scene in project.scenes:
        if scene.location_reference_id:
            asset = assets.get(scene.location_reference_id)
            if asset is None or asset.state != "approved":
                found.append(_warn("SCENE_REFERENCE_INVALID",
                                   f"Scene {scene.number} selects a location reference that is missing or not "
                                   "approved.", entity_type="scene", entity_id=scene.id, scene_id=scene.id))
        for selection in scene.appearance_selections:
            if selection.asset_id:
                asset = assets.get(selection.asset_id)
                if asset is None or asset.state != "approved":
                    found.append(_warn("SCENE_REFERENCE_INVALID",
                                       f"Scene {scene.number} selects an appearance image that is missing or not "
                                       "approved.", entity_type="scene", entity_id=scene.id, scene_id=scene.id))
        for character_id in scene.character_ids:
            if not _subject_exists(project, "character", character_id):
                found.append(_warn("SCENE_UNKNOWN_SUBJECT",
                                   f"Scene {scene.number} is linked to a character record that no longer exists.",
                                   entity_type="scene", entity_id=scene.id, scene_id=scene.id))
    return found


def _subject_exists(project: Project, subject_type: str | None, subject_id: str) -> bool:
    field = {"character": "characters", "location": "locations", "prop": "props"}.get(subject_type or "")
    return bool(field) and any(item.id == subject_id for item in getattr(project, field))


def summarize(warnings: list[ParseWarning]) -> dict[str, int]:
    open_items = [w for w in warnings if not w.acknowledged]
    return {
        "errors": sum(w.severity == "error" for w in open_items),
        "warnings": sum(w.severity == "warning" for w in open_items),
        "info": sum(w.severity == "info" for w in open_items),
        "acknowledged": len(warnings) - len(open_items),
    }
