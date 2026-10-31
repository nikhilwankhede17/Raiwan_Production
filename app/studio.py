"""Phase 3 studio logic: characters, wardrobe variants, locations, props and the reference library.

Rules enforced here (the routes stay thin):

* identity vs appearance — character identity fields never change when a costume variant is added;
* explicit canonical selection — only an **approved** asset can become canonical, and replacing an
  existing canonical reference requires an explicit confirmation;
* preservation — replaced or demoted references are kept, never deleted as a side effect;
* safe deletion — an asset that something still points at cannot be deleted until it is unlinked.
"""
from __future__ import annotations

from typing import Any, Iterable

from app import storage
from app.models import (
    AppearanceSelection, CanonicalChange, Character, Location, Prop, Project, ReferenceAsset, Scene, StateChange,
    WardrobeVariant, utc_now,
)

SUBJECT_FIELDS = {"character": "characters", "location": "locations", "prop": "props"}


class StudioError(Exception):
    def __init__(self, code: str, message: str, detail: Any = None, status: int = 409):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


# --------------------------------------------------------------------------- lookups

def subjects(project: Project, subject_type: str) -> list[Any]:
    return getattr(project, SUBJECT_FIELDS[subject_type])


def find_subject(project: Project, subject_type: str | None, subject_id: str | None) -> Any | None:
    if not subject_type or not subject_id or subject_type not in SUBJECT_FIELDS:
        return None
    return next((item for item in subjects(project, subject_type) if item.id == subject_id), None)


def require_subject(project: Project, subject_type: str, subject_id: str) -> Any:
    found = find_subject(project, subject_type, subject_id)
    if found is None:
        raise StudioError("SUBJECT_NOT_FOUND", f"No {subject_type} with id '{subject_id}' in this project.",
                          status=404)
    return found


def find_asset(project: Project, asset_id: str | None) -> ReferenceAsset | None:
    return next((a for a in project.assets if a.id == asset_id), None)


def require_asset(project: Project, asset_id: str) -> ReferenceAsset:
    asset = find_asset(project, asset_id)
    if asset is None:
        raise StudioError("ASSET_NOT_FOUND", f"No reference asset with id '{asset_id}' in this project.", status=404)
    return asset


def find_wardrobe(character: Character, wardrobe_id: str | None) -> WardrobeVariant | None:
    return next((w for w in character.wardrobe if w.id == wardrobe_id), None)


def canonical_holder(project: Project, subject_type: str, subject_id: str,
                     wardrobe_id: str | None = None) -> Any:
    """The record that owns a canonical pointer: a subject, or one of a character's wardrobe variants."""
    subject = require_subject(project, subject_type, subject_id)
    if not wardrobe_id:
        return subject
    if subject_type != "character":
        raise StudioError("WARDROBE_NOT_ALLOWED", "Only characters have wardrobe variants.", status=400)
    variant = find_wardrobe(subject, wardrobe_id)
    if variant is None:
        raise StudioError("WARDROBE_NOT_FOUND",
                          f"Character '{subject.name}' has no wardrobe variant '{wardrobe_id}'.", status=404)
    return variant


def label_for(project: Project, subject_type: str | None, subject_id: str | None) -> str:
    found = find_subject(project, subject_type, subject_id)
    return getattr(found, "name", subject_id or "")


# --------------------------------------------------------------------------- subject CRUD

def _touch(record: Any) -> Any:
    record.updated_at = utc_now()
    return record


def create_character(project: Project, data: Any) -> Character:
    name = " ".join(data.name.split())
    if any(c.name.strip().upper() == name.upper() for c in project.characters):
        raise StudioError("DUPLICATE_NAME", f"A character named '{name}' already exists.", status=400)
    character = Character(
        id=storage.short_id(), name=name, cue_name=name.upper(), origin="manual",
        description=data.description.strip() or "To be defined in the character bible.",
        age_range=data.age_range.strip(), physical_description=data.physical_description.strip(),
        facial_traits=data.facial_traits.strip(), hairstyle=data.hairstyle.strip(),
        body_type=data.body_type.strip(), personality=data.personality.strip(),
        visual_signature=data.visual_signature.strip(), voice_direction=data.voice_direction.strip(),
        notes=data.notes.strip(), manually_edited=True,
    )
    project.characters.append(character)
    return character


def create_location(project: Project, data: Any) -> Location:
    name = " ".join(data.name.split())
    if any(l.name.strip().upper() == name.upper() for l in project.locations):
        raise StudioError("DUPLICATE_NAME", f"A location named '{name}' already exists.", status=400)
    location = Location(
        id=storage.short_id(), name=name, description=data.description.strip(),
        interior_exterior=data.interior_exterior.strip() or "Unspecified",
        time_of_day_variants=_clean_list(data.time_of_day_variants),
        geography_notes=data.geography_notes.strip(),
        continuity_requirements=data.continuity_requirements.strip(), notes=data.notes.strip(),
        scene_ids=_valid_scene_ids(project, data.scene_ids),
    )
    project.locations.append(location)
    return location


def create_prop(project: Project, data: Any) -> Prop:
    name = " ".join(data.name.split())
    if any(p.name.strip().upper() == name.upper() for p in project.props):
        raise StudioError("DUPLICATE_NAME", f"A prop named '{name}' already exists.", status=400)
    if data.owner_character_id:
        require_subject(project, "character", data.owner_character_id)
    prop = Prop(
        id=storage.short_id(), name=name, description=data.description.strip(),
        prop_type=data.prop_type.strip(), owner_character_id=data.owner_character_id,
        continuity_requirements=data.continuity_requirements.strip(), notes=data.notes.strip(),
        scene_ids=_valid_scene_ids(project, data.scene_ids),
    )
    project.props.append(prop)
    return prop


def create_wardrobe(project: Project, character: Character, data: Any) -> WardrobeVariant:
    name = " ".join(data.name.split())
    if any(w.name.strip().upper() == name.upper() for w in character.wardrobe):
        raise StudioError("DUPLICATE_NAME", f"'{character.name}' already has a wardrobe variant named '{name}'.",
                          status=400)
    variant = WardrobeVariant(id=storage.short_id(), name=name, description=data.description.strip(),
                              scene_numbers=sorted({int(n) for n in data.scene_numbers}),
                              notes=data.notes.strip())
    character.wardrobe.append(variant)
    _touch(character)
    return variant


def apply_edit(record: Any, data: Any, skip: Iterable[str] = ("base_revision",)) -> list[str]:
    """Apply only the fields that were sent. Returns the names that changed."""
    changes = data.model_dump(exclude_unset=True, exclude=set(skip))
    applied = []
    for key, value in changes.items():
        if value is None or not hasattr(record, key):
            continue
        setattr(record, key, " ".join(value.split()) if isinstance(value, str) and key == "name"
                else value.strip() if isinstance(value, str) else value)
        applied.append(key)
    if applied:
        _touch(record)
    return applied


def _clean_list(values: Iterable[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        text = " ".join(str(value).split())[:120]
        if text and text not in result:
            result.append(text)
    return result


def _valid_scene_ids(project: Project, scene_ids: Iterable[str] | None) -> list[str]:
    known = {s.id for s in project.scenes}
    unknown = [s for s in (scene_ids or []) if s not in known]
    if unknown:
        raise StudioError("SCENE_NOT_FOUND", f"These scene ids are not in this project: {', '.join(unknown[:5])}",
                          status=400)
    result: list[str] = []
    for scene_id in scene_ids or []:
        if scene_id not in result:
            result.append(scene_id)
    return result


def delete_subject(project: Project, subject_type: str, subject_id: str) -> dict[str, Any]:
    """Remove a subject. Its reference images are preserved and simply unassigned."""
    subject = require_subject(project, subject_type, subject_id)
    unassigned, scenes_touched, props_touched = [], [], []
    for asset in project.assets:
        if asset.subject_type == subject_type and asset.subject_id == subject_id:
            asset.subject_type = None
            asset.subject_id = None
            asset.wardrobe_id = None
            asset.updated_at = utc_now()
            unassigned.append(asset.id)
    for scene in project.scenes:
        before = (list(scene.character_ids), scene.location_id, list(scene.prop_ids),
                  len(scene.appearance_selections))
        if subject_type == "character":
            scene.character_ids = [c for c in scene.character_ids if c != subject_id]
            scene.appearance_selections = [s for s in scene.appearance_selections
                                           if s.character_id != subject_id]
        elif subject_type == "location" and scene.location_id == subject_id:
            scene.location_id = None
            scene.location_reference_id = None
        elif subject_type == "prop":
            scene.prop_ids = [p for p in scene.prop_ids if p != subject_id]
        if before != (list(scene.character_ids), scene.location_id, list(scene.prop_ids),
                      len(scene.appearance_selections)):
            scenes_touched.append(scene.id)
    if subject_type == "character":
        for prop in project.props:
            if prop.owner_character_id == subject_id:
                prop.owner_character_id = None
                _touch(prop)
                props_touched.append(prop.id)
    subjects(project, subject_type).remove(subject)
    return {"deleted": subject_id, "name": getattr(subject, "name", ""),
            "assets_unassigned": unassigned, "scenes_updated": scenes_touched,
            "props_updated": props_touched,
            "note": "Reference image files were kept; they are now unassigned in the library."}


def delete_wardrobe(project: Project, character: Character, wardrobe_id: str) -> dict[str, Any]:
    variant = find_wardrobe(character, wardrobe_id)
    if variant is None:
        raise StudioError("WARDROBE_NOT_FOUND", f"Character '{character.name}' has no wardrobe variant "
                                                f"'{wardrobe_id}'.", status=404)
    unassigned = []
    for asset in project.assets:
        if asset.wardrobe_id == wardrobe_id:
            asset.wardrobe_id = None
            asset.updated_at = utc_now()
            unassigned.append(asset.id)
    for scene in project.scenes:
        for selection in scene.appearance_selections:
            if selection.wardrobe_id == wardrobe_id:
                selection.wardrobe_id = None
    character.wardrobe.remove(variant)
    _touch(character)
    return {"deleted": wardrobe_id, "name": variant.name, "assets_unassigned": unassigned,
            "note": "Images that showed this costume were kept and are now attached to the character only."}


# --------------------------------------------------------------------------- assets

def next_version(project: Project, subject_type: str | None, subject_id: str | None,
                 wardrobe_id: str | None, role: str) -> int:
    """Versions count up per subject + wardrobe + role, and never reuse a number."""
    if not subject_type or not subject_id:
        return 1
    existing = [a.version for a in project.assets
                if a.subject_type == subject_type and a.subject_id == subject_id
                and a.wardrobe_id == wardrobe_id and a.role == role]
    return (max(existing) + 1) if existing else 1


def validate_association(project: Project, association: Any) -> dict[str, Any]:
    """Check a requested subject/wardrobe/role association without applying it."""
    subject_type = association.subject_type
    subject_id = association.subject_id
    wardrobe_id = association.wardrobe_id
    if wardrobe_id and not (subject_type == "character" and subject_id):
        raise StudioError("WARDROBE_WITHOUT_CHARACTER",
                          "A wardrobe variant can only be set together with its character.", status=400)
    if subject_type and not subject_id:
        raise StudioError("SUBJECT_ID_REQUIRED", f"Give the id of the {subject_type} this image belongs to.",
                          status=400)
    if subject_id and not subject_type:
        raise StudioError("SUBJECT_TYPE_REQUIRED", "Give subject_type together with subject_id.", status=400)
    if subject_type and subject_id:
        subject = require_subject(project, subject_type, subject_id)
        if wardrobe_id and find_wardrobe(subject, wardrobe_id) is None:
            raise StudioError("WARDROBE_NOT_FOUND",
                              f"Character '{subject.name}' has no wardrobe variant '{wardrobe_id}'.", status=404)
    return {"subject_type": subject_type, "subject_id": subject_id, "wardrobe_id": wardrobe_id,
            "role": association.role}


def duplicate_of(project: Project, sha256: str) -> ReferenceAsset | None:
    return next((a for a in project.assets if a.sha256 == sha256), None)


def record_asset(project: Project, asset: ReferenceAsset) -> ReferenceAsset:
    asset.version = next_version(project, asset.subject_type, asset.subject_id, asset.wardrobe_id, asset.role)
    project.assets.append(asset)
    return asset


def change_state(project: Project, asset: ReferenceAsset, new_state: str, note: str = "",
                 confirm_demote_canonical: bool = False) -> dict[str, Any]:
    """draft <-> approved <-> rejected, with the canonical pointer kept consistent."""
    if new_state == asset.state:
        return {"asset": asset, "changed": False, "unlinked": []}
    canonical_refs = [r for r in references_to(project, asset.id) if r["kind"] == "canonical"]
    unlinked: list[dict[str, Any]] = []

    if new_state != "approved" and canonical_refs:
        if new_state == "rejected":
            raise StudioError(
                "REFERENCED_AS_CANONICAL",
                "This image is the canonical reference for "
                + ", ".join(r["label"] for r in canonical_refs)
                + ". Choose a different canonical reference first, then reject this one.",
                detail=canonical_refs)
        if not confirm_demote_canonical:
            raise StudioError(
                "REFERENCED_AS_CANONICAL",
                "This image is the canonical reference for "
                + ", ".join(r["label"] for r in canonical_refs)
                + ". Send confirm_demote_canonical=true to clear that and move it back to draft.",
                detail=canonical_refs)
        for reference in canonical_refs:
            holder = _holder_by_reference(project, reference)
            if holder is not None:
                holder.canonical_history.append(CanonicalChange(from_asset_id=asset.id, to_asset_id=None,
                                                                note=note or "canonical cleared on demotion"))
                holder.canonical_reference_id = None
                _touch(holder)
                unlinked.append({"kind": "canonical", "label": reference["label"]})

    if new_state != "approved":
        for scene in project.scenes:
            if scene.location_reference_id == asset.id:
                scene.location_reference_id = None
                unlinked.append({"kind": "scene_location", "label": f"Scene {scene.number}"})
            kept = [s for s in scene.appearance_selections if s.asset_id != asset.id]
            if len(kept) != len(scene.appearance_selections):
                scene.appearance_selections = kept
                unlinked.append({"kind": "scene_appearance", "label": f"Scene {scene.number}"})

    previous = asset.state
    asset.state = new_state                               # type: ignore[assignment]
    asset.state_history.append(StateChange(from_state=previous, to_state=new_state, note=note.strip()[:500]))
    asset.updated_at = utc_now()
    if new_state == "approved":
        asset.approved_at = utc_now()
    elif new_state == "rejected":
        asset.rejected_at = utc_now()
    return {"asset": asset, "changed": True, "unlinked": unlinked}


def set_canonical(project: Project, subject_type: str, subject_id: str, asset_id: str,
                  wardrobe_id: str | None = None, confirm_replace: bool = False,
                  note: str = "") -> dict[str, Any]:
    """Select the one canonical reference. Replacing an existing one must be explicit."""
    holder = canonical_holder(project, subject_type, subject_id, wardrobe_id)
    asset = require_asset(project, asset_id)
    if asset.state != "approved":
        raise StudioError("ASSET_NOT_APPROVED",
                          f"Only an approved image can be canonical; this one is '{asset.state}'. "
                          "Approve it first.")
    if asset.subject_type != subject_type or asset.subject_id != subject_id:
        raise StudioError("ASSET_WRONG_SUBJECT",
                          f"That image is not attached to this {subject_type}. Change its association first.")
    if (asset.wardrobe_id or None) != (wardrobe_id or None):
        where = f"wardrobe variant '{wardrobe_id}'" if wardrobe_id else "the character's base look"
        raise StudioError("ASSET_WRONG_WARDROBE", f"That image is not attached to {where}.")
    current = holder.canonical_reference_id
    if current and current != asset.id and not confirm_replace:
        existing = find_asset(project, current)
        raise StudioError(
            "CANONICAL_EXISTS",
            f"'{getattr(holder, 'name', subject_id)}' already has a canonical reference"
            + (f" (version {existing.version}, approved {existing.approved_at})" if existing else "")
            + ". Send confirm_replace=true to replace it. The old image is kept.",
            detail={"current_asset_id": current, "new_asset_id": asset.id})
    if current != asset.id:
        holder.canonical_history.append(CanonicalChange(from_asset_id=current, to_asset_id=asset.id,
                                                        note=note.strip()[:500]))
        holder.canonical_reference_id = asset.id
        _touch(holder)
    return {"holder": holder, "asset": asset, "replaced": current if current != asset.id else None}


def clear_canonical(project: Project, subject_type: str, subject_id: str, wardrobe_id: str | None = None,
                    note: str = "") -> dict[str, Any]:
    holder = canonical_holder(project, subject_type, subject_id, wardrobe_id)
    previous = holder.canonical_reference_id
    if previous:
        holder.canonical_history.append(CanonicalChange(from_asset_id=previous, to_asset_id=None,
                                                        note=note.strip()[:500] or "canonical cleared"))
        holder.canonical_reference_id = None
        _touch(holder)
    return {"holder": holder, "cleared": previous}


def _holder_by_reference(project: Project, reference: dict[str, Any]) -> Any | None:
    if reference.get("wardrobe_id"):
        character = find_subject(project, "character", reference["subject_id"])
        return find_wardrobe(character, reference["wardrobe_id"]) if character else None
    return find_subject(project, reference.get("subject_type"), reference.get("subject_id"))


# --------------------------------------------------------------------------- reference scanning

def references_to(project: Project, asset_id: str, include_jobs: bool = False) -> list[dict[str, Any]]:
    """Everything that points at this asset. `blocking` entries must be cleared before deletion."""
    found: list[dict[str, Any]] = []
    for subject_type, field in SUBJECT_FIELDS.items():
        for subject in getattr(project, field):
            if subject.canonical_reference_id == asset_id:
                found.append({"kind": "canonical", "blocking": True, "subject_type": subject_type,
                              "subject_id": subject.id, "wardrobe_id": None,
                              "label": f"{subject_type} '{subject.name}' (canonical)"})
            if subject_type == "character":
                for variant in subject.wardrobe:
                    if variant.canonical_reference_id == asset_id:
                        found.append({"kind": "canonical", "blocking": True, "subject_type": "character",
                                      "subject_id": subject.id, "wardrobe_id": variant.id,
                                      "label": f"{subject.name} / wardrobe '{variant.name}' (canonical)"})
    for scene in project.scenes:
        if scene.location_reference_id == asset_id:
            found.append({"kind": "scene_location", "blocking": True, "subject_type": "scene",
                          "subject_id": scene.id, "label": f"Scene {scene.number} location reference"})
        for selection in scene.appearance_selections:
            if selection.asset_id == asset_id:
                found.append({"kind": "scene_appearance", "blocking": True, "subject_type": "scene",
                              "subject_id": scene.id,
                              "label": f"Scene {scene.number} appearance selection"})
    for subject_type, field in SUBJECT_FIELDS.items():
        for subject in getattr(project, field):
            holders = [subject] + (list(subject.wardrobe) if subject_type == "character" else [])
            for holder in holders:
                for change in holder.canonical_history:
                    if asset_id in (change.from_asset_id, change.to_asset_id):
                        found.append({"kind": "history", "blocking": False, "subject_type": subject_type,
                                      "subject_id": subject.id,
                                      "label": f"{getattr(holder, 'name', '')} canonical history"})
                        break
    if include_jobs:
        asset = find_asset(project, asset_id)
        for job in storage.list_jobs(project.id):
            request = job.request
            if getattr(request, "reference_asset_id", None) == asset_id or (
                    asset and request.reference_path and request.reference_path == asset.path):
                found.append({"kind": "generation_job", "blocking": False, "subject_type": "job",
                              "subject_id": job.id,
                              "label": f"generation job {job.id} ({job.status}) used it as a reference"})
    return found


def assert_deletable(project: Project, asset_id: str, confirm: bool) -> list[dict[str, Any]]:
    """Raise unless this asset may be deleted right now. Changes nothing."""
    references = references_to(project, asset_id, include_jobs=True)
    blocking = [r for r in references if r["blocking"]]
    if blocking:
        raise StudioError("ASSET_IN_USE",
                          "This image is still used by " + ", ".join(r["label"] for r in blocking)
                          + ". Unlink it there first; nothing was deleted.", detail=blocking)
    if not confirm:
        raise StudioError("CONFIRMATION_REQUIRED",
                          "Deleting a reference image also deletes the stored file. Send confirm=true to go "
                          "ahead; the project is snapshotted first.", detail=references, status=400)
    return references


def delete_asset(project: Project, asset_id: str, confirm: bool) -> dict[str, Any]:
    """Delete a reference image. Only possible once nothing points at it, and only on confirmation."""
    asset = require_asset(project, asset_id)
    references = assert_deletable(project, asset_id, confirm)
    # Remove the record first: an orphaned file is recoverable, a record pointing at nothing is not.
    project.assets = [a for a in project.assets if a.id != asset_id]
    return {"asset": asset, "path": asset.path, "references_seen": references}


def resolve_reference_asset(project: Project, asset_id: str, allow_draft: bool = False) -> dict[str, Any]:
    """Phase 4 bridge: turn a library asset into a verified, project-relative image path."""
    asset = require_asset(project, asset_id)
    if asset.state == "rejected":
        raise StudioError("ASSET_REJECTED", "That reference image is marked rejected; pick another one.")
    if asset.state != "approved" and not allow_draft:
        raise StudioError("ASSET_NOT_APPROVED",
                          "That reference image is still a draft. Approve it, or send "
                          "allow_draft_reference=true to use it anyway.")
    status = storage.asset_file_status(project.id, asset.path, asset.sha256)
    if status != "ok":
        raise StudioError("ASSET_FILE_" + status.upper(),
                          f"The stored file for this reference image is {status} "
                          f"({asset.path}). Nothing was queued.", status=409)
    return {"asset": asset, "path": asset.path}


# --------------------------------------------------------------------------- scene integration

def set_scene_associations(project: Project, scene: Scene, data: Any) -> dict[str, Any]:
    """Attach characters/location/props and pick approved looks. The screenplay text is untouched."""
    problems: list[str] = []
    if data.character_ids is not None:
        for character_id in data.character_ids:
            require_subject(project, "character", character_id)
        scene.character_ids = list(dict.fromkeys(data.character_ids))
    if data.prop_ids is not None:
        for prop_id in data.prop_ids:
            require_subject(project, "prop", prop_id)
        scene.prop_ids = list(dict.fromkeys(data.prop_ids))
    if data.clear_location:
        scene.location_id = None
        scene.location_reference_id = None
    elif data.location_id is not None:
        require_subject(project, "location", data.location_id)
        if scene.location_id != data.location_id:
            scene.location_reference_id = None
        scene.location_id = data.location_id

    if data.location_reference_id is not None:
        if not scene.location_id:
            raise StudioError("SCENE_HAS_NO_LOCATION",
                              "Attach a location to this scene before selecting a location reference.", status=400)
        asset = require_asset(project, data.location_reference_id)
        _require_approved_for_scene(asset, "location reference")
        if asset.subject_type != "location" or asset.subject_id != scene.location_id:
            raise StudioError("ASSET_WRONG_SUBJECT",
                              "That image is not attached to this scene's location.")
        scene.location_reference_id = asset.id

    if data.appearance_selections is not None:
        selections: list[AppearanceSelection] = []
        for selection in data.appearance_selections:
            character = require_subject(project, "character", selection.character_id)
            if selection.character_id not in scene.character_ids:
                scene.character_ids.append(selection.character_id)
            if selection.wardrobe_id and find_wardrobe(character, selection.wardrobe_id) is None:
                raise StudioError("WARDROBE_NOT_FOUND",
                                  f"Character '{character.name}' has no wardrobe variant "
                                  f"'{selection.wardrobe_id}'.", status=404)
            if selection.asset_id:
                asset = require_asset(project, selection.asset_id)
                _require_approved_for_scene(asset, f"appearance for '{character.name}'")
                if asset.subject_type != "character" or asset.subject_id != selection.character_id:
                    raise StudioError("ASSET_WRONG_SUBJECT",
                                      f"That image is not attached to '{character.name}'.")
                if (asset.wardrobe_id or None) != (selection.wardrobe_id or None):
                    raise StudioError("ASSET_WRONG_WARDROBE",
                                      "The selected image does not belong to the chosen wardrobe variant.")
            selections.append(AppearanceSelection(character_id=selection.character_id,
                                                  wardrobe_id=selection.wardrobe_id,
                                                  asset_id=selection.asset_id))
        scene.appearance_selections = selections
    scene.manually_edited = True
    return {"scene": scene, "problems": problems}


def _require_approved_for_scene(asset: ReferenceAsset, what: str) -> None:
    if asset.state != "approved":
        raise StudioError("ASSET_NOT_APPROVED",
                          f"A scene can only use an approved image as its {what}; that one is '{asset.state}'.")


# --------------------------------------------------------------------------- read models

def asset_view(project: Project, asset: ReferenceAsset) -> dict[str, Any]:
    references = references_to(project, asset.id)
    return asset.model_dump() | {
        "file_status": storage.asset_file_status(project.id, asset.path, asset.sha256),
        "subject_label": label_for(project, asset.subject_type, asset.subject_id),
        "is_canonical": any(r["kind"] == "canonical" for r in references),
        "reference_count": len([r for r in references if r["blocking"]]),
    }


def studio_overview(project: Project) -> dict[str, Any]:
    """Everything the studio UI needs in one call."""
    assets = [asset_view(project, asset) for asset in project.assets]
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for view in assets:
        key = f"{view['subject_type']}:{view['subject_id']}"
        by_subject.setdefault(key, []).append(view)
    return {
        "revision": project.revision,
        "characters": [character.model_dump() | {
            "assets": by_subject.get(f"character:{character.id}", []),
            "canonical": next((a for a in assets if a["id"] == character.canonical_reference_id), None),
        } for character in project.characters],
        "locations": [location.model_dump() | {
            "assets": by_subject.get(f"location:{location.id}", []),
            "canonical": next((a for a in assets if a["id"] == location.canonical_reference_id), None),
        } for location in project.locations],
        "props": [prop.model_dump() | {
            "assets": by_subject.get(f"prop:{prop.id}", []),
            "canonical": next((a for a in assets if a["id"] == prop.canonical_reference_id), None),
        } for prop in project.props],
        "assets": assets,
        "scenes": [{"id": s.id, "number": s.number, "heading": s.heading, "location": s.location,
                    "cue_names": s.characters, "character_ids": s.character_ids, "location_id": s.location_id,
                    "prop_ids": s.prop_ids, "location_reference_id": s.location_reference_id,
                    "appearance_selections": [a.model_dump() for a in s.appearance_selections]}
                   for s in project.scenes],
        "counts": {"characters": len(project.characters), "locations": len(project.locations),
                   "props": len(project.props), "assets": len(project.assets),
                   "approved": sum(a["state"] == "approved" for a in assets),
                   "draft": sum(a["state"] == "draft" for a in assets),
                   "rejected": sum(a["state"] == "rejected" for a in assets)},
    }
