from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


ProjectStatus = Literal[
    "script_required",
    "script_ready",
    "plan_ready",
    "in_production",
    "review",
    "rendered",
]

ShotStatus = Literal["planned", "references_ready", "generated", "approved", "needs_regeneration"]

ElementType = Literal[
    "title_page",
    "scene_heading",
    "action",
    "character",
    "parenthetical",
    "dialogue",
    "transition",
    "centered",
    "section",
    "synopsis",
    "note",
    "page_break",
]

WarningSeverity = Literal["info", "warning", "error"]


class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class ScriptUpdate(BaseModel):
    script: str = Field(min_length=1, max_length=2_000_000)
    aspect_ratio: Literal["16:9", "9:16", "1:1", "2.39:1"] = "16:9"
    visual_style: str = Field(default="cinematic live-action realism", max_length=500)
    language: str = Field(default="English", max_length=80)
    note: str = Field(default="", max_length=300)
    base_revision: int | None = None


class ScriptElement(BaseModel):
    """One block of a parsed screenplay. `line` is 1-based in the source text."""
    type: ElementType
    text: str
    line: int
    character: str | None = None          # canonical cue name for character/parenthetical/dialogue
    extensions: list[str] = Field(default_factory=list)  # e.g. ["V.O.", "CONT'D"]
    dual: bool = False


class ParseWarning(BaseModel):
    id: str
    code: str
    severity: WarningSeverity = "warning"
    message: str
    line: int | None = None
    scene_id: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    acknowledged: bool = False


class Character(BaseModel):
    id: str
    name: str
    cue_name: str = ""                    # uppercase cue as written in the script, used to merge re-parses
    description: str = "To be defined in the character bible."
    visual_signature: str = ""
    voice_direction: str = ""
    notes: str = ""
    reference_images: list[str] = Field(default_factory=list)   # Phase 0 free-text list; superseded by assets
    scene_numbers: list[int] = Field(default_factory=list)
    dialogue_count: int = 0
    extensions_seen: list[str] = Field(default_factory=list)
    manually_edited: bool = False
    # ---- Phase 3: identity fields (kept separate from costume/appearance variants) ----
    age_range: str = ""
    physical_description: str = ""
    facial_traits: str = ""
    hairstyle: str = ""
    body_type: str = ""
    personality: str = ""
    wardrobe: list["WardrobeVariant"] = Field(default_factory=list)
    canonical_reference_id: str | None = None
    canonical_history: list["CanonicalChange"] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    origin: Literal["parser", "manual"] = "parser"


class Scene(BaseModel):
    id: str
    number: int
    heading: str
    body: str
    characters: list[str] = Field(default_factory=list)        # cue names from the screenplay (never rewritten)
    location: str = "Unspecified"
    time_of_day: str = "Unspecified"
    interior_exterior: str = "Unspecified"
    scene_tag: str | None = None          # scene number written in the script, e.g. "12A"
    start_line: int | None = None
    end_line: int | None = None
    elements: list[ScriptElement] = Field(default_factory=list)
    notes: str = ""
    manually_edited: bool = False
    # ---- Phase 3: production associations (additive; the screenplay is never modified) ----
    character_ids: list[str] = Field(default_factory=list)
    location_id: str | None = None
    prop_ids: list[str] = Field(default_factory=list)
    location_reference_id: str | None = None
    appearance_selections: list["AppearanceSelection"] = Field(default_factory=list)


class Shot(BaseModel):
    id: str
    scene_id: str
    scene_number: int
    number: int
    shot_type: str
    duration_seconds: float = 4.0
    characters: list[str] = Field(default_factory=list)
    visual_prompt: str
    motion_prompt: str
    audio_notes: str = "Dialogue and ambience to be added during sound pass."
    status: ShotStatus = "planned"
    media_path: str | None = None
    notes: str = "Review and adjust before production."
    manually_edited: bool = False
    updated_at: str = Field(default_factory=utc_now)


class ScriptVersion(BaseModel):
    number: int
    path: str                             # relative to the project directory
    sha256: str
    created_at: str = Field(default_factory=utc_now)
    source: Literal["import", "edit", "legacy_import"] = "edit"
    note: str = ""
    characters: int = 0


# ---- Phase 3: character / location / prop studio and the reference-image library ----

AssetState = Literal["draft", "approved", "rejected"]
SubjectType = Literal["character", "location", "prop"]
AssetRole = Literal[
    "face", "three_quarter", "profile", "full_body", "expression", "wardrobe",
    "location_wide", "location_detail", "lighting", "prop_turnaround", "other",
]


class StateChange(BaseModel):
    at: str = Field(default_factory=utc_now)
    from_state: str
    to_state: str
    note: str = ""


class CanonicalChange(BaseModel):
    """Audit trail for 'this is now the canonical look'. Replaced assets are kept, never deleted."""
    at: str = Field(default_factory=utc_now)
    from_asset_id: str | None = None
    to_asset_id: str | None = None
    note: str = ""


class ReferenceAsset(BaseModel):
    """One image in the project's reference library. The file itself is write-once on disk."""
    id: str
    path: str                             # project-relative, inside assets/
    original_filename: str = ""
    sha256: str
    bytes: int = 0
    width: int | None = None
    height: int | None = None
    image_format: str = ""                # PNG / JPEG / WEBP / ...
    imported_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    source: Literal["import", "generated"] = "import"
    source_detail: str = ""               # original file path, or the job output it came from
    job_id: str | None = None
    license: str = ""                     # yours to record: model/photo licence for this image
    license_url: str = ""
    notes: str = ""
    subject_type: SubjectType | None = None
    subject_id: str | None = None
    wardrobe_id: str | None = None        # set when the image shows a costume variant
    role: AssetRole = "other"
    state: AssetState = "draft"
    version: int = 1                      # per subject + wardrobe + role
    approved_at: str | None = None
    rejected_at: str | None = None
    state_history: list[StateChange] = Field(default_factory=list)
    integrity: str = "verified"           # "verified" | "stdlib-header" | "unverified"


class WardrobeVariant(BaseModel):
    """A costume/appearance change. Identity fields stay on the character."""
    id: str
    name: str
    description: str = ""
    scene_numbers: list[int] = Field(default_factory=list)
    notes: str = ""
    canonical_reference_id: str | None = None
    canonical_history: list[CanonicalChange] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class Location(BaseModel):
    id: str
    name: str
    description: str = ""
    interior_exterior: str = "Unspecified"
    time_of_day_variants: list[str] = Field(default_factory=list)
    geography_notes: str = ""
    continuity_requirements: str = ""
    notes: str = ""
    scene_ids: list[str] = Field(default_factory=list)
    canonical_reference_id: str | None = None
    canonical_history: list[CanonicalChange] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class Prop(BaseModel):
    id: str
    name: str
    description: str = ""
    prop_type: str = ""
    owner_character_id: str | None = None
    continuity_requirements: str = ""
    notes: str = ""
    scene_ids: list[str] = Field(default_factory=list)
    canonical_reference_id: str | None = None
    canonical_history: list[CanonicalChange] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class AppearanceSelection(BaseModel):
    """A scene's chosen approved look for one character."""
    character_id: str
    wardrobe_id: str | None = None
    asset_id: str | None = None


# ---- Phase 3 payloads ----

class CharacterCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    age_range: str = Field(default="", max_length=80)
    description: str = Field(default="", max_length=10_000)
    physical_description: str = Field(default="", max_length=10_000)
    facial_traits: str = Field(default="", max_length=5_000)
    hairstyle: str = Field(default="", max_length=2_000)
    body_type: str = Field(default="", max_length=2_000)
    personality: str = Field(default="", max_length=10_000)
    visual_signature: str = Field(default="", max_length=5_000)
    voice_direction: str = Field(default="", max_length=5_000)
    notes: str = Field(default="", max_length=10_000)
    base_revision: int | None = None


class WardrobeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=10_000)
    scene_numbers: list[int] = Field(default_factory=list, max_length=500)
    notes: str = Field(default="", max_length=10_000)
    base_revision: int | None = None


class WardrobeEdit(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=10_000)
    scene_numbers: list[int] | None = Field(default=None, max_length=500)
    notes: str | None = Field(default=None, max_length=10_000)
    base_revision: int | None = None


class LocationCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=10_000)
    interior_exterior: str = Field(default="Unspecified", max_length=20)
    time_of_day_variants: list[str] = Field(default_factory=list, max_length=50)
    geography_notes: str = Field(default="", max_length=10_000)
    continuity_requirements: str = Field(default="", max_length=10_000)
    notes: str = Field(default="", max_length=10_000)
    scene_ids: list[str] = Field(default_factory=list, max_length=500)
    base_revision: int | None = None


class LocationEdit(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=10_000)
    interior_exterior: str | None = Field(default=None, max_length=20)
    time_of_day_variants: list[str] | None = Field(default=None, max_length=50)
    geography_notes: str | None = Field(default=None, max_length=10_000)
    continuity_requirements: str | None = Field(default=None, max_length=10_000)
    notes: str | None = Field(default=None, max_length=10_000)
    scene_ids: list[str] | None = Field(default=None, max_length=500)
    base_revision: int | None = None


class PropCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=10_000)
    prop_type: str = Field(default="", max_length=120)
    owner_character_id: str | None = Field(default=None, max_length=64)
    continuity_requirements: str = Field(default="", max_length=10_000)
    notes: str = Field(default="", max_length=10_000)
    scene_ids: list[str] = Field(default_factory=list, max_length=500)
    base_revision: int | None = None


class PropEdit(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=10_000)
    prop_type: str | None = Field(default=None, max_length=120)
    owner_character_id: str | None = Field(default=None, max_length=64)
    continuity_requirements: str | None = Field(default=None, max_length=10_000)
    notes: str | None = Field(default=None, max_length=10_000)
    scene_ids: list[str] | None = Field(default=None, max_length=500)
    base_revision: int | None = None


class AssetAssociation(BaseModel):
    subject_type: SubjectType | None = None
    subject_id: str | None = Field(default=None, max_length=64)
    wardrobe_id: str | None = Field(default=None, max_length=64)
    role: AssetRole = "other"


class AssetImport(BaseModel):
    """Import one image. Supply exactly one of data_base64 or source_path."""
    filename: str = Field(default="", max_length=260)
    data_base64: str | None = Field(default=None, max_length=60_000_000)
    source_path: str | None = Field(default=None, max_length=1_000)
    association: AssetAssociation = Field(default_factory=AssetAssociation)
    license: str = Field(default="", max_length=300)
    license_url: str = Field(default="", max_length=500)
    notes: str = Field(default="", max_length=10_000)
    allow_duplicate: bool = False
    base_revision: int | None = None


class AssetFromJob(BaseModel):
    """Copy a generated job output into the reference library as a draft."""
    job_id: str = Field(min_length=1, max_length=64)
    output_index: int = Field(default=0, ge=0, le=999)
    association: AssetAssociation = Field(default_factory=AssetAssociation)
    notes: str = Field(default="", max_length=10_000)
    license: str = Field(default="", max_length=300)
    allow_duplicate: bool = False
    base_revision: int | None = None


class AssetEdit(BaseModel):
    subject_type: SubjectType | None = None
    subject_id: str | None = Field(default=None, max_length=64)
    wardrobe_id: str | None = Field(default=None, max_length=64)
    role: AssetRole | None = None
    license: str | None = Field(default=None, max_length=300)
    license_url: str | None = Field(default=None, max_length=500)
    notes: str | None = Field(default=None, max_length=10_000)
    clear_association: bool = False
    base_revision: int | None = None


class AssetStateChange(BaseModel):
    state: AssetState
    note: str = Field(default="", max_length=500)
    confirm_demote_canonical: bool = False
    base_revision: int | None = None


class CanonicalSelect(BaseModel):
    """Make one approved asset the canonical look. Replacing an existing one must be explicit."""
    asset_id: str = Field(min_length=1, max_length=64)
    confirm_replace: bool = False
    note: str = Field(default="", max_length=500)
    base_revision: int | None = None


class AssetDelete(BaseModel):
    confirm: bool = False
    base_revision: int | None = None


class SceneAssociations(BaseModel):
    character_ids: list[str] | None = Field(default=None, max_length=200)
    location_id: str | None = Field(default=None, max_length=64)
    prop_ids: list[str] | None = Field(default=None, max_length=200)
    location_reference_id: str | None = Field(default=None, max_length=64)
    appearance_selections: list[AppearanceSelection] | None = Field(default=None, max_length=200)
    clear_location: bool = False
    base_revision: int | None = None


class SnapshotInfo(BaseModel):
    id: str
    created_at: str
    revision: int
    reason: str
    label: str = ""
    path: str


# ---- Phase 4: ComfyUI workflows and image generation jobs ----

WorkflowRole = Literal["prompt", "negative_prompt", "seed", "width", "height", "reference_image"]
JobStatus = Literal["queued", "submitted", "running", "succeeded", "failed", "cancelled"]
GenerationTarget = Literal["character", "location", "shot", "freeform"]


class NodeFieldRef(BaseModel):
    """One explicitly mapped workflow input."""
    node_id: str = Field(min_length=1, max_length=64)
    input: str = Field(min_length=1, max_length=128)


class WorkflowMapping(BaseModel):
    prompt: NodeFieldRef | None = None
    negative_prompt: NodeFieldRef | None = None
    seed: NodeFieldRef | None = None
    width: NodeFieldRef | None = None
    height: NodeFieldRef | None = None
    reference_image: NodeFieldRef | None = None
    output_nodes: list[str] = Field(default_factory=list, max_length=50)


class ValidationProblem(BaseModel):
    code: str
    message: str
    role: str | None = None
    node_id: str | None = None


class WorkflowValidation(BaseModel):
    checked_at: str = Field(default_factory=utc_now)
    scope: Literal["offline", "live"] = "offline"
    valid: bool = False
    errors: list[ValidationProblem] = Field(default_factory=list)
    warnings: list[ValidationProblem] = Field(default_factory=list)
    comfyui_version: str | None = None
    comfyui_url: str | None = None


class WorkflowRecord(BaseModel):
    """An imported API-format workflow. Pinned by SHA-256: workflows are never interchangeable."""
    id: str
    name: str
    path: str                             # relative to the project directory; write-once file
    sha256: str
    created_at: str = Field(default_factory=utc_now)
    source_filename: str = ""
    node_count: int = 0
    class_types: dict[str, int] = Field(default_factory=dict)
    models: list[dict[str, str]] = Field(default_factory=list)
    mapping: WorkflowMapping = Field(default_factory=WorkflowMapping)
    mapping_set_at: str | None = None
    offline_validation: WorkflowValidation | None = None
    live_validation: WorkflowValidation | None = None

    @property
    def ready(self) -> bool:
        return bool(self.offline_validation and self.offline_validation.valid)


class WorkflowImport(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    workflow: dict[str, Any] | str = Field(description="API-format workflow JSON (object or text)")
    source_filename: str = Field(default="", max_length=260)
    base_revision: int | None = None


class MappingUpdate(BaseModel):
    mapping: WorkflowMapping
    base_revision: int | None = None


class JobOutput(BaseModel):
    index: int
    path: str                             # relative to the project directory
    filename: str
    bytes: int
    sha256: str
    node_id: str
    comfy_subfolder: str = ""
    comfy_type: str = "output"


class GenerationRequest(BaseModel):
    workflow_id: str = Field(min_length=1, max_length=64)
    prompt: str = Field(min_length=1, max_length=20_000)
    negative_prompt: str | None = Field(default=None, max_length=20_000)
    seed: int | None = Field(default=None, ge=0, le=2 ** 63 - 1)
    randomize_seed: bool = False
    width: int | None = Field(default=None, ge=64, le=8192)
    height: int | None = Field(default=None, ge=64, le=8192)
    reference_path: str | None = Field(default=None, max_length=400,
                                       description="Project-relative path of a reference image")
    reference_asset_id: str | None = Field(default=None, max_length=64,
                                           description="Phase 3 library asset to use as the reference image")
    allow_draft_reference: bool = Field(default=False,
                                        description="Opt in to using a not-yet-approved reference asset")
    target_type: GenerationTarget = "freeform"
    target_id: str | None = Field(default=None, max_length=64)
    label: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def _one_reference_source(self) -> "GenerationRequest":
        """Never leave it ambiguous which image goes into the workflow."""
        if self.reference_path and self.reference_asset_id:
            raise ValueError("Give either reference_path or reference_asset_id, not both.")
        return self


class GenerationJob(BaseModel):
    id: str
    project_id: str
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    status: JobStatus = "queued"
    provider: str = "comfyui"
    comfyui_url: str | None = None
    comfyui_version: str | None = None
    workflow_id: str
    workflow_name: str = ""
    workflow_sha256: str = ""
    prompt_id: str | None = None
    queue_position: int | None = None
    request: GenerationRequest
    seed_used: int | None = None
    models: list[dict[str, str]] = Field(default_factory=list)
    reference_uploaded_as: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    error: dict[str, Any] | None = None
    cancel_requested: bool = False
    log: list[str] = Field(default_factory=list)
    outputs: list[JobOutput] = Field(default_factory=list)
    metadata_path: str | None = None


class JobAction(BaseModel):
    randomize_seed: bool = False


class Project(BaseModel):
    id: str
    title: str
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    status: ProjectStatus = "script_required"
    revision: int = 0
    script: str = ""                      # text of the current script version (never the only copy)
    script_versions: list[ScriptVersion] = Field(default_factory=list)
    current_script_version: int | None = None
    breakdown_script_version: int | None = None
    aspect_ratio: str = "16:9"
    visual_style: str = "cinematic live-action realism"
    language: str = "English"
    characters: list[Character] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
    shots: list[Shot] = Field(default_factory=list)
    locations: list["Location"] = Field(default_factory=list)
    props: list["Prop"] = Field(default_factory=list)
    assets: list["ReferenceAsset"] = Field(default_factory=list)
    workflows: list["WorkflowRecord"] = Field(default_factory=list)
    parse_warnings: list[ParseWarning] = Field(default_factory=list)
    acknowledged_warnings: list[str] = Field(default_factory=list)
    production_notes: list[str] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)


# ---- Edit payloads (all fields optional; only supplied fields change) ----

class SceneEdit(BaseModel):
    base_revision: int | None = None
    heading: str | None = Field(default=None, min_length=1, max_length=300)
    location: str | None = Field(default=None, max_length=200)
    time_of_day: str | None = Field(default=None, max_length=80)
    interior_exterior: str | None = Field(default=None, max_length=20)
    characters: list[str] | None = None
    notes: str | None = Field(default=None, max_length=10_000)


class CharacterEdit(BaseModel):
    base_revision: int | None = None
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=10_000)
    visual_signature: str | None = Field(default=None, max_length=5_000)
    voice_direction: str | None = Field(default=None, max_length=5_000)
    notes: str | None = Field(default=None, max_length=10_000)
    # Phase 3 identity fields
    age_range: str | None = Field(default=None, max_length=80)
    physical_description: str | None = Field(default=None, max_length=10_000)
    facial_traits: str | None = Field(default=None, max_length=5_000)
    hairstyle: str | None = Field(default=None, max_length=2_000)
    body_type: str | None = Field(default=None, max_length=2_000)
    personality: str | None = Field(default=None, max_length=10_000)


class ShotEdit(BaseModel):
    base_revision: int | None = None
    shot_type: str | None = Field(default=None, min_length=1, max_length=80)
    duration_seconds: float | None = Field(default=None, gt=0, le=120)
    characters: list[str] | None = None
    visual_prompt: str | None = Field(default=None, max_length=10_000)
    motion_prompt: str | None = Field(default=None, max_length=10_000)
    audio_notes: str | None = Field(default=None, max_length=10_000)
    status: ShotStatus | None = None
    notes: str | None = Field(default=None, max_length=10_000)


class BreakdownRequest(BaseModel):
    base_revision: int | None = None
    force: bool = False


class SnapshotCreate(BaseModel):
    label: str = Field(default="", max_length=120)


class RevisionOnly(BaseModel):
    base_revision: int | None = None


class WarningAck(BaseModel):
    base_revision: int | None = None
    acknowledged: bool = True
