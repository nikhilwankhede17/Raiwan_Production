# Architecture

## Principle: production system first, model wrappers second
The difficult part is not a single prompt. It is maintaining a consistent set of creative decisions across dozens or hundreds of short clips and managing assets, retries, approvals and edit decisions. The application owns the production data; media models are replaceable workers.

## Production pipeline

1. **Project setup** — title, target duration, aspect ratio, language, frame rate, visual style, output quality and local hardware profile.
2. **Screenplay ingestion** — save original source untouched; parse scene headings, action blocks, dialogue cues and transitions; report uncertain parses.
3. **Story bible** — logline, tone, themes, timeline, continuity rules and reference palette. Optional local LLM may draft this from the screenplay, but the user approves it.
4. **Character bible** — stable character IDs, age range, face/hair/body descriptors, wardrobe by scene, props, voice direction, emotional baseline, reference images and allowed variants.
5. **Location/prop bible** — stable IDs and reference images for rooms, vehicles, key objects, lighting/time-of-day variants and geography.
6. **Shot list** — shot ID, scene ID, lens/framing language, subject, action, composition, camera movement, lighting, duration, continuity constraints, source dialogue, prompt, model/workflow version, seed and approval state.
7. **Storyboard/reference generation** — make still frames before expensive motion generation. Reject inconsistent faces, wardrobe, props, screen direction and geography early.
8. **Video generation** — generate short shots from approved references, generally a few seconds at a time; queue jobs with progress, cancellation, retries, VRAM/error capture and reproducible settings.
9. **Voice/dialogue** — assign a licensed local voice to each fictional character; render dialogue line by line; record sample rate and line timing. Use consent-based voice cloning only, never clone a real voice without authorization.
10. **Lip-sync / performance** — optional later module; keep it replaceable because workflows and hardware needs vary. Prefer shots that do not require precise lip motion when resources are limited.
11. **Sound design** — dialogue cleanup, room tone, ambience, Foley/SFX, music cues, loudness and ducking. Track the source/license for each audio asset.
12. **Editorial timeline** — trim approved shots, arrange cuts, set durations, overlay titles/subtitles, mix audio and define transitions.
13. **Final render** — use FFmpeg to create the selected MP4 profile. Store render command, output hash and source asset manifest.
14. **Quality control** — failed/missing clip check, black-frame/frozen-frame check, audio peaks/silence, subtitle alignment, aspect ratio, runtime and continuity checklist.

## Main data objects

- `Project`: settings, original screenplay, version, status and paths.
- `Scene`: scene heading, time/location, cast, action/dialogue blocks and continuity notes. Phase 3 adds production-only links (`character_ids`, `location_id`, `prop_ids`, `location_reference_id`, `appearance_selections`) that never alter the parsed screenplay.
- `Character`: stable ID, canon appearance, wardrobe variants, voice assignment and reference media. Implemented as identity fields (age range, physical description, facial traits, hairstyle, body type, personality, voice direction) plus `wardrobe: [WardrobeVariant]` and one `canonical_reference_id`.
- `WardrobeVariant`: a costume/appearance change with its own description, scene numbers and canonical reference. Adding one never rewrites the character's identity fields.
- `Location` / `Prop`: stable ID, canon design and variants — description, continuity requirements, geography notes, scene links and their own canonical reference.
- `ReferenceAsset`: one image in the library — `path` (project-relative, write-once), `sha256`, `bytes`, `width`/`height`, `image_format`, `source` (`import`/`generated`) with `source_detail`/`job_id`, `license`, association (`subject_type`, `subject_id`, `wardrobe_id`, `role`), `state` (`draft`/`approved`/`rejected`) with timestamps and `state_history`, and `version` counted per subject + wardrobe + role.
- `Shot`: scene, coverage type, prompt, references, seed, model/workflow, expected duration, output media, notes, version and approval status.
- `GenerationJob`: queued/running/succeeded/failed/cancelled, provider, progress, timestamps, log and output paths. Stored per job outside `project.json`.
- `AudioCue`: dialogue/SFX/ambience/music, source/license, timeline start, gain and fade.
- `Timeline`: ordered clips, trims, transitions, audio tracks, subtitles and render profile.
- `ModelRegistryEntry`: local path, model/version, source, code license, weight license, commercial-use flags, required VRAM, disk size and known constraints.

## Reference approval lifecycle (Phase 3)

Every imported or generated image enters as a **draft**. Approval is a deliberate human checkpoint, and the canonical look is a separate, explicit choice:

1. **draft** — validated and stored, usable for experiments only.
2. **approved** — `approved_at` recorded; now eligible to become canonical or be selected by a scene.
3. **rejected** — `rejected_at` recorded; never used as a reference.
4. **canonical** — one approved asset per character, wardrobe variant, location or prop. Replacing it requires an explicit confirmation; the previous asset is kept and the change is appended to `canonical_history`.

Guard rails: a canonical asset cannot be rejected, and demoting it to draft requires a second confirmation that clears the pointer. When an asset stops being approved, scene selections that used it are dropped and reported back. An asset that is canonical or scene-selected cannot be deleted until it is unlinked, and deletion is confirmed, snapshotted, record-first, then file.

## Phase 3 → Phase 4 integration

A generation request names **exactly one** reference source: `reference_path` (a project-relative file) or `reference_asset_id` (a library asset). Sending both is rejected. A library asset must be `approved` unless `allow_draft_reference` is set, and a `rejected` asset is never usable. If the selected workflow has no `reference_image` node mapped, the job is refused with an explanation — the app never substitutes an arbitrary image, and never assumes a workflow supports references. Generated outputs can be copied back into the library as drafts, so they pass through the same validation, approval and canonical rules as imported images.

## Adapter contract
A generator adapter must declare `capabilities()`, `validate_config()`, `submit(job)`, `status(job_id)`, `cancel(job_id)` and `collect_outputs(job_id)`. Keep prompts, model identifiers, seeds and outputs in the job record. No silent cloud fallback.

Implemented: `app/comfy.py` (`ComfyAdapter`, image generation via a localhost-only ComfyUI). It refuses any non-loopback URL at construction, so there is no path to a remote endpoint. Workflows are not interchangeable: each imported API workflow is pinned by SHA-256 and carries its own node mapping and validation result, and a job records the workflow hash it used.

## Persistence and directories

```text
workspace/projects/<project-id>/
  project.json                      # current editable state; atomic writes; `revision` counter for conflict detection
  screenplay/original.txt           # first import, read-only, never overwritten (implemented)
  screenplay/versions/vNNNN.txt     # later script versions, write-once, SHA-256 in project.json (implemented)
  snapshots/<id>.json + index.json  # full production-data snapshots (implemented)
  workflows/<workflow-id>.json      # imported ComfyUI API workflows, write-once, SHA-256 pinned (implemented)
  jobs/<job-id>.json                # generation jobs with their own logs, outside project.json (implemented)
  assets/<asset-id>.<ext>           # reference library images, write-once + read-only, SHA-256 in project.json (implemented)
  references/input/                 # reference images you supply (implemented)
  references/<job-id>/              # generated reference frames + metadata.json (implemented)
  storyboards/<job-id>/             # generated storyboard frames + metadata.json (implemented)
  bibles/story.json
  bibles/characters.json
  bibles/locations.json
  shots/shot-list.json
  storyboards/<shot-id>/
  media/video/<shot-id>/
  media/audio/dialogue/<character-id>/
  media/audio/sfx/
  media/audio/music/
  edit/timeline.json
  renders/
  reports/continuity.json
  manifests/assets.json
```

Keep source files, generated candidates and approved finals separate. Never make the only copy of a generated candidate the editable/final version.

Record-vs-file consistency rules used by the implementation:

- **Write-once binaries.** Script versions, snapshots, imported workflows and reference images are created with `open(..., "x")` and then made read-only, so an import can never overwrite an existing asset.
- **Atomic mutable writes.** `project.json`, job records and the manifest are written to a temporary file and `os.replace`d; readers retry briefly, because on Windows a reader can otherwise observe the swap.
- **Import ordering.** Validate → write the file → create the record → save the project. If the save fails, the just-written file is removed, so a file never exists without a record.
- **Delete ordering.** Check references → confirm → snapshot → remove the record → delete the file. An orphaned file is recoverable and shows up in the manifest; a record pointing at nothing is not, so the record goes first.
- **Integrity checks.** Every asset read can verify the stored SHA-256; `missing` and `modified` files are surfaced as validation errors rather than crashing a request.
