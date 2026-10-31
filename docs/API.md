# HTTP API

Local-only JSON API served by `uvicorn app.main:app` (default `http://127.0.0.1:8000`). Interactive docs: `/docs`.

Conventions:

- **Revision checks.** Any write may carry `base_revision`. If the project changed in the meantime the request fails with `409` and `{"code": "REVISION_CONFLICT", "current_revision": N}` instead of overwriting. Omit it to skip the check.
- **Errors.** `4xx`/`5xx` bodies are `{"detail": {"code": ..., "message": ..., "detail": ...}}` (FastAPI validation errors use its own shape). `code` is stable; `message` is written for the user.
- **Nothing leaves the machine.** Only `COMFYUI_URL`/`OLLAMA_URL` on loopback are contacted.

## Project, screenplay, breakdown (Phase 1)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health`, `/api/services`, `/api/diagnostics` | status, local services, hardware report |
| GET/POST | `/api/projects` | list / create |
| GET | `/api/projects/{project_id}` | whole project document |
| PUT | `/api/projects/{project_id}/script` | import or edit the screenplay (always a new version) |
| GET | `/api/projects/{project_id}/script/versions`, `/script/versions/{n}`, `/script/original` | version list, one version, immutable original |
| POST | `/api/projects/{project_id}/breakdown` | parse into scenes/characters/shots (`force` to replace hand edits) |
| PATCH | `/api/projects/{project_id}/scenes/{scene_id}` | heading, location, time, cast, notes |
| PATCH | `/api/projects/{project_id}/characters/{character_id}` | bible + Phase 3 identity fields |
| PATCH | `/api/projects/{project_id}/shots/{shot_id}` | type, duration, status, prompts, notes |
| GET/POST | `/api/projects/{project_id}/warnings`, `/warnings/{id}/ack` | validation warnings, acknowledge |
| GET/POST | `/api/projects/{project_id}/snapshots`, `/snapshots/{id}/restore` | snapshots and restore |
| GET | `/api/projects/{project_id}/manifest` | every file with size, SHA-256, read-only flag |
| GET | `/api/projects/{project_id}/shot-list.csv` | CSV export |

## Studio: characters, locations, props (Phase 3)

| Method | Path | Notes |
|---|---|---|
| GET | `/api/projects/{pid}/studio` | one payload with characters, locations, props, assets, scene links, counts |
| POST | `/api/projects/{pid}/characters` | create; duplicate names refused (`DUPLICATE_NAME`) |
| DELETE | `/api/projects/{pid}/characters/{cid}` | keeps the images, unassigns them, reports what was unlinked |
| POST | `/api/projects/{pid}/characters/{cid}/wardrobe` | costume variant (identity fields untouched) |
| PATCH/DELETE | `/api/projects/{pid}/characters/{cid}/wardrobe/{wid}` | edit / remove a variant |
| POST | `/api/projects/{pid}/locations`, `/props` | create |
| PATCH/DELETE | `/api/projects/{pid}/locations/{id}`, `/props/{id}` | edit / delete |
| POST | `.../characters/{cid}/canonical` | `{asset_id, confirm_replace, note}` → the one canonical look |
| POST | `.../characters/{cid}/wardrobe/{wid}/canonical` | canonical look for that costume |
| POST | `.../locations/{id}/canonical`, `.../props/{id}/canonical` | same rules |
| DELETE | any `.../canonical` | clear the pointer (images are kept) |

Canonical rules: the asset must be `approved` and attached to that exact subject (and wardrobe variant). Replacing an existing canonical requires `confirm_replace=true` — otherwise `409 CANONICAL_EXISTS` with the current asset id. Every change appends to `canonical_history`.

## Reference image library (Phase 3)

| Method | Path | Notes |
|---|---|---|
| POST | `/api/projects/{pid}/assets` | import; exactly one of `data_base64` or `source_path` |
| POST | `/api/projects/{pid}/assets/from-job` | copy a generation output into the library as a draft |
| GET | `/api/projects/{pid}/assets` | filters: `subject_type`, `subject_id`, `state`, `unassigned` |
| GET | `/api/projects/{pid}/assets/{aid}` | record + `references` + `file_status` |
| GET | `/api/projects/{pid}/assets/{aid}/file` | the image bytes (`?download=true` for an attachment) |
| PATCH | `/api/projects/{pid}/assets/{aid}` | role, licence, notes, re-assign, `clear_association` |
| POST | `/api/projects/{pid}/assets/{aid}/state` | `draft` / `approved` / `rejected` |
| GET | `/api/projects/{pid}/assets/{aid}/references` | what points at it, and `deletable` |
| DELETE | `/api/projects/{pid}/assets/{aid}` | needs `confirm=true`; refused while referenced |
| GET | `/api/projects/{pid}/reference-images` | `images` (files on disk) + `assets` (library) for the Images screen |

Import validation: content-sniffed type (PNG, JPEG, WEBP, GIF, BMP, TIFF), full container walk (truncated or altered files are refused), size limit `FILM_HOUSE_MAX_ASSET_MB` (default 40), pixel limits. The file is copied into `assets/<asset-id>.<ext>`, written once, made read-only and hashed. Re-importing identical bytes returns `409 DUPLICATE_ASSET` unless `allow_duplicate=true`. The user's original file is never moved or modified.

State rules: rejecting or demoting an image that is canonical is refused (`REFERENCED_AS_CANONICAL`); demoting to draft is allowed with `confirm_demote_canonical=true`, which clears the pointer and reports it in `unlinked`. Leaving `approved` also drops scene selections that used it (reported in `unlinked`).

## Scene associations (Phase 3)

`PUT /api/projects/{pid}/scenes/{scene_id}/associations`

```json
{
  "character_ids": ["..."], "location_id": "...", "prop_ids": ["..."],
  "location_reference_id": "...",
  "appearance_selections": [{"character_id": "...", "wardrobe_id": "...", "asset_id": "..."}],
  "clear_location": false, "base_revision": 12
}
```

Only `approved` assets can be selected, and each must belong to the scene's location / that character + wardrobe. The screenplay text, headings and parsed blocks are never modified by this call.

## ComfyUI workflows and generation (Phase 4)

| Method | Path | Notes |
|---|---|---|
| GET | `/api/comfyui/status` | adapter capabilities, local ComfyUI config, mappable roles |
| POST/GET | `/api/projects/{pid}/workflows` | import an API-format workflow / list |
| GET | `/api/projects/{pid}/workflows/{wid}`, `/json` | record + node inspection + suggestions; raw file |
| PUT | `/api/projects/{pid}/workflows/{wid}/mapping` | explicit node mapping + offline validation |
| POST | `/api/projects/{pid}/workflows/{wid}/validate` | live validation against the running ComfyUI |
| POST/GET | `/api/projects/{pid}/jobs` | queue a generation / list |
| GET | `/api/projects/{pid}/jobs/{jid}` | poll one job |
| POST | `/api/projects/{pid}/jobs/{jid}/cancel`, `/retry` | cancel (queue delete or interrupt), re-run |
| GET | `/api/projects/{pid}/jobs/{jid}/outputs/{index}` | a downloaded output image |

Reference images for a job come from **either** `reference_path` (a project-relative file) **or** `reference_asset_id` (a Phase 3 library asset) — sending both is a `422`. A library asset must be `approved` unless `allow_draft_reference=true`; a `rejected` asset is never usable. If the workflow has no `reference_image` node mapped the job is refused with `ROLE_UNMAPPED` and an explanation that not every workflow supports a reference.

## UI pages

`/` projects and editing · `/studio` characters, locations, props, references · `/images` ComfyUI generation · `/diagnostics` hardware and model guidance.
