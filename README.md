# AI Film House — Local-First Starter

A local-first foundation for building an AI film production engine. This is intentionally an MVP scaffold, not a finished AI video generator. It currently saves projects and screenplay text locally, parses common screenplay scene headings and character cues, creates an editable draft shot plan, checks local ComfyUI/Ollama availability, and exports a CSV shot list.

## Zero-budget design principle

- No paid model APIs are configured.
- No script, project or asset is uploaded by this app.
- Local models and their weights must be installed separately.
- Generated-media quality and speed depend on the machine and model license.
- “Free software” does not automatically mean every model weight or output workflow is licensed for every commercial use; review each model card and its license before public or paid distribution.

## Run on Windows

Use Python 3.11 or 3.12. Open PowerShell in this directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. API docs are at `http://127.0.0.1:8000/docs`.

If PowerShell blocks environment activation, use `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m uvicorn app.main:app --reload`.

## Run on macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Test

```bash
pytest -q
```

Local end-to-end checks that need no GPU and no ComfyUI install (they run the real HTTP server against a stdlib fake ComfyUI):

```bash
python tools/smoke_phase3.py   # studio records, reference library, approvals, Phase 4 bridge
python tools/smoke_phase4.py   # workflow import/mapping/validation, queue, poll, cancel, outputs
```

## What works now

1. Create a film project, or reopen any saved project from the header picker.
2. Paste and save your screenplay. The first save is stored as a read-only original (`screenplay/original.txt`); every later save becomes a new write-once version (`screenplay/versions/vNNNN.txt`) with a SHA-256 hash. Earlier versions can be viewed or loaded back into the editor.
3. Block parser: scene headings (INT./EXT./INT./EXT./I/E/EST., lowercase, numbered, Fountain `.forced` and `#tag#`), action, character cues with V.O./O.S./O.C./CONT'D and other extensions, parentheticals (inline, multi-line), dialogue, transitions, dual dialogue, Fountain title page/notes/sections/boneyard, and PDF page artefacts such as `(MORE)` and `CONTINUED:`.
4. Uncertain parses and data problems are listed as warnings with script line numbers. You fix them in the editor or acknowledge them; warnings never block saving.
5. Editing tabs for scenes (heading, location, time, INT/EXT, cast, notes), characters (name, description, visual signature, voice direction, notes; renames propagate) and shots (type, duration, status, cast, visual/motion prompts, audio notes, notes). Edits autosave after a short pause. If another tab changed the project in the meantime, the edit is refused (HTTP 409) instead of overwriting.
6. Snapshots of the production data are taken automatically while you edit (at most every 120 s, keeping the last 30), before every rebuild, script change or restore, and on request. Any snapshot can be restored; the current state is snapshotted first.
7. `manifests/assets.json` lists every project file with its size, SHA-256 and read-only flag.
8. Generate a draft coverage plan, export shots to CSV, and check whether local ComfyUI (default `127.0.0.1:8188`) and Ollama (default `127.0.0.1:11434`) are running.

The parser and planner are heuristic. Screenplays with nonstandard formatting need manual correction (use the Warnings tab). A generated plan is a draft, not a director-approved storyboard. No media is generated yet.

Rebuilding the breakdown replaces scene and shot records. If any were edited by hand, the app asks first and takes a snapshot. Character bible fields are carried over by cue name.

Set `FILM_HOUSE_AUTOSNAPSHOT_SECONDS` to change the autosave snapshot interval.

## Hardware & Models diagnostic

Open `http://127.0.0.1:8000/diagnostics` (or the "Hardware & Models" button). It shows OS, Python, CPU, RAM, free disk for the workspace, GPUs/VRAM, FFmpeg/ffprobe version and build licence, ComfyUI status/devices/model files and Ollama status/local models. JSON is at `/api/diagnostics` (cached 30 s; `?refresh=true` re-runs).

- Every probe is optional. Missing `nvidia-smi`, `psutil`, `rocm-smi`, registry access, FFmpeg or services show "unknown"/"not found" instead of failing.
- GPU sources, best first: `nvidia-smi`, ComfyUI `/system_stats`, then the OS (Windows registry, `rocm-smi`/`lspci`, `system_profiler`). Integrated graphics are not counted as usable for generation.
- Recommendations pick at most one candidate per category from a short curated list (`app/model_catalog.py`), with links to the official model card and licence. They are planning guidance; nothing is downloaded or installed.
- Only localhost `COMFYUI_URL`/`OLLAMA_URL` values are contacted; any other value is ignored.

## Character, location and prop studio

Open `http://127.0.0.1:8000/studio` (or the "Studio" button). Everything is stored in your project folder; no image ever leaves this computer.

**Records.** Characters hold identity (age range, physical description, facial traits, hairstyle, body type, personality, voice direction) plus wardrobe variants for costume changes — adding a variant never alters the identity fields. Locations and props hold their own description, continuity requirements, geography notes and scene links. Each record has a stable id and can be created, edited and deleted; deleting one keeps its images and simply unassigns them.

**Reference library.** Import images with the file picker (or an absolute path on this machine). Each file is checked by content, not by name: PNG, JPEG, WebP, GIF, BMP and TIFF are accepted, and the container is walked end to end so truncated or altered files are refused with a readable reason. Accepted images are copied into `assets/<asset-id>.<ext>`, written once, made read-only and recorded with their SHA-256, pixel size and format. Your original file is never moved or changed. Re-importing identical bytes is refused as a duplicate unless you ask for a second copy. Set `FILM_HOUSE_MAX_ASSET_MB` (default 40) to change the size limit.

**Approval lifecycle.** Every image starts as a **draft** and can be **approved** or **rejected**, with timestamps and a note recorded in its history. One approved image per character (or wardrobe variant, location, prop) can be marked **canonical**:

- only an approved image attached to that exact subject can be canonical;
- replacing an existing canonical reference requires an explicit confirmation, and the previous image is kept (the change is recorded in `canonical_history`);
- an image that is canonical cannot be rejected, and moving it back to draft needs a second confirmation that clears the pointer;
- versions count up per subject + wardrobe + role and are never reused.

**Scenes.** A scene can be linked to characters, a location and props, and can select the approved look to use (including which wardrobe variant). This is production metadata only: the screenplay, scene headings and parsed blocks are never modified.

**Deletion.** An image that is canonical or selected by a scene cannot be deleted until it is unlinked; the API lists exactly what points at it. A confirmed delete snapshots the project first, removes the record, then deletes the stored file — the image file itself is gone afterwards.

**Generated frames.** Any output of a ComfyUI job can be copied into the library as a draft, then approved and made canonical, so generated references flow through the same checks as imported ones.

## Local image generation (ComfyUI)

Open `http://127.0.0.1:8000/images` (or the "Images" button). Generation runs on **your** ComfyUI on this computer; this app never downloads models or contacts a hosted API.

1. In ComfyUI, export your workflow with **Workflow → Export (API)** (older builds: "Save (API Format)"). Editor-format files are rejected with that instruction.
2. Import the file. It is stored write-once as `workflows/<id>.json` and pinned by SHA-256. Workflows are never treated as interchangeable: each import gets its own mapping and validation.
3. Map each field — prompt, negative prompt, seed, width, height, reference image — to an explicit node id and input. Suggestions are offered (the positive/negative prompt nodes are found by tracing the sampler's links) but are never applied for you. Only typed-in inputs can be driven; a wired input is refused with an explanation.
4. Validate. The offline check verifies nodes, input names, types and that a Save node exists. "Validate against ComfyUI" additionally confirms the node classes and inputs exist in your build and warns when a named model file is not installed. Changing the mapping clears the live result.
5. Generate. Jobs run one at a time, are polled through ComfyUI's queue/history, and can be cancelled (removed from the queue, or interrupted while running).

Each job is saved as `jobs/<job-id>.json` with its own log, outside `project.json`, so generating never conflicts with editing. Outputs are downloaded into `references/<job-id>/` (or `storyboards/<job-id>/` for a shot) next to a `metadata.json` recording the prompt, seed, workflow SHA-256, mapping, model filenames and ComfyUI version.

- If ComfyUI is unavailable, nothing is queued and no output is claimed: the request fails with a readable message.
- A job only reports images that were actually downloaded and hashed. "Finished with no images", a failed download, a lost prompt and a timeout are all reported as failures.
- Reference images come from this project: previous outputs, files you put in `references/input/`, or an **approved** asset from the Studio library (`reference_asset_id`). They are uploaded to ComfyUI's input folder when a job runs.
- A job names exactly one reference source. A draft asset needs `allow_draft_reference`, a rejected one is never used, and a workflow with no reference node mapped is refused with an explanation instead of a silent substitution.
- ComfyUI has no per-prompt progress over HTTP, so the UI shows queue position and state rather than a percentage.
- Timeouts: `FILM_HOUSE_JOB_TIMEOUT_SECONDS` (default 900) for a whole job, `FILM_HOUSE_JOB_POLL_SECONDS` (default 1.5) between polls.

**Not yet proven on real hardware.** The adapter is verified against mocked HTTP in the test suite and against a local fake ComfyUI (`python tools/smoke_phase4.py`, which drives the real HTTP paths including upload, polling, cancel and the ComfyUI-down case). It has not been run against a real ComfyUI with real model weights on this machine — that is the remaining Phase 4 proof and needs a suitable GPU.

## Next milestones

Follow `docs/ROADMAP.md`. The HTTP API is listed in `docs/API.md`. Keep generation backends replaceable, save every prompt/seed/model/version, and require human approval at important creative checkpoints.
