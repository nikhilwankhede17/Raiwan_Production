# Project state — read this first

A handoff note for whoever (person or agent) picks this project up next. Keep it updated when a
phase lands. For the plan see `docs/ROADMAP.md`; for the HTTP surface see `docs/API.md`; for the
non-negotiable rules see `CLAUDE.md`.

## Status by phase

| Phase | What it is | State |
|---|---|---|
| 0 | Project creation, screenplay saving, draft shot plan, CSV export | done |
| 1 | Reliable parser, editing UI, autosave, versions, snapshots, warnings | done |
| — | Hardware & Models diagnostic (`/diagnostics`) | done |
| 2 | Optional local LLM assistant (Ollama) | **not started** |
| 3 | Character / location / prop studio + reference library (`/studio`) | done |
| 4 | Local image generation via ComfyUI (`/images`) | code done, **not proven on real hardware** |
| 5 | Local video generation | **not started** |
| 6 | Dialogue (TTS), lip-sync, sound | **not started** |
| 7 | Timeline and FFmpeg render | **not started** |
| 8 | Continuity and QC | **not started** |

## What is actually verified

- **322 automated tests pass** (`python -m pytest -q`), run three times consecutively with no flakes.
  Breakdown: parser 81, studio 44, assets 48, comfy 56, generation 35, hardware 26, api 29, legacy 3.
- **`python tools/smoke_phase3.py`** — 32 checks over real HTTP: import by upload and by absolute path,
  duplicate/corrupt/traversal refusals, approval and canonical replacement, wardrobe separation,
  scene selection, generating with an approved asset, importing a generated frame, safe deletion.
- **`python tools/smoke_phase4.py`** — 23 checks over real HTTP: workflow import, mapping, live
  validation, reference upload, queue, poll, download, cancel, and the ComfyUI-unavailable path.

## What is NOT verified — do not claim otherwise

Image generation has **never run against a real ComfyUI with real model weights**. Both smoke tests
drive `tools/fake_comfyui.py`, a stdlib stand-in that speaks the same HTTP API. Until a real run
produces a real image, treat Phase 4 as unproven. The browser UIs (`/`, `/studio`, `/images`,
`/diagnostics`) pass a JavaScript syntax check and every control calls a tested endpoint, but they
have not been clicked through in a browser.

## Hardware reality

The machine this was built on has integrated graphics only and cannot generate anything. The
intended target is a laptop with an **NVIDIA RTX 3050, 6 GB VRAM** (CUDA available).

What 6 GB can do, from published community reports (verify before relying on it):

- **Stills: yes.** SDXL at 768px, or SD 1.5 at 512px comfortably. Keep workflows simple — stacking
  ControlNet + face detailer + hires fix at once will exceed 6 GB.
- **Video: marginal.** Short (1–3 s), ~480p clips using GGUF-quantised models, several minutes each.
  Needs 24–32 GB of *system* RAM for offloading. The official Wan2.2 TI2V-5B path wants 24 GB VRAM
  and is out of reach.

Also needed on the target machine: Python 3.11 or 3.12, ComfyUI, one checkpoint (~7 GB), and FFmpeg
on PATH for later phases. Check free space before downloading models.

## Next actions, in order

1. **Fix the model catalogue for 6 GB cards.** `app/model_catalog.py` sets SDXL's minimum at 8 GB
   VRAM, so a 6 GB GPU is wrongly reported as incompatible. Add a 6 GB tier (SD 1.5, and SDXL at
   768px with low-VRAM flags). There is no test pinning the 8 GB figure, so this is a safe edit —
   but add one.
2. **Run `/diagnostics` on the target laptop.** First real test of GPU detection against an actual
   NVIDIA card; also records system RAM and free disk, which decide whether video is feasible.
3. **Close Phase 4 for real.** Install ComfyUI + one SDXL checkpoint, export an API-format workflow,
   map it in `/images`, and generate one character frame and one location frame from the UI. Then
   update `docs/ROADMAP.md` and `README.md` to say image generation is proven, and delete the
   "not proven" warnings.
4. **Then choose:** Phase 2 (LLM assistant — runs on CPU, works today) or Phase 5 (video — target
   short quantised clips, not the 720p path).

## Invariants a new session must not break

- No paid or hosted AI APIs, no telemetry, no uploads. Only loopback `COMFYUI_URL` / `OLLAMA_URL`
  are ever contacted; `ComfyClient` refuses a non-loopback URL at construction.
- Never overwrite the original screenplay or an imported asset. Script versions, snapshots,
  workflows and reference images are write-once and made read-only.
- Writes to `project.json` are revision-checked: a stale `base_revision` returns HTTP 409 rather
  than clobbering. Generation jobs live in `jobs/<id>.json`, outside `project.json`, so generating
  never conflicts with editing.
- Only an **approved** reference asset can become canonical or be used by a scene; replacing a
  canonical reference needs explicit confirmation and keeps the old image.
- A job reports only outputs that were actually downloaded and hashed. If ComfyUI is unavailable,
  nothing is queued and nothing is claimed.
- Do not auto-download model weights. Do not add dependencies without need — `requirements.txt` is
  four lines, and Pillow/psutil are optional enhancements with stdlib fallbacks.
- Phase order matters: do not start video generation before Phase 4 is proven on real hardware.

## Where things live

```
app/screenplay.py     screenplay block parser
app/models.py         every Pydantic model (project, studio, workflow, job)
app/storage.py        atomic + write-once persistence, snapshots, manifest
app/studio.py         Phase 3 rules: subjects, approval, canonical, safe deletion
app/assets.py         image validation (stdlib sniffers, optional Pillow)
app/comfy.py          localhost-only ComfyUI client and adapter
app/workflows.py      API-workflow parsing, mapping, validation
app/jobs.py           serial generation-job manager
app/validation.py     project warnings
app/hardware.py       hardware/model diagnostics
app/model_catalog.py  curated model candidates + licences
app/static/*.html     the four UI pages
tools/                fake ComfyUI + the two smoke tests
workspace/projects/   per-project data (gitignored — never committed)
```
