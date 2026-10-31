# Build Roadmap

## Phase 0 — Working foundation (included)
- Local project creation and screenplay saving.
- Heuristic scene/character parsing.
- Draft shot plan and CSV export.
- Local service health checks.

**Exit criteria:** app starts, persists a project, parses a sample script, exports a shot list and passes unit tests.

## Phase 1 — Reliable production data (implemented; automated tests pass, browser UI awaiting manual check)
- Add screenplay block parser for action, dialogue, parentheticals, transitions and continued dialogue. (`app/screenplay.py`)
- Add scene/character/shot editing UI and autosave. (`app/static/index.html`, `PATCH /api/projects/{id}/scenes|characters|shots/{id}`)
- Add validation warnings and explicit manual correction. (`app/validation.py`, Warnings tab, acknowledge/re-open)
- Add project version snapshots and asset manifest. (`app/storage.py`: immutable original, write-once script versions, snapshots, `manifests/assets.json`)
- Add unit tests for unusual headings, cue suffixes and malformed scripts. (`tests/test_parser_blocks.py`, `tests/test_api.py`)

**Exit criteria:** all tests pass; manually confirm in a browser that you can open a project, edit records in each tab, see "All changes saved", trigger a conflict from a second tab, and restore a snapshot.

## Phase 2 — Optional local LLM director assistant
- Add Ollama-compatible client behind a `LocalLLMAdapter`.
- Use strict JSON schema output to draft logline, story bible, character descriptions, scene summaries, continuity notes, shot types and prompts.
- Validate every response with Pydantic; retry a limited number of times and show parse errors.
- Never change the original script; compare generated changes as proposals only.
- Allow the app to work fully without Ollama.

## Phase 3 — Character and location references (implemented)
- Add reference image upload and asset management. (`app/assets.py` validation, `app/studio.py` logic, write-once `assets/<asset-id>.<ext>` storage, `/studio` UI.)
- Build character sheets: front, three-quarter, profile, full body, wardrobe/costume variations and expressions. (Asset `role` plus `WardrobeVariant`; identity fields are never touched by adding a costume variant.)
- Create location wide shots and prop turnarounds. (`Location`/`Prop` records with their own references, continuity requirements and geography notes.)
- Add a manual approval checkpoint before shots can use a reference as canonical. (draft → approved → rejected; only an approved asset can be canonical or chosen by a scene; replacing a canonical reference needs an explicit confirmation and keeps the old image.)
- Store reference path, source, license and version/hash. (`ReferenceAsset`: path, SHA-256, bytes, pixel size, format, source/`job_id`, licence, per-subject version, state history.)

**Verified:** 92 automated tests (`tests/test_assets.py`, `tests/test_studio.py`) plus `python tools/smoke_phase3.py` — a real HTTP run covering import by upload and by absolute path, duplicate and corrupt refusals, path-traversal refusal, approval and canonical replacement, wardrobe separation, scene selection, generating with an approved asset, importing a generated frame, and safe deletion.
**Limitation:** the generation step of that smoke test talks to a fake local ComfyUI. Using a reference in a real workflow still needs ComfyUI installed with your own weights (see Phase 4).

## Phase 4 — Local image generation (implemented; proof on real weights still pending)
- Support ComfyUI through a local-only adapter. (`app/comfy.py`: `ComfyClient` refuses any non-loopback URL, every request has a timeout, failures become readable `ComfyError`s; `ComfyAdapter` implements capabilities/validate_config/submit/status/cancel/collect_outputs.)
- Let user import an API-format workflow and map named workflow nodes to prompt, negative prompt, seed, dimensions and references. (`app/workflows.py`, `/images` UI; editor-format files rejected with instructions; suggestions offered, never auto-applied.)
- Validate workflow inputs before submitting to `POST /prompt`; monitor job status/history; capture output files and generation metadata. (Offline + live validation; serial `JobManager` polls queue/history, downloads and hashes outputs, writes `metadata.json`.)
- Do not bundle a checkpoint or auto-download multi-gigabyte weights. (Nothing is downloaded; missing model files are reported as warnings.)
- First proof: generate one character reference frame and one location frame, locally, from the UI.

**Verified:** 91 automated tests with mocked local HTTP, plus `python tools/smoke_phase4.py` (a stdlib fake ComfyUI on 127.0.0.1 driving import → map → live validate → upload reference → queue → poll → download → cancel → ComfyUI-down refusal).
**Not verified:** a real ComfyUI with real checkpoints. Until that runs on your hardware, do not treat image generation as proven.

## Phase 5 — Local video generation
- Add a video adapter using a ComfyUI workflow verified on the user's computer.
- Input: approved still/reference, motion prompt, duration/frame count, seed, aspect ratio and model preset.
- Output: short clip plus seed/model/workflow metadata.
- Queue jobs serially by default to avoid VRAM exhaustion; add cancel/retry and OOM messages.
- First proof: produce 3–5 short shots from approved stills and assemble a test sequence.

## Phase 6 — Dialogue, lip-sync and sound
- Add a local TTS adapter after selecting a model and voice pack with terms appropriate to the intended distribution.
- Generate each dialogue line as a separately named WAV; retain exact text and timing metadata.
- Add optional alignment/lip-sync as a separate adapter.
- Add SFX and music tracks from user-owned, generated or properly licensed assets; do not assume a model's weights or samples are royalty-free.
- Implement basic gain, fades and dialogue ducking.

## Phase 7 — Timeline and renderer
- Create an editable timeline UI or a simpler shot-order editor first.
- Render by invoking FFmpeg with argument arrays (not shell-interpolated strings), checked paths and timeouts.
- Support 1080p/24 fps final output only if source clips/hardware can sustain it; allow draft render profiles.
- Add captions/subtitles, title cards, audio mixing, end credits and output report.

## Phase 8 — Continuity and production quality
- Compare reference/shot outputs with optional vision embeddings and scene-specific checks.
- Flag character/costume drift, mismatched props, geography/screen direction, duplicate/frozen shots and dialogue/audio gaps.
- Require human approval for meaningful creative choices; automated QC is advisory, not a perfect judge.
- Add final manifest, model license report and deterministic rerender for unchanged inputs.

## First complete short-film target
Aim for a 30–90 second proof film first: 5–12 shots, one location, one or two characters, limited visible lip-sync and a small number of actions. Then target 3–5 minutes after the entire pipeline is stable. A 10-minute film with 4-second generated shots may need roughly 150 shots before alternate takes and cutaways; this makes workflow reliability and consistency more important than a single model demo.
