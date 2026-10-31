# Start Here — Build Your AI Film House with Claude Code

## What we are building

A local web app that transforms your screenplay into a managed production: structured scenes, canonical character/location references, storyboards, short shots, dialogue, sound, an edit timeline and a final MP4. The aim is to avoid paid APIs and keep work on your computer.

## Do this first

1. Extract this folder somewhere easy, for example `D:\AI-Film-House\`.
2. Open that folder in a terminal.
3. Run the installation commands in `README.md`.
4. Start the local app and paste a short sample script. Confirm that project saving and shot-plan export work.
5. Run `claude` from the project root. Claude Code should read `CLAUDE.md` automatically when available; if it does not, tell it to read `CLAUDE.md`, `docs/ARCHITECTURE.md` and `docs/ROADMAP.md`.

## First development prompt for Claude Code

Copy and paste this as your first task:

> Read `CLAUDE.md`, `README.md`, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md` and the existing app before making changes. Do not rewrite the project from scratch. First report the current working features and missing pieces, and run the test suite. Then implement only Phase 1 from `docs/ROADMAP.md`: a more reliable screenplay block parser, a UI to inspect/edit scenes and character records, editable shot prompts/durations/status, and safe autosave/version snapshots. Keep the original screenplay immutable once imported and save edits as a new version. Keep everything local; do not add any hosted AI APIs or automatically download models. Add tests for parsing dialogue, parentheticals, V.O./O.S./CONT'D cues, unusual scene headings, empty input and malformed scripts. Run all tests and report exactly what passes. Do not begin video-generation integration until this phase is verified.

## Hardware survey prompt

After Phase 1, give Claude Code this task:

> Add a local Hardware & Models diagnostic screen. Show OS, Python version, available disk space for the workspace, RAM if safely detectable, GPU names/VRAM if supported, FFmpeg availability/version, ComfyUI connectivity and Ollama connectivity/local models. The screen must work when GPU-detection libraries are absent and show “unknown” rather than failing. Do not download anything. Use the results to recommend only a few compatible model candidates, and link to their official model cards and licenses.

## Image-generation integration prompt

Only after the hardware survey:

> Implement Phase 4 from `docs/ROADMAP.md`. Add a ComfyUI adapter that only connects to a localhost URL by default. Let me import an API-format workflow JSON and explicitly map its prompt, negative prompt, seed, width/height and reference image nodes. Validate all nodes before queueing; submit jobs through the ComfyUI API; poll status/history; capture output paths and model/workflow/seed metadata. Do not assume arbitrary ComfyUI workflows are interchangeable. Add timeouts, cancellation where supported, readable errors and tests with mocked local HTTP responses. Do not generate or claim output if ComfyUI is unavailable.

## Important implementation rule

Do not try to produce a full-length film as one giant text-to-video generation. Work scene-by-scene and shot-by-shot. Build canonical character references first, generate still storyboards, approve them, animate short shots, then edit and mix the approved clips. Keep a manual review step so drift can be corrected before wasting compute.
