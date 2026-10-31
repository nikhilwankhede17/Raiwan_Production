# Claude Code Instructions — AI Film House

## Product goal
Build a local-first, zero-paid-API AI film production application. The user supplies the screenplay. The engine should help transform it into a production-ready project, repeatable character/location bibles, storyboard frames, short generated shots, character dialogue audio, sound design, an editable timeline and a final rendered short film.

## Non-negotiable constraints
- Do not add paid hosted AI APIs, API keys, billing steps, telemetry or external uploads.
- Do not transmit scripts, assets, prompts or project metadata over the network except to explicitly configured localhost services.
- Prefer local model execution through replaceable adapters.
- Do not claim the application generates media until an end-to-end local generation test passes.
- Respect licenses independently for code, model weights, voice packs, music/SFX and datasets. Record the license URL and commercial-use status for every selected model/asset.
- Never overwrite original scripts or approved media without a versioned backup.
- All generated outputs must be reviewable and regenerable at shot level.
- Keep a documented CPU/fallback mode; avoid making large model downloads mandatory for basic app startup.
- Validate file paths and never allow a project to write outside its workspace.

## Coding approach
1. Read `docs/STATE.md` first — it records what is done, what is verified, what is still unproven and the agreed next actions. Then read README.md and docs/ARCHITECTURE.md and docs/ROADMAP.md before implementation.
2. Work in small phases. Before each phase, state the files to change and acceptance tests.
3. Implement one complete vertical slice at a time. Add tests for parsing, schema validation, project persistence, job lifecycle and rendering.
4. Keep provider/model operations behind adapters (LLM, image, video, TTS, music/SFX, render).
5. Store project state in portable JSON/SQLite and media in a predictable workspace directory.
6. Never make a network API call as a hidden fallback. If a local backend is unavailable, show a useful status and allow the user to continue planning.
7. Add structured logs, job progress, cancel/retry controls, explicit errors, and timeouts for long-running tasks.
8. Avoid building a complex distributed architecture. This is a single-user desktop/local web app first.
9. Do not automatically download large model files. Provide clearly labelled instructions and estimate disk space first.
10. Run tests and report what was actually verified; distinguish stubbed code from working generation.

## Product workflow
Screenplay import -> scene breakdown -> character bible -> location/prop bible -> shot list -> storyboard/reference images -> approved shot prompts -> video generation -> dialogue/TTS -> optional lip-sync -> ambience/SFX/music -> timeline assembly -> subtitles -> render -> continuity/QC report.

## Suggested architecture
- FastAPI local backend and local web UI.
- Project store: SQLite or portable JSON with versioned assets.
- Local LLM adapter: optional Ollama-compatible local endpoint; no cloud fallback.
- Image/video generation adapter: ComfyUI HTTP API, using user-imported and versioned API workflows.
- TTS adapter: selected local TTS engine with voice licenses recorded.
- Audio/video processing: FFmpeg/ffprobe, invoked safely with checked arguments and timeouts.
- Optional timeline interchange: OpenTimelineIO if needed, while retaining a portable internal timeline schema.

## Acceptance bar for an end-to-end short film
Given a user screenplay and locally installed models, the app must create all media locally, keep character/setting references attached to every shot, permit per-shot regeneration, make failures recoverable, assemble an edit from approved clips, preserve dialogue intelligibility, and produce a playable MP4 plus a project manifest and a list of model/asset licenses. Do not claim feature completion before this is tested on the user's actual hardware.
