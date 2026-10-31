# Model and License Selection Rules

The app is local-first, but that alone does not make every model usable for every purpose. Track the source code license and the weights/assets license separately. Before publishing or monetizing a film, check the specific model card and voice/audio asset terms, including attribution, commercial-use restrictions, prohibited-use clauses and redistribution rules.

## Initial candidates to evaluate (not installed by this starter)

- **ComfyUI** — local node-based workflow runner; suitable as a replaceable image/video execution backend. Its source repository is GPL-3.0; understand software distribution obligations if you redistribute a modified application/bundle.
- **LTX-2.5** — open-weight video/audio generation path integrated with ComfyUI and Python workflows. Current docs list demanding hardware for the full local workflow (32 GB+ NVIDIA VRAM, 32 GB RAM and 100 GB free disk). Verify the exact checkpoint and all component licenses before production.
- **Wan 2.2 TI2V-5B** — another local video candidate reported for a 24 GB VRAM class GPU with offload. Use its official repository/model card and verify hardware/weight-license terms before choosing it.
- **Kokoro-82M** — a candidate for local English TTS; the project/weights are described under permissive licenses in its repository/model card, but audit dependencies and the exact voice pack you install.
- **Piper** — local CPU TTS. Development moved from `rhasspy/piper` (MIT, archived) to `OHF-Voice/piper1-gpl`, which is **GPL-3.0**; individual voice model licenses still need independent review.
- **FFmpeg** — local media muxing, transcoding and filtering. Its main codebase is LGPL with optional GPL components, so track how the installed build is licensed if you distribute the application.
- **Whisper** — local speech transcription/alignment helper; upstream Python project lists MIT. Model files and distributions should be checked separately.

## Hardware-aware implementation

- **No discrete GPU / low VRAM:** build the entire production-management system; use simple stills, hand-authored keyframes, offline TTS, short preview renders and user-provided media until compatible image/video generation is available. Do not promise modern video generation will run smoothly on CPU.
- **8–12 GB VRAM:** start with small image models and modest resolutions; test actual workflows before making a model part of the engine.
- **16–24 GB VRAM:** expand image workflows and test carefully selected quantized/offloaded video models. Generation can still be slow and may require lower resolution/short clips.
- **32 GB+ VRAM:** evaluate current high-quality video workflows, such as LTX-2.5, against the actual model requirements.

This is a planning guide, not a guarantee that a given model will run; exact model versions, quantization, CUDA, RAM and workflow nodes matter.

The in-app Hardware & Models screen (`/diagnostics`) applies these tiers automatically using `app/model_catalog.py`. Update that file (and its `CATALOG_CHECKED` date) when model cards or licences change.
