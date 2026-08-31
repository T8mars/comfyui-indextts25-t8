# comfyui-indextts25-T8

[简体中文](README.md) | **English**

Native ComfyUI V3 nodes for IndexTTS 2.5. The nodes are available under:

`T8star-Aix / Audio / IndexTTS 2.5`

Creator: **Bilibili: T8star-Aix**.

## Creator and resources

- Bilibili: [T8star-Aix](https://space.bilibili.com/385085361)
- YouTube: [T8star-Aix](https://www.youtube.com/@T8star-Aix/)
- Hugging Face: [t8star](https://huggingface.co/t8star)
- Complete IndexTTS 2.5 bundle: [t8star/IndexTTS-2.5-Comfy](https://huggingface.co/t8star/IndexTTS-2.5-Comfy)
- API sign-up (affiliate link): [api.seedance.nz](https://api.seedance.nz/sign-up?aff=5f4w)
- Online AI apps (RunningHub): [T8star-Aix profile](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- ComfyUI portable package: [Quark Drive](https://pan.quark.cn/s/264edb7e36bd)
- IndexTTS 2.5 model mirror: [Quark Drive](https://pan.quark.cn/s/c9c267081fbf)

This repository is locked to the IndexTTS 2.5 inference core and the official 2.5 model manifest. It will not fall back to or accidentally load IndexTTS 2.0.

Current baseline: **ComfyUI Node 0.22.0 · Desktop 0.23.0 · Core `ee40fa7d` · Upstream Model `c39ce5ba`**.

### v0.22.0 Chinese number/date normalization verification

- Sampling settings identify number/date normalization explicitly, while the environment node performs a real `1939年 → 一九三九年` smoke test instead of trusting package presence alone.
- Windows/macOS can optionally use `wetext`; Linux can optionally use `WeTextProcessing`. Missing or failed backends preserve the source text and do not block generation.
- Adds a separate optional requirements file and bilingual installation guidance so platform wheel availability cannot break the normal node installation.
- Regenerates and validates all 33 UI and 33 API example workflows.

### v0.21.4 PyTorch/Triton acceleration fallback

- Detects PyTorch/Triton combinations that incorrectly pass the AMD-only `waves_per_eu` option on NVIDIA and reports an actionable compatibility diagnostic.
- When optional GPT/torch.compile acceleration fails at runtime, the node releases the accelerated model, reloads normal mode, and retries instead of failing the whole workflow.
- Regenerates and validates all 33 UI example workflows; Desktop 0.22.2 applies the same fix to streaming playback and uses a conservative default profile.

### v0.21.3 Linux TorchCodec release compatibility

- Linux current CI now installs the TorchCodec version matching current Torch/Torchaudio; the pinned 2.9 matrix keeps TorchCodec 0.9.
- Both Linux TorchCodec jobs install FFmpeg shared development libraries and perform a real WAV encode/decode roundtrip instead of hiding native-loader failures behind a skipped test.
- PCM normalization tests now validate the public audio-write boundary rather than the legacy Torchaudio 2.8 `torchaudio.save` implementation. Inference behavior is unchanged from 0.21.2.

### v0.21.2 Arabic, Cache API, and dual-VRAM quality gates

- OpenAI Whisper stays pinned to `20250625`. ZH/EN/JA/ES use `base`; AR uses `small`, which reduced WER on the same WAV from `0.6154` to `0.1923`. Arabic comparison removes diacritics/tatweel and normalizes common Alef/Ya forms without hiding consonant errors.
- GPT inference now exposes the Transformers `Cache` / `DynamicCache` API while accepting legacy tuple input. Real five-language generation no longer emits the legacy `past_key_values` deprecation warning.
- Torchaudio 2.9+ startup preflight checks TorchCodec and FFmpeg shared libraries before model loading. Windows systems with only `ffmpeg.exe` receive an actionable DLL diagnostic. The bundled Torchaudio 2.8 path is unaffected.
- The weekly/manual GPU workflow serializes formal 8 GB and 24 GB profiles. Both passed on RTX hardware with peaks near `3.34 GiB` and `5.52 GiB`. Path-free baselines are `quality_baselines/openai-whisper-mixed-8gb-gpu.json` and `quality_baselines/openai-whisper-mixed-24gb-gpu.json`.
- A trend job publishes JSON, Markdown, and SVG history for mean CER/WER, median RTF, and peak VRAM. Full WAV/report artifacts are retained for 30 days and trends for 90 days.

### v0.21.1 pinned ASR baseline and dependency migrations

- OpenAI Whisper is pinned to `20250625`; the formal GPU baseline uses `base`, CUDA, one reference voice, and one seed across ZH/EN/JA/ES/AR CER/WER measurements.
- A separate weekly/manual GPU workflow runs only on a self-hosted Windows runner labeled `gpu` and `indextts25`, so ordinary PR/push CI never loads the roughly 10 GB model.
- IndexTTS GPT inference classes now inherit `GenerationMixin` explicitly. Torchaudio 2.9+ uses native TorchCodec `AudioDecoder/AudioEncoder`, while the bundled 2.8 runtime retains its stable backend.
- Normal CI adds a Linux torch/torchaudio 2.9 + TorchCodec 0.9 compatibility job. All 33 UI and 33 API workflows were regenerated after the version bump.
- On Windows, Torchaudio 2.9 / TorchCodec also needs discoverable FFmpeg shared libraries (DLLs); a standalone `ffmpeg.exe` is not sufficient to load TorchCodec. The official portable bundle remains pinned to Torchaudio 2.8 and is unaffected.
- The path-free baseline is `quality_baselines/openai-whisper-base-gpu.json`; complete WAVs and reports are retained as 30-day Actions artifacts.

### Five-language quality regression and CLI

- `scripts/run_multilingual_quality_regression.py` synthesizes fixed long-form ZH/EN/JA/ES/AR cases with one reference voice and writes every WAV plus `quality-report.json`.
- The report includes optional CER/WER, internal-segment rate variation, clipping, silence, duration, RTF, and peak VRAM. `--baseline` performs explainable regression checks against an earlier report.
- `indextts.cli` now uses the official IndexTTS 2.5 multilingual inference class and exposes reference/vector/text emotion, duration, sampling, CFM, precision, reference-device, and optional acceleration controls.
- The Desktop development branch also adds a visual internal-segment rate chart, separate original/automatic-retry/current previews, and segment-only regeneration followed by safe reassembly.

```powershell
# Run from this node repository. It does not download IndexTTS/reference audio;
# optional ASR may download Whisper into the output directory.
..\.venv\Scripts\python.exe scripts\run_multilingual_quality_regression.py `
  --model-dir D:\ComfyUI\models\TTS\IndexTTS-2.5 `
  --voice D:\ComfyUI\input\voice_reference.wav `
  --asr-backend auto --output-dir .\quality-regression --strict

..\.venv\Scripts\python.exe -m indextts.cli "An IndexTTS 2.5 CLI sample." `
  --voice D:\ComfyUI\input\voice_reference.wav `
  --model-dir D:\ComfyUI\models\TTS\IndexTTS-2.5 --language EN `
  --output-path .\cli-output.wav
```

### v0.21.0 cross-segment speech-rate anomaly guard

- Every real internal long-text segment records speech units, actual audio duration, and units per second. A median baseline is created only after at least two stable segments.
- A later segment is suspicious only when its rate collapses below 45% of that baseline by a meaningful margin. Short lines, ordinary emotional slowing, deterministic sampling, and native target-duration synthesis are not forcibly accelerated.
- Only the suspicious segment is regenerated with a smaller token limit and independent seed. The retry replaces the original only when it is materially closer to the baseline without becoming too fast.
- Status output includes `segment_rate_guard` with the baseline, ratio, retry decision, and accepted candidate for diagnosing slow long-text tails.
- Version 0.21.0 passed Comfy Registry security scanning and is Active; its Publish workflow is fully green after an idempotent rerun.

### v0.20.9 Registry scanner compatibility and stability fixes

- Linux and Windows release gates now run a Pylint/Astroid fatal-error check against `services/model_store.py`, catching any source-reproducible `F0002/astroid-error` before publication.
- Hugging Face model downloads now use chunked streaming in the current process instead of launching a download worker process, while preserving progress, cancellation, resume, pinned revisions, size checks, and SHA-256 verification.
- Fixed GPT acceleration tensors falling back to the default `cuda:0` device when the loader selects `cuda:1` or a higher-numbered GPU.
- Fixed delayed cancellation and orphaned subprocesses across audio.cpp, Hugging Face model downloads, segmented generation, and ASR candidate review.
- Fixed non-finite output from voice post-processing presets on 4/8 kHz audio.
- Dialogue lines without an explicit language now inherit the matching Voice Profile language; per-line string `false`, invalid actual subtitle times, and example 33 role mapping are also corrected.
- Added concurrency protection to the model hash cache and made the update check enforce its 1 MiB limit while streaming.
- Removed unused training weights from Registry packages and bundled the complete upstream BigVGAN third-party license texts.
- Stop checks now reuse ComfyUI's already-loaded runtime module instead of initializing CUDA from CPU-only environments.

### v0.20.5 remove the networked audio.cpp installer

- Removed the audio.cpp Component Installer node and all in-node GitHub/Hugging Face runtime and GGUF download code so the optional network installer is no longer flagged by Comfy Registry.
- Kept the isolated audio.cpp Experimental Generation node. Users must download components from the official pages and enter local absolute paths; the node never installs, updates, or modifies audio.cpp over the network.
- Removed former example 34. The collection now contains 33 UI workflows and 33 API prompts; example 27 continues to demonstrate the manually installed local audio.cpp backend.

### v0.20.4 model integrity, concurrent release, and portable asset hardening

- Full SHA-256 validation and model-cache fingerprints now include a file change marker that ordinary timestamp restoration cannot overwrite, detecting same-size replacements.
- “Release all models” defers active entries instead of allowing a second model copy to load; ASR cache clearing also waits for the current transcription to finish.
- Model downloads stop before network transfer when missing files plus the 1 GiB reserve exceed available space, reporting both required and available capacity.
- The audio.cpp component installer adds resume integrity checks, extraction-space preflight, transactional replacement, manifest path containment, and Windows file-change detection.
- Saved voice bundles reject Windows case/Unicode path aliases and duplicate roles, while refreshes no longer create an unbounded sequence of cache directories.

### v0.20.2 ComfyUI V3 test fix

- Node schema tests now address inputs and outputs by stable node ID instead of list positions that change when nodes are added.

### v0.20.1 cross-platform test fix

- Platform detection in the audio.cpp installer is now local and safely mockable, preventing Windows simulation from contaminating Linux test processes. User-facing behavior is unchanged from v0.20.0.

### v0.20.0 shared voice library and one-click optional audio.cpp components (installer removed in v0.20.5)

- New Saved Voice node reads `.t8voice.zip` bundles exported by Desktop. Each bundle carries voice audio, role name, language, role emotion, tags, favorite state, notes, and quality metadata, so workflows no longer need duplicate reference uploads.
- Put bundles under `ComfyUI/models/TTS/IndexTTS-2.5/voices/`. Refresh the browser after adding or replacing a bundle; increment the node's refresh token when only file contents changed.
- New audio.cpp Component Installer downloads official Windows CUDA/Vulkan/CPU runtimes and Q8/F16/original GGUF weights only after explicit confirmation, with resume support, disk preflight, and SHA-256 verification. The default Python path is untouched.
- audio.cpp Generation can leave both paths blank and automatically consume the verified installed component; explicit absolute paths still take precedence.
- The collection now contains 34 UI workflows and 34 API prompts, adding `33_saved_voice_library.json` and `34_audiocpp_one_click.json`.

### v0.19.0 model-download progress and disk preflight

- Model Loader now maps scanning, transfer, and SHA-256 verification to ComfyUI task progress when opt-in download/repair is enabled, instead of exposing progress only in the terminal.
- Files are fetched individually from the pinned manifest. ComfyUI shows overall progress, while the terminal reports the active phase, filename, speed, and ETA. Hugging Face resume behavior remains intact, while damaged files are forcibly refreshed.
- A conservative missing-byte estimate and available target-disk space are reported before large writes; critically low free space fails with an actionable error.
- The complete set of 26 main and auxiliary files is verified again after repair, and failures retain the exact missing or mismatched path.

### v0.18.0 long-text/SRT stability and reference-cache management

- Long English, Spanish, and SRT lines are checked for `max_mel_tokens` warnings and implausible output duration. Only a suspicious speech block is retried once with a smaller segment budget, and the generation report records the decision.
- Windows GBK consoles no longer abort generation when Spanish text contains non-ASCII characters. Real IndexTTS 2.5 long-English and long-Spanish WAV regressions pass.
- The model/memory control node adds `reference_cache_status` and `clear_reference_cache`. Reports include entries, bytes, hits, misses, and hit rate; clearing removes only this node's `safetensors` cache, never model files or source audio.
- Example 06 demonstrates the long-English guard, example 13 includes a long Spanish SRT line, and example 26 demonstrates cache status.

### v0.17.0 context-aware per-line emotion suggestions

- A new Context Emotion Suggestion node uses local QwenEmotion while separating previous lines, the target line, following lines, and speaker roles.
- Existing manual `text:` / `vector:` overrides are preserved unless `overwrite existing` is explicitly enabled.
- The node returns a suggested dialogue script, editable JSON, and a summary. It **does not generate audio**; review it before connecting Timeline Editor or Multi-role Generation.
- New `32_context_emotion_suggestions.json` is intentionally a two-stage example with no generation node connected.

### v0.16.3 Registry security-scan compatibility

- The audio.cpp experimental node remains available and now uses an awaitable ComfyUI V3 asynchronous, shell-free fixed-argument process interface while preserving absolute-path, backend, and argument validation.
- Manual update checks are restricted to three built-in official endpoints, reject cross-site redirects, and cap responses at 1 MiB. The node version is read through the GitHub Contents API.
- Preserved the complete bundled upstream language mapping while rewriting one language-code literal that the scanner mistook for a privilege-escalation command; every language index and behavior remains unchanged.
- `.comfyignore` excludes tests and CI files from the Registry artifact while the complete source, test suite, and examples remain on GitHub.

### v0.16.2 renamed model repository

- The complete Hugging Face model repository is now `t8star/IndexTTS-2.5-Comfy`; the downloader, manifest, and documentation all use the new address.
- The repository name does not change the standard ComfyUI location. Keep the model files directly under `ComfyUI/models/TTS/IndexTTS-2.5/`.

### v0.16.1 model-directory guidance

- The English and Chinese GitHub READMEs and Hugging Face model card now show the separate node/model directory tree, complete Windows paths, and a concrete extra-nesting mistake to avoid.
- Keep the complete model at `ComfyUI/models/TTS/IndexTTS-2.5/` and the node code at `ComfyUI/custom_nodes/comfyui-indextts25-T8/`.

### v0.16.0 complete model bundle and opt-in repair

- Added `t8star/IndexTTS-2.5-Comfy`, a single complete repository containing the unmodified official 2.5 main model, the required `bpe.model` from the official 2.0 model repository, and the Wav2Vec2-BERT, CAMPPlus, and BigVGAN runtime assets.
- Model Loader now offers “download/repair the complete model when missing.” It is off by default and requires explicit license acceptance before downloading approximately 7.7 GiB.
- The downloader pins revisions, verifies the SHA-256 manifest, and repairs only missing or damaged files. A manually downloaded complete repository works directly as well.
- The official `IndexTeam/IndexTTS-2.5` repository currently has no `bpe.model`, so that snapshot alone is correctly reported as incomplete; the user's destination directory is not the problem.

### v0.15.0 per-line emotion and tail-safe subtitle slots

- The same role can override text emotion or an eight-dimensional vector on every batch, JSON, SRT, or timeline line; omitted values inherit the Voice Profile default.
- New `31_per_line_emotion.json` demonstrates one voice switching between calm, angry, and inherited emotion line by line.
- Subtitle-slot defaults now use `pad`, preserving overlong tails; `native/exact` explicitly warn that they can trim.
- Desktop 0.16.0 streams through bundled PyAV and no longer requires system FFmpeg/FFprobe.
- `duration_factor` is now consistently described as official acoustic-duration adaptation rather than natural prosodic speaking rate.

Version 0.14.0 adds an active-mode runtime benchmark, a manual upstream update check, safe conditioning reuse
across model reloads, and retained multi-candidate quality selection. Desktop 0.16.0 can benchmark acceleration
modes sequentially and recommend one; every expensive or network operation remains explicitly user-triggered.
The complete model repository preserves upstream source and revision attribution for every component. Main weights remain the official 2.5 files; `bpe.model` is the unmodified file from the official 2.0 model repository.
Desktop and Node are separate deliverables with independent versions; Core/Model identify the pinned official code and weight revisions.

### v0.14.0 runtime benchmark, candidates, and cache update

- Runtime Benchmark uses the same reference, text, and seed, then reports median/best RTF and peak VRAM for the mode that actually initialized. Change the Model Loader mode and rerun to compare fairly.
- Speech Generation retains every requested candidate and exposes them as an AUDIO list. With local ASR it combines transcript similarity and waveform quality; without ASR it still selects by clipping, silence, DC offset, and related technical checks.
- Model Loader enables a persistent conditioning cache by default. Content, model revision/fingerprint, precision, and reference device isolate entries; files use `safetensors`, are capped at 128 entries, and can be disabled in advanced settings.
- Update Check manually compares the official code, Hugging Face model, and node version. It reports only and never downloads or overwrites anything.
- Desktop 0.15.0 sequentially benchmarks acceleration modes before startup and can apply its recommendation; it never starts the service automatically.

### v0.13.0 acceleration diagnostics update

- Environment diagnostics now report exact installed dependency versions without loading the IndexTTS model, separating missing packages from runtime initialization failures.
- The Desktop launcher can refresh BigVGAN CUDA, Torch Compile, GPT acceleration, and DeepSpeed preflight results and explains each expected enablement or fallback.
- A JSON report exports the system, GPU, model validation, selected runtime settings, pinned revisions, and troubleshooting notes for issue reports.
- Preflight availability is not claimed as actual activation; the startup log and WebUI environment report remain authoritative. Optional DeepSpeed AIO/cuFile warnings do not determine speech-inference success.

### v0.12.0 low-VRAM and precision update

- `auto` selects BF16 only when the GPU supports it natively; older CUDA GPUs fall back to FP16, which is also selectable explicitly.
- Reference encoders now support `auto / same / cpu`; `auto` moves Wav2Vec/CAMPPlus to CPU below 10 GB VRAM.
- Optional fast default emotion reuses the speaker condition only when no independent emotion is supplied; it is off by default.
- Environment diagnostics now report GPU/VRAM and recommend precision, reference placement, and a safe acceleration mode without loading the model.
- A 28th low-VRAM workflow was added, and Registry publishing now succeeds only after the version becomes Active.

### v0.11.5 release protection

- Registry publish jobs are serialized per repository, preventing manual dispatches and delayed push events from uploading the same version concurrently.
- Before uploading, the workflow checks the exact version through the official read-only endpoint; existing Active/Pending versions are skipped successfully, and only a 404 permits publishing.
- Timeouts, 5xx responses, and malformed Registry responses are retried and then fail closed instead of publishing with unknown state.

### v0.11.4 stability update

- A cold or changed speaker reference now performs one Wav2Vec pass and one audio load/resample instead of duplicating both operations.
- Model Loader status explicitly reports Node/Core/Model versions, and the runtime node version is read from `pyproject.toml` to prevent drift.
- CI covers both the current Linux environment and the Windows Portable Python 3.10 / torch 2.8 baseline.
- The synthetic-prompt GPT KV-cache false-hit fix remains enabled and has a dedicated regression test.

- Plain batch lines can contain official `<text|pronunciation>` annotations without corrupting field parsing.
- JSON/SRT timelines validate paired bounds, ordering, and safe ranges; composition rejects abnormal allocations above 1 GiB before allocating memory.
- Single-speaker and multi-role ASR review preserve generated audio when a backend is missing or transcription/download fails.
- v0.10 Model Loader custom-path widget shifts are restored automatically; replacing weights in place now changes the runtime cache identity.
- Release All clears both IndexTTS and Whisper caches owned by this extension; audio.cpp now exposes Apple Metal.
- Windows DeepSpeed BF16 requests use the more compatible FP16 inference workspace and still fall back safely if initialization fails.
- All 27 UI workflows validate widget order/types, and Registry publishing is gated by tests against the current ComfyUI V3 API.
- Reference-audio caching now writes standard PCM16 WAV directly, including with current `torchaudio` installations that do not include TorchCodec.

## Implemented nodes

1. `IndexTTS 2.5 Model Loader · T8star-Aix`
   - Scans the standard model directory and `TTS` paths from `extra_model_paths.yaml`
   - Validates official model file sizes, with optional full SHA-256 verification
   - `auto / CUDA / CPU` device selection and `auto / bfloat16 / float16 / float32` precision
   - `auto / same / cpu` reference-encoder placement and optional default-emotion condition reuse
   - Persistent `safetensors` reference-conditioning cache enabled by default, with an advanced opt-out
   - Global lazy cache, per-model inference lock, and optional unload after generation or safe recycle every N runs
2. `IndexTTS 2.5 Emotion Control · T8star-Aix`
   - Follow the speaker reference
   - Independent emotion reference audio
   - Eight-dimensional emotion vector: happy, angry, sad, afraid, disgusted, melancholic, surprised, and calm
   - Text emotion description, with QwenEmotion loaded on demand on the inference device
3. `IndexTTS 2.5 Sampling Settings · T8star-Aix`
   - Stable defaults, random sampling, beam, temperature, top-p, and top-k
   - CFM diffusion steps, guidance rate, and noise temperature; official defaults remain `25 / 0.7 / 1.0`
   - Automatic language-aware segmentation (EN/ES 60, AR 80, JA 100, ZH 120 tokens) or a custom limit
   - Punctuation pause presets, custom milliseconds, explicit `<pause=0.5>`, segment gaps, and text normalization
4. `IndexTTS 2.5 Segmentation and Pause Preview · T8star-Aix`
   - Reads the official tokenizer vocabulary without loading neural network weights
   - Reports tokens, speech blocks, trailing pauses, and GPT acceleration risks for every segment; processed text can be passed directly to generation
5. `IndexTTS 2.5 Pronunciation Control · T8star-Aix`
   - Official `<text|pronunciation>` annotation syntax
   - Validation for tone-numbered Chinese Pinyin, English CMU phonemes, and Japanese Kana
   - Workflow-embedded pronunciation dictionary with longest-match-first and manual-annotation priority
   - Returns processed text and a complete replacement/validation report without mutating the cached model
6. `IndexTTS 2.5 Speech Generation · T8star-Aix`
   - Standard ComfyUI `AUDIO` input and output
   - Chinese, English, Japanese, Spanish, and Arabic entry points
   - Voice cloning, seed, and official `duration_factor=0.5–2.0` acoustic-duration adaptation; this is not natural prosodic speaking rate, and extreme values may sound stretched
   - Target duration through one-pass native length regulation, natural second-pass adaptation, silence padding, or exact compatibility mode
   - Optional voice clarity, clear narration, de-harsh, warmth, and peak-normalization post-processing
   - Generates one to four retained candidates and exposes an AUDIO list; selection combines ASR similarity with waveform quality when ASR is available and falls back to waveform quality alone
7. `IndexTTS 2.5 Voice Profile · T8star-Aix`
   - Packages a role name, standard AUDIO, default language, and optional role-specific emotion into a workflow voice profile
   - The companion Saved Voice node loads Desktop `.t8voice.zip` bundles without another reference-audio upload
8. `IndexTTS 2.5 Voice / Emotion Merge · T8star-Aix`
   - Dynamically accepts 1–16 voice profiles, each with its own voice and emotion; duplicate role names fail before queueing
9. `IndexTTS 2.5 Merge Voice Emotions · T8star-Aix`
   - Compatibility/search name matching the community term `Merge Voice Emotions`; output is identical to Voice / Emotion Merge
   - Merges role configurations, not multiple eight-dimensional vectors into a new emotion
10. `IndexTTS 2.5 Batch Dialogue / SRT · T8star-Aix`
    - Parses `role|text|language|duration factor|per-line emotion`, JSON arrays, and standard SRT
    - Every line can override the same role's emotion with a text description or eight-dimensional vector; an empty value inherits the role default
    - `<text|pronunciation>` inside a plain batch line is preserved as text; JSON remains available for complex content
    - SRT supports `[Role] text`, `Role: text`, and `[Role|emotion=text:angry] text`, with a structured preview
    - Dynamic prompt parsing is disabled for the script input, so JSON braces are preserved when queueing
11. `IndexTTS 2.5 Context-aware Per-line Emotion Suggestions · T8star-Aix`
    - Uses local QwenEmotion with surrounding dialogue while keeping speaker roles separate
    - Preserves manual per-line emotion by default and returns editable script/JSON without generating audio
12. `IndexTTS 2.5 Multi-role / SRT Generation · T8star-Aix`
    - Per-line inference, per-line AUDIO list, merged AUDIO, and a JSON report
    - `shift` conflict resolution or `overlay` timeline mixing; subtitle slots default to tail-safe `pad`, while `native/exact` are reserved for hard slots where trimming is acceptable
    - Optional per-line ASR retry, with every seed, score, and final selection recorded in the task report
13. `IndexTTS 2.5 Voice Post-processing · T8star-Aix`
    - Processes any ComfyUI AUDIO independently, with wet/dry strength and target peak, without FFmpeg
14. `IndexTTS 2.5 Environment and Optional Acceleration · T8star-Aix`
    - Checks native BF16, FP16, VRAM, the CUDA toolchain, Triton, FlashAttention, and DeepSpeed without loading the model
    - Reports installed torch/CUDA Runtime/FlashAttention/Triton/DeepSpeed/Ninja versions, then recommends precision, reference placement, and a safe mode
    - Never installs optional dependencies and keeps preflight availability distinct from the mode that actually initializes after startup
15. `IndexTTS 2.5 Timeline Editor · T8star-Aix`
    - Accepts a batch/SRT script and editable JSON to change per-line start, end, role, language, duration factor, and emotion override
    - Outputs a strictly validated script, structured JSON, and a standard `IMAGE` timeline preview
16. `IndexTTS 2.5 ASR Proofreading · T8star-Aix`
    - Uses optional local OpenAI Whisper or faster-whisper to transcribe AUDIO and compare it with target text
    - Reports normalized CER/WER, differences, word timestamps, threshold result, waveform alignment image, and complete JSON
17. `IndexTTS 2.5 Subtitle Rewrite · T8star-Aix`
    - Preserves original SRT timing or uses the generated audio timeline
    - Writes original text, all ASR results, or only results that pass proofreading
18. `IndexTTS 2.5 Reference Audio Quality · T8star-Aix`
    - Measures duration, leading/trailing silence, silence ratio, loudness, clipping, estimated SNR, and DC offset
    - Can trim silence and select the highest-energy section of an overlong reference without overwriting the source
19. `IndexTTS 2.5 Memory Control · T8star-Aix`
    - Reports IndexTTS/ASR caches and CUDA memory; Release All also clears this extension's Whisper cache
    - Never invokes ComfyUI-wide cleanup or unloads another node's models
20. `IndexTTS 2.5 audio.cpp Experimental Generation · T8star-Aix`
    - Isolated optional `audiocpp_cli` + IndexTTS2.5 GGUF route with CUDA/CPU/Vulkan/HIP/Metal, five languages, speed, and emotion controls
    - Accepts only manually installed local CLI and GGUF absolute paths; it performs no downloads and does not replace Python inference
21. `IndexTTS 2.5 Runtime Benchmark · T8star-Aix`
    - Warms up and measures the Model Loader mode that actually initialized, returning median/best RTF and CUDA peak VRAM
    - Keeps text, reference audio, and seed fixed; change the loader acceleration mode and rerun for a fair comparison
22. `IndexTTS 2.5 Update Check · T8star-Aix`
    - Manually checks the official main branch, official Hugging Face model, and node version
    - Returns JSON and a summary only; never downloads models, modifies the node, or runs automatically

Output is standard ComfyUI AUDIO at `22050 Hz`, `float32`, and `[1,1,T]`, ready for Save Audio, audio-combine, video, and other native nodes.

## Installation

After publication to the official Comfy Registry, search for `IndexTTS 2.5 · T8star-Aix` or node ID `indextts25-t8` in **ComfyUI Manager**, install it, and restart ComfyUI. Model weights must still be downloaded separately as described in [Model location](#model-location).

For manual installation, place this entire repository at:

```text
ComfyUI/custom_nodes/comfyui-indextts25-T8
```

Or clone it directly:

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-indextts25-t8.git comfyui-indextts25-T8
```

Install dependencies with **ComfyUI's own Python**:

```powershell
python -m pip install -r ComfyUI/custom_nodes/comfyui-indextts25-T8/requirements.txt
```

For the Windows portable build, run this from the ComfyUI directory:

```powershell
..\python_embeded\python.exe -m pip install -r custom_nodes\comfyui-indextts25-T8\requirements.txt
```

`requirements.txt` intentionally omits `torch`, `torchaudio`, and `torchvision` to avoid replacing ComfyUI's existing CUDA/PyTorch combination. If torchaudio is missing, install the version that exactly matches the existing torch build instead of installing the upstream project's complete torch lock file. Base dependencies include only runtime packages not provided by ComfyUI and have no upper bounds except the upstream compatibility boundary `transformers<5`.

DeepSpeed, FlashAttention, Triton, and the BigVGAN CUDA build toolchain are **optional accelerators** and are not in `requirements.txt`. The model loader and all standard inference features work without them.

ASR proofreading is also optional. Install it with ComfyUI's Python only when needed:

```powershell
python -m pip install "openai-whisper>=20250625" "opencc-python-reimplemented>=0.1.7"
```

Windows portable build:

```powershell
..\python_embeded\python.exe -m pip install "openai-whisper>=20250625" "opencc-python-reimplemented>=0.1.7"
```

For the faster CTranslate2 backend, install `faster-whisper>=1.2.0` and `opencc-python-reimplemented>=0.1.7`, then select the backend explicitly if desired. `pyproject.toml` also exposes `asr` and `asr-fast` optional dependency groups. The first proofreading run downloads the selected Whisper model to `ComfyUI/models/TTS/Whisper/`. Existing IndexTTS generation, timeline editing, and original-text subtitle rewriting continue to work without Whisper.

The pinned core has been verified on Windows, Python 3.10, and PyTorch 2.8. For other ComfyUI Python versions, run the environment checker first because upstream compatibility may be narrower.

### Dependency compatibility

`transformers` has been tested with `4.52.1` and `4.57.6` (the latter with `tokenizers 0.22.2`). The supported range is `transformers>=4.52.1,<5`; transformers 5.x changes generation and cache APIs and is not currently supported.

An existing ComfyUI installation using transformers 4.57.6 does not need a downgrade. To upgrade explicitly to the tested combination, use ComfyUI's own Python:

```powershell
python -m pip install --upgrade "transformers==4.57.6" "tokenizers==0.22.2"
```

Windows portable build:

```powershell
..\python_embeded\python.exe -m pip install --upgrade "transformers==4.57.6" "tokenizers==0.22.2"
```

Restart ComfyUI afterward. Do not install transformers 5.x by itself or reinstall ComfyUI's PyTorch for this purpose.

## Chinese numbers, dates, and years

Incorrect Arabic-digit pronunciation is normally a **text-front-end** issue, not a voice-cloning or reference-audio failure.
Enable **Text normalization (numbers/dates)** in `IndexTTS 2.5 Sampling Config`. Before inference, context determines
whether the value should be read digit by digit or as a quantity:

| Source text | Intended spoken form | Meaning |
|---|---|---|
| `1939年` | `一九三九年` | a year, read digit by digit |
| `1939个人` | `一千九百三十九个人` | a quantity |

This feature is an independent optional dependency: `wetext>=0.1.7,<0.2` on Windows/macOS and
`WeTextProcessing>=1.2.0,<2` on Linux. Run `IndexTTS 2.5 Environment & Optional Acceleration` and inspect
`text_normalization` in its JSON report. The backend is ready only when `verified` is `true` and the smoke-test output is
`一九三九年`.

From the node directory in a standard environment:

```powershell
python -m pip install -r requirements-text-normalization.txt
```

From the `ComfyUI_windows_portable` directory:

```powershell
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\comfyui-indextts25-T8\requirements-text-normalization.txt
```

Package-manager users can alternatively run `python -m pip install -e ".[text-normalization]"`. Restart ComfyUI after
installation. If the dependency is absent or fails the smoke test, generation continues with the original text; write the
intended spoken form explicitly in that case. This optional package does not affect model loading, duration adaptation,
or text emotion.

## Model location

Node code and model weights belong in two separate directories:

```text
ComfyUI/
├─ custom_nodes/
│  └─ comfyui-indextts25-T8/       # this GitHub node repository
└─ models/
   └─ TTS/
      └─ IndexTTS-2.5/             # every file from the complete HF repository
         ├─ config.yaml
         ├─ bpe.model
         ├─ gpt.pth
         ├─ codec.pth
         ├─ s2mel.pth
         ├─ feat1.pt
         ├─ feat2.pt
         ├─ wav2vec2bert_stats.pt
         ├─ multilingual_zh_ja_yue_char_del.tiktoken
         ├─ qwen0.6bemo4-merge/
         └─ hf_cache/
            ├─ w2v-bert-2.0/
            ├─ campplus_cn_common.bin
            └─ bigvgan/
```

Example Windows paths:

```text
D:\ComfyUI\models\TTS\IndexTTS-2.5\config.yaml
D:\ComfyUI\models\TTS\IndexTTS-2.5\bpe.model
D:\ComfyUI\models\TTS\IndexTTS-2.5\hf_cache\bigvgan\bigvgan_generator.pt
```

`config.yaml`, `bpe.model`, and `gpt.pth` must be direct children of `IndexTTS-2.5`. Do not leave an extra nested directory such as:

```text
ComfyUI/models/TTS/IndexTTS-2.5/IndexTTS-2.5-Comfy/config.yaml  # wrong
```

If a browser download extracts as `IndexTTS-2.5-Comfy`, rename it to `IndexTTS-2.5` or move all its contents into the standard directory. Restart or refresh ComfyUI after manual placement.

This is the only exception to shipping the whole repository: node code, manifests, scripts, licenses, examples, and tests are included here, while the approximately 7.7 GiB complete model follows ComfyUI's shared model-directory convention.

When sharing the node with another user, send the complete `comfyui-indextts25-T8` directory. The recipient must still download the model as described here, or receive the complete `IndexTTS-2.5` directory separately under `ComfyUI/models/TTS/`. Do not send only one `.py` file.

### Option 1: opt-in download in Model Loader (recommended)

1. Add the IndexTTS 2.5 Model Loader.
2. Enable “download/repair the complete model when missing.”
3. Read the license and disclaimer, then enable the license-acceptance checkbox.
4. Queue the workflow once. The node downloads or repairs the standard directory and continues loading when complete; it never downloads by default.

The complete bundle is approximately 7.7 GiB. Ensure adequate disk space and a stable connection. You can disable both download options afterward.

### Option 2: command-line download

After installing the node in `custom_nodes`, Hugging Face is the default and recommended source:

```powershell
python ComfyUI/custom_nodes/comfyui-indextts25-T8/scripts/download_models.py `
  --source huggingface `
  --accept-license
```

You can also download the complete directory with the Hugging Face CLI:

```powershell
hf download t8star/IndexTTS-2.5-Comfy `
  --local-dir "ComfyUI/models/TTS/IndexTTS-2.5"
```

ModelScope remains a compatible download path. Install its optional dependency first:

```powershell
python -m pip install -r ComfyUI/custom_nodes/comfyui-indextts25-T8/requirements-modelscope.txt
```

Then download from ModelScope:

```powershell
python ComfyUI/custom_nodes/comfyui-indextts25-T8/scripts/download_models.py `
  --source modelscope `
  --accept-license
```

When the node is not installed in `custom_nodes`, specify the ComfyUI root explicitly:

```powershell
python scripts/download_models.py --comfy-root "D:\ComfyUI" --source huggingface --accept-license
```

The downloader is pinned to the complete model manifest, performs full SHA-256 verification, and prepares the Wav2Vec2-BERT, CAMPPlus, and BigVGAN helper models. A direct download of `IndexTeam/IndexTTS-2.5` lacks the required `bpe.model`; use the complete repository or this downloader. Restart or refresh ComfyUI after manual installation so the model selector refreshes.

## Quick start

1. Use ComfyUI's native `Load Audio` to load a clean, single-speaker reference without background music, preferably 3–10 seconds.
2. Add the model loader and select `IndexTTS-2.5`.
3. Add Speech Generation, connect the model and reference audio, then enter text and language.
4. For polyphonic characters or proper nouns, add Pronunciation Control and connect its text output to Speech Generation.
5. Optionally connect Emotion Control and Sampling Settings. For long text, preview it with Segmentation and Pause Preview first.
6. Connect the generated AUDIO to `Save Audio`, optionally through Voice Post-processing.

### Multi-role, batch dialogue, and SRT

1. Add one Voice Profile per role and connect its corresponding `Load Audio`.
2. For independent emotions, add one Emotion Control per role and connect it to that profile's role emotion input.
3. Connect the profiles to Voice / Emotion Merge, or use the equivalent `Merge Voice Emotions` node.
4. Add Batch Dialogue / SRT and select the desired format.
5. Connect the model, role library, and script to Multi-role / SRT Generation.

```text
Emotion Control A ──► Voice Profile A ┐
Reference Audio A ──►                 │
                                      ├─► Voice / Emotion Merge ─► Multi-role / SRT Generation
Emotion Control B ──► Voice Profile B │
Reference Audio B ──►                 ┘
```

Each line inherits the emotion stored for its assigned role unless that line supplies an override. See
`23_multi_role_emotions.json` for role defaults and `31_per_line_emotion.json` for one role changing emotion
from line to line. Here, “merge” means collecting role configurations. To mix “60% sadness + 40% anger”
into one eight-dimensional emotion, set both dimensions in one vector.

For contextual suggestions, connect Model Loader and Batch Dialogue / SRT to Context Emotion Suggestions. The default
window reads two lines on each side while the prompt keeps roles separate. The first run only returns a suggested
script and JSON; review the vector and `strength`, then connect the script through Timeline Editor to Multi-role
Generation. See `32_context_emotion_suggestions.json` for the intentionally non-generating first stage.

Batch text uses one line per utterance. Language, duration factor, and per-line emotion are optional:

```text
Role A|Let me explain this calmly first.|EN|1.0|text:calm and composed
Role A|Why have you been lying to me!|EN|1.0|vector:0,0.8,0,0,0,0,0,0
Role A|This line returns to the role default.|EN|1.0
```

JSON is also supported:

```json
[
  {
    "role": "Role A",
    "text": "Let me explain this calmly first.",
    "language": "EN",
    "duration_factor": 1.0,
    "emotion": {"mode": "text", "text": "calm and composed", "strength": 0.75}
  },
  {
    "role": "Role A",
    "text": "Why have you been lying to me!",
    "language": "EN",
    "duration_factor": 1.0,
    "emotion": {
      "mode": "vector",
      "vector": [0, 0.8, 0, 0, 0, 0, 0, 0],
      "strength": 0.85
    }
  }
]
```

Vector order is always **happy, angry, sad, afraid, disgusted, melancholic, surprised, calm**, with each
value in `0–1`. Supported line modes are `inherit / speaker / text / vector`: omitted or `inherit` uses
the Voice Profile default, `speaker` follows the speaker reference, `text` uses a description, and
`vector` uses the eight values.

JSON can be pasted directly or loaded with an example workflow. Since v0.5.1, dynamic prompt parsing is explicitly disabled for this input; older behavior could interpret JSON braces as dynamic prompts and fail with `Expecting ',' delimiter` when queueing.

SRT example:

```srt
1
00:00:00,500 --> 00:00:02,600
[Role A|emotion=text:calm and composed] This is the first subtitle.

2
00:00:02,800 --> 00:00:05,000
[Role A|emotion=vector:0,0.8,0,0,0,0,0,0] The same role is angry now.
```

`shift` delays conflicting clips to avoid overlap. `overlay` preserves original SRT start times and safely
mixes overlaps. “Fit subtitle slot” defaults to tail-safe `pad`: short clips receive silence, while long clips
are preserved. `native` sends target frames directly to the length regulator but still pads or trims at the
end; `exact` also trims. Use those two only when hard alignment is more important than preserving a tail.

### Automatic segmentation, pauses, and target duration

Sampling Settings uses `auto` segmentation by default: 60 tokens for English and Spanish, 80 for Arabic, 100 for Japanese, and 120 for Chinese. Switch to `custom` to reproduce specific experimental settings. Segmentation and Pause Preview returns the tokenized segments and externally inserted silence before inference.

Pause presets are `off / natural / narration / dialogue / custom`. Explicit pauses work in the text with every preset:

```text
The first sentence ends here.<pause=0.8>Continue after eight hundred milliseconds.
You can also write <pause=500ms>five hundred milliseconds.
```

Punctuation presets split text into independent speech blocks, making them more precise than segment-gap silence but increasing inference calls. v0.8.1 backports the GPT synthesis-prompt KV-cache fix, so multiple pause blocks and long text no longer disable `gpt_accel` only because of that boundary issue. Incompatible sampling semantics still fall back safely to the standard GPT path.

Speech Generation provides five target-duration modes:

- `off`: use only the original `duration_factor`.
- `native`: distribute the total duration to the length regulator by text weight in one pass, then finish exactly; an overlong tail may be trimmed.
- `natural`: measure a first pass, calculate a new factor within 0.5–2.0, and infer a second time without trimming.
- `pad`: perform natural adaptation, pad short output with silence, and retain long output with a report warning; this is the safe subtitle-slot default.
- `exact`: perform natural adaptation, then pad or hard-trim to the exact sample; trimming may cut the tail and is intended for hard subtitle slots.

Built-in post-processing presets are `voice_clarity / clear_narration / deharsh / warm / normalize`. Select one directly on Speech Generation or use the standalone Voice Post-processing node for A/B comparison. `off` does not alter the waveform.

### Advanced CFM parameters

Advanced Sampling Settings inputs act directly on the spectrogram CFM stage:

- `diffusion_steps`: default 25; 40–50 is often steadier but runtime scales approximately with the step count.
- `inference_cfg_rate`: default 0.7; higher values strengthen voice/pitch conditioning but can over-smooth.
- `cfm_temperature`: default 1.0; lowering it to 0.8 may reduce jitter.

Change one value at a time and keep the generation `seed` fixed for A/B comparisons. Example `19_cfm_advanced.json` uses `40 / 0.85 / 0.8`; those are not new mandatory defaults.

### Optional acceleration modes

The model loader defaults to `off`, the most compatible mode with no extra dependencies:

- `precision=auto` selects native `bfloat16` when available, otherwise `float16` on CUDA; CPU uses `float32`.
- `reference_device=auto` moves reference encoders to CPU below 10 GB VRAM; `same` forces the main device and `cpu` always saves VRAM.
- `reuse_spk_cond_for_emo` reuses the speaker condition only without an independent emotion. It is off by default and may slightly change the result.

- `auto_safe`: enables the fused BigVGAN CUDA kernel only when Ninja and a CUDA/C++ build toolchain already exist.
- `bigvgan_cuda`: explicitly requests the fused BigVGAN kernel; the first run may compile it and failures fall back safely.
- `torch_compile`: requires Triton matching the current PyTorch build and has first-run compilation overhead.
- `gpt_accel`: requires FlashAttention and Triton. This path cannot represent every beam/top-p/top-k combination, so incompatible settings temporarily use standard GPT rather than silently changing sampling semantics. The synthesis-prompt KV cache is fixed in the bundled core.
- `deepspeed`: enabled only when explicitly selected and already installed. The node never installs it or treats it as required; performance varies by hardware.

When a dependency is missing or initialization fails, model info reports `effective=off` and the fallback reason. Use Environment and Optional Acceleration first, then decide whether to maintain a separate acceleration environment. Do not replace ComfyUI's torch/CUDA combination merely to enable acceleration.

The following exact wheels were verified for Windows, Python 3.10, and `torch 2.8.0+cu128` in the desktop package:

```powershell
pip install "triton-windows==3.4.0.post21"
pip install "https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu128torch2.8.0cxx11abiFALSE-cp310-cp310-win_amd64.whl"
pip install "https://github.com/6Morpheus6/deepspeed-windows-wheels/releases/download/v0.17.5/deepspeed-0.17.5%2Be1560d84-2.8torch_cu128-cp310-cp310-win_amd64.whl"
```

These wheels apply only to that exact ABI. Other Python, torch, or CUDA combinations require matching wheels. They are never installed automatically or added to base `requirements.txt`, which protects the host ComfyUI PyTorch environment.

vLLM-Omni is better suited to Linux server-throughput use and may be evaluated later as an isolated sidecar. It is not bundled into the Windows/ComfyUI base environment. The official TensorRT backend currently supplies an IndexTTS 2.0 engine only, so this integration does not claim IndexTTS 2.5 TensorRT support.

Reference audio longer than 15 seconds is truncated with a warning. Audio is content-hash cached in ComfyUI's temporary directory and does not pollute input/output. Concurrent inference on the same model is serialized because upstream voice/emotion caches are mutable.

`duration_factor` is a target-duration multiplier:

- `0.5`: shorter and faster
- `1.0`: default duration
- `2.0`: longer and slower

It controls the model's length regulator rather than applying a simple post-process time stretch. It does not guarantee word- or subtitle-level exact timing.

### ASR proofreading, subtitle rewriting, and timeline editing

ASR Proofreading accepts standard ComfyUI `AUDIO` and target text, then runs Whisper locally. Backends are `auto / openai_whisper / faster_whisper`; languages are `AUTO / ZH / EN / JA / ES / AR`; model sizes are `tiny / base / small / medium / turbo`; devices are `auto / CUDA / CPU`. Chinese and Japanese use character error rate (CER), while English, Spanish, and Arabic use word error rate (WER). Automatic language selects the metric from the text. Comparison normalizes NFKC, case, Simplified/Traditional Chinese, Chinese numbers, and punctuation. Reports include differences and word timestamps. `tiny` downloads quickly; larger models are usually more accurate but use more disk, VRAM, and time. Waveforms are passed directly to Whisper without requiring system FFmpeg.

Multi-role / SRT Generation can proofread each generated line and output both `rewritten_srt` and `timeline_report`. Subtitle timing modes are:

- `original`: preserve original SRT timing; batch lines without source timing use the actual generated timeline.
- `actual`: use real start/end times in the final mixed audio.

Subtitle text modes are:

- `original`: always use input text.
- `asr_all`: replace whenever a transcription exists.
- `asr_passed`: replace only when similarity passes the threshold.

ASR is optional post-generation proofreading. If its backend is unavailable or one line cannot be transcribed,
the node records a warning, keeps the original subtitle, and still returns the generated audio. Legacy workflows
that stored subtitle choices as `0 / 1 / 2`, or whose values shifted after a widget-order change, are normalized to
safe modes automatically. To repair an old node manually, set subtitle timing to `actual` and subtitle text to
`asr_passed`; if numeric choices remain visible, delete and add the generation node again.

Timeline Editor JSON uses milliseconds per row. This is a complete example; line numbers must be unique and contiguous:

```json
[
  {"line": 1, "role": "Role A", "text": "First line.", "language": "EN", "duration_factor": 1.0, "start_ms": 500, "end_ms": 2100},
  {"line": 2, "role": "Role B", "text": "Second line.", "language": "EN", "duration_factor": 0.9, "start_ms": 2300, "end_ms": 4500}
]
```

Connect `timeline image` to ComfyUI's native `Preview Image` to see the colored tracks. Connect the edited script to Multi-role / SRT Generation to mix with the new timeline. ASR reports can also be sent to Subtitle Rewrite to test different timing and text policies without rerunning TTS.

### Polyphonic characters and exact pronunciation

Even without Pronunciation Control, the official syntax can be used directly in generation text:

```text
小明<要求|YAO4 QIU2>这个题的答案是多少。
他在<银行|YIN2 HANG2>里<行走|XING2 ZOU3>了半天。
He had a <minute|M IH1 . N AH0 T> to check the <minute|M AY0 . N UW1 T> details.
彼は料理が<上手|じょうず>だが、囲碁では<上手|うわて>に負けた。
```

Chinese requires one tone-numbered Pinyin syllable per Han character. When a polyphonic character belongs to a continuous word, annotate the complete word. In upstream issue #792, `小明<要|YAO4>求…` may be overridden by contextual word meaning; use `小明<要求|YAO4 QIU2>…`. The node warns about isolated annotations inside continuous Chinese and mismatched character/syllable counts.

For batch rules, use Pronunciation Control. Each dictionary line is `text|pronunciation|language`:

```text
银行|YIN2 HANG2|ZH
行长|HANG2 ZHANG3|ZH
Bilibili|B IY1 . L IY1 . B IY1 . L IY1|EN
```

The dictionary is embedded in workflow JSON and travels with the workflow. Existing manual annotations always win; dictionary replacements use longest match first and never rewrite inside `<text|pronunciation>`. Strict validation is enabled by default and fails before queueing. When disabled, invalid entries remain unchanged and are recorded in the report. This node requires no additional G2P model and never mutates the cached model's global glossary.

See `example_workflows/README.md` for 33 ready-to-open UI workflows and 33 API prompts, including saved-voice and manual local audio.cpp examples. Upload `voice_reference.wav` and, for emotion-audio examples, `emotion_reference.wav` to ComfyUI input first.

### Optional audio.cpp experimental backend

Nothing is bundled, downloaded, or updated by the node, and the default loader is never changed. Download a runtime manually from the [official audio.cpp releases](https://github.com/0xShug0/audio.cpp/releases) and a GGUF from the [audio.cpp GGUF repository](https://huggingface.co/audio-cpp/audio.cpp-gguf), then enter the absolute `audiocpp_cli` and GGUF directory/file paths in Experimental Generation. The Q8 model is roughly 3.5 GB. audio.cpp uses an independent C++ text normalizer, so unusual dates, units, URLs, and Japanese/Spanish tokenization boundaries can differ from the official Python path. Compare all five languages, emotion modes, pronunciation overrides, and speed before production use.

## Environment and model checks

```powershell
python scripts/check_environment.py
python scripts/check_environment.py "D:\ComfyUI\models\TTS\IndexTTS-2.5"
python scripts/download_models.py --target "D:\ComfyUI\models\TTS\IndexTTS-2.5" --verify-only
```

The final command reads approximately 7.7 GiB and performs full hash verification.

See [`SECURITY_REVIEW.md`](SECURITY_REVIEW.md) for Registry scan notes and the boundaries of remaining sensitive operations. The release workflow no longer treats archive upload as Manager availability; only `NodeVersionStatusActive` passes.

## Pinned revisions

- IndexTTS code: `ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c`
- Upstream IndexTTS 2.5 model: `c39ce5ba981572cb187443877ff559dfb246ce63`
- Complete model bundle: `14166a7401f9f87f53770a1784390e8c0e9da15a`
- Model manifest: `manifests/model_2_5.json`

## Official project, model downloads, and acknowledgements

- Official IndexTTS repository: [index-tts/index-tts](https://github.com/index-tts/index-tts)
- Complete model for this node (recommended): [t8star/IndexTTS-2.5-Comfy](https://huggingface.co/t8star/IndexTTS-2.5-Comfy)
- IndexTTS 2.5 model on ModelScope: [IndexTeam/IndexTTS-2.5](https://modelscope.cn/models/IndexTeam/IndexTTS-2.5)
- IndexTTS 2.5 model on Hugging Face: [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5)

The complete model repository only reorganizes required runtime files without modifying upstream weights, and its model card records every source, pinned revision, and license. Special thanks to the IndexTTS, Wav2Vec2-BERT, CAMPPlus, and BigVGAN authors for their open-source work.

Thanks to the IndexTTS team for open-sourcing IndexTTS and the IndexTTS 2.5 model. This project is a third-party ComfyUI integration built on their open-source work; please support and follow the official project.

The bundled core includes upstream 2.5 compatibility for QwenEmotion labels, removal of the invalid `use_gpt_latent` path, and the torchaudio 2.9+ WAV anti-clipping fix. Below 10 GB VRAM, it automatically uses a low-memory strategy: segment long text, release QwenEmotion after text-emotion analysis, then run speech generation.

## Release maintenance

Before pushing to GitHub, update the semantic version in `pyproject.toml` with the repository script:

```powershell
python scripts/bump_version.py patch
```

Use `patch` for fixes and maintenance, `minor` for backward-compatible features, and `major` for breaking changes. Repository-level `AGENTS.md` requires future automation agents to check and update the version before every push, preventing reuse of immutable Comfy Registry versions.

## License and disclaimer

Read `LICENSE`, `LICENSE_ZH.txt`, `DISCLAIMER`, and `THIRD_PARTY_NOTICES.md` before use or distribution. The model license requires downstream projects to preserve applicable copyright and license notices and include non-endorsement and warranty disclaimers.

This node is a third-party integration, not an official product of Bilibili or the original IndexTTS rights holders. The original rights holders do not endorse, warrant, or accept liability for this derivative work.
