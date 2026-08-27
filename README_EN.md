# comfyui-indextts25-T8

[简体中文](README.md) | **English**

Native ComfyUI V3 nodes for IndexTTS 2.5. The nodes are available under:

`T8star-Aix / Audio / IndexTTS 2.5`

Creator: **Bilibili: T8star-Aix**.

## Creator and resources

- Bilibili: [T8star-Aix](https://space.bilibili.com/385085361)
- YouTube: [T8star-Aix](https://www.youtube.com/@T8star-Aix/)
- Hugging Face: [t8star](https://huggingface.co/t8star)
- API sign-up (affiliate link): [api.seedance.nz](https://api.seedance.nz/sign-up?aff=5f4w)
- Online AI apps (RunningHub): [T8star-Aix profile](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- ComfyUI portable package: [Quark Drive](https://pan.quark.cn/s/264edb7e36bd)
- IndexTTS 2.5 model mirror: [Quark Drive](https://pan.quark.cn/s/c9c267081fbf)

This repository is locked to the IndexTTS 2.5 inference core and the official 2.5 model manifest. It will not fall back to or accidentally load IndexTTS 2.0.

Current baseline: **ComfyUI Node 0.12.1 · Desktop 0.12.0 · Core `ee40fa7d` · Model `c39ce5ba`**.
Desktop and Node are separate deliverables with independent versions; Core/Model identify the pinned official code and weight revisions.

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
   - Voice cloning, seed, and the official `duration_factor=0.5–2.0` duration/speed adaptation
   - Target duration through one-pass native length regulation, natural second-pass adaptation, silence padding, or exact compatibility mode
   - Optional voice clarity, clear narration, de-harsh, warmth, and peak-normalization post-processing
   - Optional local-ASR quality retries that change the seed and retain the highest-similarity result
7. `IndexTTS 2.5 Voice Profile · T8star-Aix`
   - Packages a role name, standard AUDIO, default language, and optional role-specific emotion into a workflow voice profile
8. `IndexTTS 2.5 Voice / Emotion Merge · T8star-Aix`
   - Dynamically accepts 1–16 voice profiles, each with its own voice and emotion; duplicate role names fail before queueing
9. `IndexTTS 2.5 Merge Voice Emotions · T8star-Aix`
   - Compatibility/search name matching the community term `Merge Voice Emotions`; output is identical to Voice / Emotion Merge
   - Merges role configurations, not multiple eight-dimensional vectors into a new emotion
10. `IndexTTS 2.5 Batch Dialogue / SRT · T8star-Aix`
    - Parses `role|text|language|duration factor`, JSON arrays, and standard SRT
    - `<text|pronunciation>` inside a plain batch line is preserved as text; JSON remains available for complex content
    - SRT role syntax supports `[Role] text` and `Role: text`, with a structured preview
    - Dynamic prompt parsing is disabled for the script input, so JSON braces are preserved when queueing
11. `IndexTTS 2.5 Multi-role / SRT Generation · T8star-Aix`
    - Per-line inference, per-line AUDIO list, merged AUDIO, and a JSON report
    - `shift` conflict resolution or `overlay` timeline mixing; subtitle slots use native one-pass adaptation by default, with a legacy second-pass compatibility option
    - Optional per-line ASR retry, with every seed, score, and final selection recorded in the task report
12. `IndexTTS 2.5 Voice Post-processing · T8star-Aix`
    - Processes any ComfyUI AUDIO independently, with wet/dry strength and target peak, without FFmpeg
13. `IndexTTS 2.5 Environment and Optional Acceleration · T8star-Aix`
    - Checks native BF16, FP16, VRAM, the CUDA toolchain, Triton, FlashAttention, and DeepSpeed without loading the model
    - Recommends precision, reference placement, and a safe acceleration mode; never installs optional dependencies
14. `IndexTTS 2.5 Timeline Editor · T8star-Aix`
    - Accepts a batch/SRT script and editable JSON to change per-line start, end, role, language, and duration factor in milliseconds
    - Outputs a strictly validated script, structured JSON, and a standard `IMAGE` timeline preview
15. `IndexTTS 2.5 ASR Proofreading · T8star-Aix`
    - Uses optional local OpenAI Whisper or faster-whisper to transcribe AUDIO and compare it with target text
    - Reports normalized CER/WER, differences, word timestamps, threshold result, waveform alignment image, and complete JSON
16. `IndexTTS 2.5 Subtitle Rewrite · T8star-Aix`
    - Preserves original SRT timing or uses the generated audio timeline
    - Writes original text, all ASR results, or only results that pass proofreading
17. `IndexTTS 2.5 Reference Audio Quality · T8star-Aix`
    - Measures duration, leading/trailing silence, silence ratio, loudness, clipping, estimated SNR, and DC offset
    - Can trim silence and select the highest-energy section of an overlong reference without overwriting the source
18. `IndexTTS 2.5 Memory Control · T8star-Aix`
    - Reports IndexTTS/ASR caches and CUDA memory; Release All also clears this extension's Whisper cache
    - Never invokes ComfyUI-wide cleanup or unloads another node's models
19. `IndexTTS 2.5 audio.cpp Experimental Generation · T8star-Aix`
    - Isolated optional `audiocpp_cli` + IndexTTS2.5 GGUF route with CUDA/CPU/Vulkan/HIP/Metal, five languages, speed, and emotion controls
    - Does not replace Python inference; the CLI and roughly 3.5 GB Q8 GGUF are separate downloads

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

Chinese number/date normalization is optional. On Windows you can install `wetext`; when it is missing or incompatible with the active Python version, the nodes automatically continue with the original text. Writing numbers as spoken words is recommended. This optional package does not affect the 2.5 model, duration adaptation, or text emotion.

## Model location

Model weights do not belong in the custom node directory. The standard location is:

```text
ComfyUI/models/TTS/IndexTTS-2.5/
```

This is the only exception to shipping the whole repository: node code, manifests, scripts, licenses, examples, and tests are included here, while the approximately 5 GB model follows ComfyUI's shared model-directory convention.

When sharing the node with another user, send the complete `comfyui-indextts25-T8` directory. The recipient must still download the model as described here, or receive the complete `IndexTTS-2.5` directory separately under `ComfyUI/models/TTS/`. Do not send only one `.py` file.

After installing the node in `custom_nodes`, install the optional ModelScope downloader first:

```powershell
python -m pip install -r ComfyUI/custom_nodes/comfyui-indextts25-T8/requirements-modelscope.txt
```

Then download from ModelScope:

```powershell
python ComfyUI/custom_nodes/comfyui-indextts25-T8/scripts/download_models.py `
  --source modelscope `
  --accept-license
```

Or from Hugging Face:

```powershell
python ComfyUI/custom_nodes/comfyui-indextts25-T8/scripts/download_models.py `
  --source huggingface `
  --accept-license
```

When the node is not installed in `custom_nodes`, specify the ComfyUI root explicitly:

```powershell
python scripts/download_models.py --comfy-root "D:\ComfyUI" --source modelscope --accept-license
```

The downloader is pinned to the official model manifest, performs full SHA-256 verification, and prepares the Wav2Vec2-BERT, MaskGCT, CAMPPlus, and BigVGAN helper models. Restart ComfyUI after download so the model selector refreshes.

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

Each line reads only the emotion stored for its assigned role. See `23_multi_role_emotions.json` for a complete runnable graph. Here, “merge” means collecting role configurations. To mix “60% sadness + 40% anger” into one eight-dimensional emotion, set both dimensions in a single Emotion Control node.

Batch text uses one line per utterance. Language and duration factor are optional:

```text
Role A|You finally made it.|EN|1.0
Role B|Good. Let's begin.|EN|0.9
```

JSON is also supported:

```json
[
  {"role": "Role A", "text": "First line.", "language": "EN", "duration_factor": 1.0},
  {"role": "Role B", "text": "Second line.", "language": "EN", "duration_factor": 0.9}
]
```

JSON can be pasted directly or loaded with an example workflow. Since v0.5.1, dynamic prompt parsing is explicitly disabled for this input; older behavior could interpret JSON braces as dynamic prompts and fail with `Expecting ',' delimiter` when queueing.

SRT example:

```srt
1
00:00:00,500 --> 00:00:02,600
[Role A] This is the first subtitle.

2
00:00:02,800 --> 00:00:05,000
Role B: This is the second subtitle.
```

`shift` delays conflicting clips to avoid overlap. `overlay` preserves original SRT start times and safely mixes overlaps. “Fit subtitle slot” defaults to `native`, assigning target frames directly to the length regulator in one inference pass and finishing at sample precision. When connected to an older external inference core without `target_duration`, the node falls back to the legacy second-pass `duration_factor` method and reports it. Extreme target-duration differences can still reduce naturalness.

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
- `native`: distribute the total duration to the length regulator by text weight in one pass, including external pauses; recommended first.
- `natural`: measure a first pass, calculate a new factor within 0.5–2.0, and infer a second time without trimming.
- `pad`: perform natural adaptation, pad short output with silence, and retain long output with a report warning.
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

See `example_workflows/README.md` for 28 ready-to-open UI workflows and 28 API prompts covering basic cloning, speed comparison, emotion reference audio, eight-dimensional emotion, text emotion, random-sampling long text, five-language generation, Chinese polyphonic characters, English CMU phonemes, Japanese Kana, multi-role dialogue, JSON batch lines, SRT, acceleration diagnostics, segmentation preview, explicit pauses, native target duration, advanced CFM parameters, audio post-processing, ASR proofreading, timeline editing, subtitle rewriting, independent per-role emotions, reference-audio quality, ASR retries, model recycling, the experimental audio.cpp backend, and low-VRAM FP16. Upload `voice_reference.wav` and, for emotion-audio examples, `emotion_reference.wav` to ComfyUI input first.

### Optional audio.cpp experimental backend

The node provides a shell-free CLI connector only. It does not bundle a third-party executable or GGUF weights and never changes the default loader. Download a matching Windows CLI from the [official audio.cpp releases](https://github.com/0xShug0/audio.cpp/releases), then obtain `IndexTTS2.5-GGUF` from the [audio.cpp GGUF repository](https://huggingface.co/audio-cpp/audio.cpp-gguf). The current Q8 file is roughly 3.5 GB. audio.cpp uses an independent C++ text normalizer, so unusual dates, units, URLs, and Japanese/Spanish tokenization boundaries can differ from the official Python path. Compare all five languages, emotion modes, pronunciation overrides, and speed before production use.

## Environment and model checks

```powershell
python scripts/check_environment.py
python scripts/check_environment.py "D:\ComfyUI\models\TTS\IndexTTS-2.5"
python scripts/download_models.py --target "D:\ComfyUI\models\TTS\IndexTTS-2.5" --verify-only
```

The final command reads approximately 5 GB and performs full hash verification.

See [`SECURITY_REVIEW.md`](SECURITY_REVIEW.md) for Registry scan notes and the boundaries of remaining sensitive operations. The release workflow no longer treats archive upload as Manager availability; only `NodeVersionStatusActive` passes.

## Pinned revisions

- IndexTTS code: `ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c`
- IndexTTS 2.5 model: `c39ce5ba981572cb187443877ff559dfb246ce63`
- Model manifest: `manifests/model_2_5.json`

## Official project, model downloads, and acknowledgements

- Official IndexTTS repository: [index-tts/index-tts](https://github.com/index-tts/index-tts)
- IndexTTS 2.5 model on ModelScope: [IndexTeam/IndexTTS-2.5](https://modelscope.cn/models/IndexTeam/IndexTTS-2.5)
- IndexTTS 2.5 model on Hugging Face: [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5)

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
