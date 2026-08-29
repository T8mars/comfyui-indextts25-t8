# Registry security review notes

This repository vendors the official IndexTTS 2.5 inference core and exposes
optional model-download and audio.cpp integration. These legitimate operations
can match broad static-analysis rules, so the remaining security-sensitive
surfaces are documented here for reviewers and users.

## Network behavior

- Importing the node and running inference no longer probes public hosts or
  reads a process-wide download-source environment variable.
- `scripts/download_models.py` is an explicit user command. It accepts only
  `huggingface` or `modelscope`, downloads the pinned model revision recorded in
  `manifests/model_2_5.json`, and validates the formal model files against the
  pinned size/SHA-256 manifest.
- The vendored core may download fixed auxiliary model repositories only when
  their expected files are missing. The destination is the selected IndexTTS
  model directory; responses are never executed as Python or shell code.
- The audio.cpp one-click installer was removed in 0.20.5. The remaining
  experimental node has no component-download code and requires explicit local
  absolute paths for both the CLI and GGUF model.

## Optional audio.cpp process

`runtime/audiocpp_backend.py` launches only the executable path explicitly
selected by the user. It resolves and validates the executable, model, and
reference-audio paths; builds an argument list; uses `shell=False`; enforces a
timeout; validates allowed language/backend values; and requires the expected
output file. No text is interpolated into a shell command.

## Vendored upstream code

- The inference-only Transformers compatibility file disables XLA/FSDP
  environment switches because distributed training is outside this node's
  scope.
- Unused upstream training/CLI helpers are excluded from Registry archives via
  `.comfyignore` while remaining in GitHub for source traceability.
- The Qwen tokenizer language map contains the ISO language code `su`
  (Sundanese). It is static vocabulary metadata, not privilege escalation.

## Release status

Publishing an archive is not treated as release success. The GitHub workflow
polls the official Registry version endpoint and fails on Flagged/Banned or on
an activation timeout. Only `NodeVersionStatusActive` is reported as available
through ComfyUI Manager.
