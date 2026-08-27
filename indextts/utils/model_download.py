"""
Model download utility that automatically switches between HuggingFace Hub and
ModelScope based on the detected network environment.

All auxiliary models are downloaded to ``{model_dir}/hf_cache/`` at startup
via ``ensure_models_available()``, so no downloads happen during inference.
"""

import logging
import os
import shutil
import tempfile

from indextts.utils.network_detection import need_proxy

logger = logging.getLogger(__name__)

_USING_MODELSCOPE: bool | None = None


def set_download_source(source: str) -> None:
    """Select a source explicitly for the current download command."""

    global _USING_MODELSCOPE
    normalized = str(source).strip().lower()
    if normalized not in {"huggingface", "modelscope"}:
        raise ValueError(f"Unsupported download source: {source}")
    _USING_MODELSCOPE = normalized == "modelscope"


def _get_using_modelscope() -> bool:
    global _USING_MODELSCOPE
    if _USING_MODELSCOPE is None:
        _USING_MODELSCOPE = need_proxy()
    return _USING_MODELSCOPE

# Mapping from HuggingFace repo_id to ModelScope model_id.
HF_TO_MODELSCOPE_REPO_MAP = {
    "funasr/campplus": "iic/speech_campplus_sv_zh-cn_16k-common",
    "facebook/w2v-bert-2.0": "AI-ModelScope/w2v-bert-2.0",
}

# Default BigVGAN repo (also in config.yaml, but needed for pre-download)
_BIGVGAN_REPO = "nvidia/bigvgan_v2_22khz_80band_256x"
_AUXILIARY_REVISIONS = {
    "facebook/w2v-bert-2.0": "da985ba0987f70aaeb84a80f2851cfac8c697a7b",
    "funasr/campplus": "e4b6ede7ce16997aff4ae69fbca1f0175e2afede",
    _BIGVGAN_REPO: "633ff708ed5b74903e86ff1298cf4a98e921c513",
}


def _download_single_file(
    repo_id: str,
    filename: str,
    local_path: str,
    *,
    revision: str | None = None,
) -> str:
    """Download a single file from a HF/ModelScope repo to a specific local path."""
    from indextts.utils.examples_downloader import _download_file

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if _get_using_modelscope():
        ms_model_id = HF_TO_MODELSCOPE_REPO_MAP.get(repo_id, repo_id)
        # Only try ModelScope if we have a known mapping
        if ms_model_id != repo_id:
            try:
                from modelscope.hub.file_download import model_file_download
                tmp = model_file_download(model_id=ms_model_id, file_path=filename)
                shutil.copy2(tmp, local_path)
                return local_path
            except Exception:
                pass
        # Fallback to hf-mirror.com
        resolved_revision = revision or "main"
        url = f"https://hf-mirror.com/{repo_id}/resolve/{resolved_revision}/{filename}"
    else:
        resolved_revision = revision or "main"
        url = f"https://huggingface.co/{repo_id}/resolve/{resolved_revision}/{filename}"

    print(f">> Downloading {repo_id}/{filename} from {url} ...")
    _download_file(url, local_path, timeout=300)
    print(f">> Downloaded {local_path}")
    return local_path


_VERSION_TO_REPO = {
    "2": "IndexTeam/IndexTTS-2",
    "2.5": "IndexTeam/IndexTTS-2.5",
}


def ensure_config_available(model_dir: str, version: str = "2.5") -> None:
    """Download the version-matched config.yaml when it is missing."""
    if version not in _VERSION_TO_REPO:
        supported = ", ".join(sorted(_VERSION_TO_REPO))
        raise ValueError(f"Unsupported model version {version!r}. Supported versions: {supported}")
    model_dir = model_dir or "."
    config_path = os.path.join(model_dir, "config.yaml")
    if os.path.isfile(config_path):
        return
    print(f">> config.yaml not found in {model_dir}, downloading...")
    _download_single_file(_VERSION_TO_REPO[version], "config.yaml", config_path)
    print(">> config.yaml downloaded.")


def ensure_models_available(model_dir: str, bigvgan_repo: str = _BIGVGAN_REPO) -> dict:
    """
    Download all auxiliary models to ``{model_dir}/hf_cache/`` if missing.

    Call this once at startup before creating ``IndexTTS2``.

    Returns a dict of local paths:
        - ``w2v_bert``: directory containing w2v-bert-2.0 model
        - ``campplus``: path to campplus_cn_common.bin
        - ``bigvgan``: directory containing config.json + bigvgan_generator.pt
    """
    cache_dir = os.path.join(model_dir, "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}

    # 1. w2v-bert-2.0 (full repo — needed by SeamlessM4T and Wav2Vec2BertModel)
    w2v_dir = os.path.join(cache_dir, "w2v-bert-2.0")
    w2v_required = ("config.json", "model.safetensors", "preprocessor_config.json")
    if not all(os.path.isfile(os.path.join(w2v_dir, name)) for name in w2v_required):
        print(f">> Downloading w2v-bert-2.0 to {w2v_dir}...")
        snapshot_download(
            "facebook/w2v-bert-2.0",
            local_dir=w2v_dir,
            revision=_AUXILIARY_REVISIONS["facebook/w2v-bert-2.0"],
            allow_patterns=list(w2v_required),
        )
    paths["w2v_bert"] = w2v_dir

    # 2. CAMPPlus speaker embedding model
    campplus_path = os.path.join(cache_dir, "campplus_cn_common.bin")
    if not os.path.isfile(campplus_path):
        print(f">> Downloading CAMPPlus to {campplus_path}...")
        _download_single_file(
            "funasr/campplus",
            "campplus_cn_common.bin",
            campplus_path,
            revision=_AUXILIARY_REVISIONS["funasr/campplus"],
        )
    paths["campplus"] = campplus_path

    # 3. BigVGAN vocoder (config + weights)
    bigvgan_dir = os.path.join(cache_dir, "bigvgan")
    if not all(
        os.path.isfile(os.path.join(bigvgan_dir, name))
        for name in ("config.json", "bigvgan_generator.pt")
    ):
        print(f">> Downloading BigVGAN to {bigvgan_dir}...")
        os.makedirs(bigvgan_dir, exist_ok=True)
        _download_single_file(
            bigvgan_repo,
            "config.json",
            os.path.join(bigvgan_dir, "config.json"),
            revision=_AUXILIARY_REVISIONS.get(bigvgan_repo),
        )
        _download_single_file(
            bigvgan_repo,
            "bigvgan_generator.pt",
            os.path.join(bigvgan_dir, "bigvgan_generator.pt"),
            revision=_AUXILIARY_REVISIONS.get(bigvgan_repo),
        )
    paths["bigvgan"] = bigvgan_dir

    print(">> All auxiliary models ready.")
    return paths


def snapshot_download(repo_id: str, local_dir: str, revision=None, force_download=False, **kwargs) -> str:
    """Download an entire model repository (HuggingFace or ModelScope)."""
    if _get_using_modelscope():
        return _snapshot_from_modelscope(repo_id, local_dir, revision)
    else:
        from huggingface_hub import snapshot_download as _hf_snapshot
        logger.info(f"Downloading repo from HuggingFace: {repo_id}")
        return _hf_snapshot(
            repo_id=repo_id, local_dir=local_dir, revision=revision,
            force_download=force_download, **kwargs,
        )


def _snapshot_from_modelscope(model_id: str, local_dir: str, revision=None) -> str:
    """Download an entire model repository from ModelScope."""
    ms_model_id = HF_TO_MODELSCOPE_REPO_MAP.get(model_id, model_id)
    ms_revision = revision
    if ms_model_id != model_id:
        logger.info(f"ModelScope: mapped '{model_id}' -> '{ms_model_id}'")
        # Hugging Face and ModelScope use different revision identifiers.
        # The complete manifest still verifies the downloaded bytes afterward.
        ms_revision = None

    from modelscope.hub.snapshot_download import snapshot_download as _ms_snapshot
    logger.info(f"Downloading repo from ModelScope: {ms_model_id}")

    # Check if files exist in a subdirectory from a previous download
    existing_subdir = os.path.join(local_dir, ms_model_id)
    if os.path.isdir(existing_subdir) and os.listdir(existing_subdir):
        for item in os.listdir(existing_subdir):
            src = os.path.join(existing_subdir, item)
            dst = os.path.join(local_dir, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(existing_subdir, ignore_errors=True)
        return local_dir

    with tempfile.TemporaryDirectory() as tmpdir:
        _ms_snapshot(model_id=ms_model_id, cache_dir=tmpdir, revision=ms_revision)
        downloaded = os.path.join(tmpdir, ms_model_id)
        if not os.path.isdir(downloaded):
            for root, dirs, files in os.walk(tmpdir):
                if files and root != tmpdir:
                    downloaded = root
                    break
        os.makedirs(local_dir, exist_ok=True)
        for item in os.listdir(downloaded):
            src = os.path.join(downloaded, item)
            dst = os.path.join(local_dir, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
    return local_dir
