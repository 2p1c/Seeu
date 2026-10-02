from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
log = logging.getLogger("roommind.weights")


def configure_hf_endpoint() -> str:
    endpoint = os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")
    os.environ["HF_ENDPOINT"] = endpoint
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        constants.ENDPOINT = endpoint
        if hasattr(constants, "HF_HUB_OFFLINE"):
            constants.HF_HUB_OFFLINE = False
    return endpoint


def local_model_ready(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def load_auto_model(model_path: str, device: str, dtype):
    from transformers import AutoModel

    kwargs = {"local_files_only": True, "low_cpu_mem_usage": True, "dtype": dtype}
    try:
        model = AutoModel.from_pretrained(model_path, **kwargs)
    except TypeError:
        kwargs.pop("dtype")
        model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, **kwargs)
    model.eval()
    return model.to(device=device, dtype=dtype)


def ensure_local_model(model_id: str, local_dir: Path) -> str:
    candidate = Path(model_id).expanduser()
    if local_model_ready(candidate):
        return str(candidate.resolve())
    if local_model_ready(local_dir):
        return str(local_dir)

    endpoint = configure_hf_endpoint()
    local_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s via %s", model_id, local_dir, endpoint)
    offline = os.environ.pop("HF_HUB_OFFLINE", None)
    transformers_offline = os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=model_id, local_dir=str(local_dir), local_files_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"无法下载 {model_id} 到 {local_dir}。\n"
            f"当前 HF_ENDPOINT={endpoint}\n"
            "可以先设镜像再重试:\n"
            f"  export HF_ENDPOINT={DEFAULT_HF_ENDPOINT}\n"
            f"  huggingface-cli download {model_id} --local-dir {local_dir}"
            f" --endpoint {DEFAULT_HF_ENDPOINT}"
        ) from exc
    finally:
        if offline is not None:
            os.environ["HF_HUB_OFFLINE"] = offline
        if transformers_offline is not None:
            os.environ["TRANSFORMERS_OFFLINE"] = transformers_offline
    if not local_model_ready(local_dir):
        raise RuntimeError(f"下载后的 {local_dir} 里没有 config.json 和权重文件")
    return str(local_dir)
