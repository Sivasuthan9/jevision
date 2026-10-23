"""Load a decision VLM, optionally merging a plain LoRA adapter into its weights, and expose its parts."""
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .config import ModelSpec


def load_model(spec: ModelSpec, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Return (processor, model) in eval mode on `device`."""
    processor = AutoProcessor.from_pretrained(spec.repo, revision=spec.revision)
    model = AutoModelForImageTextToText.from_pretrained(spec.repo, revision=spec.revision, dtype=dtype)
    model = model.to(device).eval()
    if spec.adapter:
        n = merge_lora(model, _local_dir(spec.adapter, spec.adapter_revision))
        print(f"merged LoRA modules: {n}", flush=True)
    return processor, model


ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _local_dir(repo_or_path: str, revision: str | None) -> Path:
    """Local directory holding the adapter files (downloads only those two files, or uses the cache)."""
    if Path(repo_or_path).is_dir():
        return Path(repo_or_path)
    from huggingface_hub import hf_hub_download

    paths = [Path(hf_hub_download(repo_or_path, name, revision=revision)) for name in ADAPTER_FILES]
    return paths[0].parent


@torch.no_grad()
def merge_lora(model, adapter_dir: Path) -> int:
    """Merge a plain LoRA adapter (no rsLoRA, no DoRA) in place: W <- W + (alpha / r) * B @ A.

    Written by hand so that the experiments do not depend on `peft`. Returns the number of merged modules.
    """
    from safetensors import safe_open

    adapter_dir = Path(adapter_dir)
    cfg = json.load(open(adapter_dir / "adapter_config.json"))
    assert not cfg.get("use_rslora") and not cfg.get("use_dora"), "only plain LoRA is supported"
    scale = cfg["lora_alpha"] / cfg["r"]
    params = dict(model.named_parameters())
    device = next(model.parameters()).device
    n_merged = 0
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as f:
        for key in f.keys():
            if not key.endswith("lora_A.weight"):
                continue
            name = key.replace("base_model.model.", "", 1).replace(".lora_A.weight", ".weight")
            a = f.get_tensor(key).float().to(device)
            b = f.get_tensor(key.replace("lora_A", "lora_B")).float().to(device)
            params[name].add_((scale * b @ a).to(params[name].dtype))
            n_merged += 1
    return n_merged


def language_model(model):
    """The decoder stack (layers, final norm) of an image-text model."""
    return model.model.language_model if hasattr(model.model, "language_model") else model.model
