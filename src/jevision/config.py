"""Model registry: short names mapped to Hugging Face repositories and pinned revisions (configs/models.yaml)."""
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "models.yaml"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    revision: str | None = None
    adapter: str | None = None
    adapter_revision: str | None = None
    note: str = ""


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, ModelSpec]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {name: ModelSpec(name=name, **fields) for name, fields in raw["models"].items()}


def resolve_model(name_or_path: str, registry_path: Path = DEFAULT_REGISTRY) -> ModelSpec:
    """A registry name, or any Hugging Face id / local directory used as is."""
    registry = load_registry(registry_path)
    if name_or_path in registry:
        return registry[name_or_path]
    return ModelSpec(name=Path(name_or_path).name, repo=name_or_path)
