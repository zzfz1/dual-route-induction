import os
from pathlib import Path

from nnsight import CONFIG, LanguageModel

REMOTE_MODEL_NAME = "meta-llama/Llama-3.1-8B"


def _iter_env_candidates():
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved


def _load_env_file(path: Path):
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _ensure_remote_env_loaded():
    missing = [name for name in ("NDIF_API_KEY", "HF_TOKEN") if not os.environ.get(name)]
    if not missing:
        return

    for candidate in _iter_env_candidates():
        if candidate.exists():
            _load_env_file(candidate)
        missing = [name for name in ("NDIF_API_KEY", "HF_TOKEN") if not os.environ.get(name)]
        if not missing:
            return


def load_remote_model(model_name, module=None):
    if model_name != REMOTE_MODEL_NAME:
        raise ValueError(
            f"Remote execution is only supported for {REMOTE_MODEL_NAME} in this script, got {model_name}."
        )

    _ensure_remote_env_loaded()
    missing = [
        name for name in ("NDIF_API_KEY", "HF_TOKEN") if not os.environ.get(name)
    ]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variables for NDIF remote execution: {joined}."
        )

    if module is not None:
        import cloudpickle

        cloudpickle.register_pickle_by_value(module)

    CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])
    return LanguageModel(model_name, device_map="cuda", attn_implementation="eager")
