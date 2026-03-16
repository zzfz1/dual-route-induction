import os

from nnsight import CONFIG, LanguageModel

REMOTE_MODEL_NAME = "meta-llama/Llama-3.1-8B"


def load_remote_model(model_name, module=None):
    if model_name != REMOTE_MODEL_NAME:
        raise ValueError(
            f"Remote execution is only supported for {REMOTE_MODEL_NAME} in this script, got {model_name}."
        )

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
