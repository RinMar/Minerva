"""
Configuration management module.
Used to load and parse configuration variables such as model hyperparameters
from config.toml, and to provide the PROMPTS dictionary for system instructions.
"""
import os
import tomllib
import tomlkit
from src.paths import CONFIG_PATH, get_resource_path

# ── Model Registry ──
# Each entry defines the HuggingFace repo, GGUF filename, display metadata,
# and VRAM estimation constants specific to that model's architecture.
MODEL_REGISTRY = {
    "qwen3.5-9b": {
        "name": "Qwen3.5 9B",
        "repo_id": "unsloth/Qwen3.5-9B-GGUF",
        "filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "description": "🧠 Stronger — Higher quality responses and deeper reasoning. Can be slower on limited hardware.",
        "total_layers": 65,
        "per_layer_mb": 80.0,
        "kv_per_token_per_layer_mb": 0.00012,
    },
    "qwen3.5-4b": {
        "name": "Qwen3.5 4B",
        "repo_id": "unsloth/Qwen3.5-4B-GGUF",
        "filename": "Qwen3.5-4B-Q4_K_M.gguf",
        "description": "⚡ Faster — Lightweight and responsive. Great for quick tasks on lower-end hardware.",
        "total_layers": 37,
        "per_layer_mb": 45.0,
        "kv_per_token_per_layer_mb": 0.00008,
    },
}

DEFAULT_MODEL_KEY = "qwen3.5-9b"

# Shared constants (model-independent)
VRAM_FIXED_OVERHEAD_MB = 300.0
CTX_MIN = 2048
CTX_MAX = 40960


def get_active_model_profile(cfg=None):
    """Return the MODEL_REGISTRY entry for the currently selected model."""
    if cfg is None:
        cfg = config if 'config' in globals() else {}
    key = cfg.get("llm", {}).get("model_key", DEFAULT_MODEL_KEY)
    return MODEL_REGISTRY.get(key, MODEL_REGISTRY[DEFAULT_MODEL_KEY])

DEFAULT_CONFIG_TOML = """\
[llm]
model_key = "qwen3.5-9b"
n_ctx = 8192
n_gpu_layers = 0
n_batch = 512
use_mmap = true
use_mlock = true
verbose = false


[models]
embedding_model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
reranker_model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
triplet_model_id = "Babelscape/rebel-large"

[user]
last_user_id = 1
last_user_name = "user"
"""


def _ensure_config_exists():
    """Write the default config.toml into the app data dir if it doesn't exist."""
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_TOML)
        print(f"Created default config at: {CONFIG_PATH}")


def load_config():
    default_profile = MODEL_REGISTRY[DEFAULT_MODEL_KEY]
    default_config = {
        "llm": {
            "model_key": DEFAULT_MODEL_KEY,
            "repo_id": default_profile["repo_id"],
            "filename": default_profile["filename"],
            "n_ctx": 8192,
            "n_gpu_layers": 0,
            "n_batch": 512,
            "use_mmap": True,
            "use_mlock": True,
            "verbose": False,
        },
        "models": {
            "embedding_model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "reranker_model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "triplet_model_id": "Babelscape/rebel-large"
        }
    }

    _ensure_config_exists()

    try:
        with open(CONFIG_PATH, "rb") as f:
            raw_config = tomllib.load(f)

            # Base LLM settings
            llm_config = default_config["llm"].copy()
            llm_config.update(raw_config.get("llm", {}))

            # Resolve repo_id and filename from model registry based on model_key
            model_key = llm_config.get("model_key", DEFAULT_MODEL_KEY)
            if model_key in MODEL_REGISTRY:
                profile = MODEL_REGISTRY[model_key]
                llm_config["repo_id"] = profile["repo_id"]
                llm_config["filename"] = profile["filename"]
            elif "repo_id" not in llm_config:
                # Fallback for old configs without model_key
                profile = MODEL_REGISTRY[DEFAULT_MODEL_KEY]
                llm_config["model_key"] = DEFAULT_MODEL_KEY
                llm_config["repo_id"] = profile["repo_id"]
                llm_config["filename"] = profile["filename"]

            # Backward compatibility: legacy performance_mode override if present
            legacy_mode = raw_config.get("settings", {}).get("performance_mode")
            if "n_gpu_layers" not in raw_config.get("llm", {}) and legacy_mode:
                total_layers = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY[DEFAULT_MODEL_KEY])["total_layers"]
                if legacy_mode == "high":
                    llm_config["n_gpu_layers"] = total_layers
                    llm_config["n_ctx"] = 40960
                else:
                    llm_config["n_gpu_layers"] = 24
                    llm_config["n_ctx"] = 8192

            # Merge models settings
            models_config = default_config["models"].copy()
            models_config.update(raw_config.get("models", {}))

            raw_config["llm"] = llm_config
            raw_config["models"] = models_config
            return raw_config

    except Exception as e:
        print(f"Warning: Failed to load configuration from {CONFIG_PATH}: {e}")
        return default_config


def update_model_settings(n_gpu_layers: int, n_ctx: int, model_key: str = None):
    """Update global config with new n_gpu_layers, n_ctx, and optionally model_key."""
    save_model_settings(n_gpu_layers, n_ctx, model_key)
    new_config = load_config()
    config.clear()
    config.update(new_config)
    key_info = f", model_key={model_key}" if model_key else ""
    print(f"[Config] Updated model settings: n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}{key_info}")
    return config


def update_config_mode(mode: str):
    """Legacy alias for backward compatibility."""
    profile = get_active_model_profile()
    n_layers = profile["total_layers"] if mode.lower() == "high" else 24
    n_ctx = 40960 if mode.lower() == "high" else 8192
    return update_model_settings(n_layers, n_ctx)


def save_last_user(user_id: int, user_name: str):
    """Safely update the [user] section in config.toml using tomlkit."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                doc = tomlkit.parse(f.read())
        else:
            doc = tomlkit.document()

        if "user" not in doc:
            doc["user"] = tomlkit.table()

        doc["user"]["last_user_id"] = user_id
        doc["user"]["last_user_name"] = user_name

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(doc))

        print(f"[Config] Persisted last user: {user_name} (ID: {user_id})")

    except Exception as e:
        print(f"Warning: Failed to save last user to {CONFIG_PATH}: {e}")


def save_model_settings(n_gpu_layers: int, n_ctx: int, model_key: str = None):
    """Safely update the [llm] section in config.toml with n_gpu_layers, n_ctx, and optionally model_key."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                doc = tomlkit.parse(f.read())
        else:
            doc = tomlkit.document()

        if "llm" not in doc:
            doc["llm"] = tomlkit.table()

        doc["llm"]["n_gpu_layers"] = int(n_gpu_layers)
        doc["llm"]["n_ctx"] = int(n_ctx)

        if model_key and model_key in MODEL_REGISTRY:
            doc["llm"]["model_key"] = model_key
            # Remove legacy repo_id/filename — these are now derived from model_key
            if "repo_id" in doc["llm"]:
                del doc["llm"]["repo_id"]
            if "filename" in doc["llm"]:
                del doc["llm"]["filename"]

        # Remove legacy settings if present
        if "settings" in doc and "performance_mode" in doc["settings"]:
            del doc["settings"]["performance_mode"]
        if "high_performance" in doc:
            del doc["high_performance"]
        if "low_performance" in doc:
            del doc["low_performance"]

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(doc))

        key_info = f", model_key={model_key}" if model_key else ""
        print(f"[Config] Persisted llm settings: n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}{key_info}")

    except Exception as e:
        print(f"Warning: Failed to save model settings to {CONFIG_PATH}: {e}")


def save_performance_mode(mode: str):
    """Legacy alias for backward compatibility."""
    profile = get_active_model_profile()
    n_layers = profile["total_layers"] if mode.lower() == "high" else 24
    n_ctx = 40960 if mode.lower() == "high" else 8192
    save_model_settings(n_layers, n_ctx)


config = load_config()
PROMPTS_PATH = get_resource_path("resources/prompts.toml")
with open(PROMPTS_PATH, "rb") as f:
    PROMPTS = tomllib.load(f)
