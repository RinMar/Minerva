"""
Configuration management module.
Used to load and parse configuration variables such as model hyperparameters
from config.toml, and to provide the PROMPTS dictionary for system instructions.
"""
import os
import tomllib
import tomlkit
from src.paths import CONFIG_PATH, get_resource_path

# Model VRAM Estimation Constants (Qwen3-8B Q4_K_M GGUF)
MODEL_TOTAL_LAYERS = 65
MODEL_PER_LAYER_MB = 80.0
KV_PER_TOKEN_PER_LAYER_MB = 0.00012
VRAM_FIXED_OVERHEAD_MB = 300.0
CTX_MIN = 2048
CTX_MAX = 40960

DEFAULT_CONFIG_TOML = """\
[llm]
repo_id = "Qwen/Qwen3-8B-GGUF"
filename = "*Q4_K_M.gguf"
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
    default_config = {
        "llm": {
            "repo_id": "unsloth/Qwen3.5-9B-GGUF",
            "filename": "Qwen3.5-9B-Q4_K_M.gguf",
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

            # Backward compatibility: legacy performance_mode override if present
            legacy_mode = raw_config.get("settings", {}).get("performance_mode")
            if "n_gpu_layers" not in raw_config.get("llm", {}) and legacy_mode:
                if legacy_mode == "high":
                    llm_config["n_gpu_layers"] = MODEL_TOTAL_LAYERS
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


def update_model_settings(n_gpu_layers: int, n_ctx: int):
    """Update global config with new n_gpu_layers and n_ctx values."""
    save_model_settings(n_gpu_layers, n_ctx)
    new_config = load_config()
    config.clear()
    config.update(new_config)
    print(f"[Config] Updated model settings: n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}")
    return config


def update_config_mode(mode: str):
    """Legacy alias for backward compatibility."""
    n_layers = MODEL_TOTAL_LAYERS if mode.lower() == "high" else 24
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


def save_model_settings(n_gpu_layers: int, n_ctx: int):
    """Safely update the [llm] section in config.toml with n_gpu_layers and n_ctx."""
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

        # Remove legacy settings if present
        if "settings" in doc and "performance_mode" in doc["settings"]:
            del doc["settings"]["performance_mode"]
        if "high_performance" in doc:
            del doc["high_performance"]
        if "low_performance" in doc:
            del doc["low_performance"]

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(doc))

        print(f"[Config] Persisted llm settings: n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}")

    except Exception as e:
        print(f"Warning: Failed to save model settings to {CONFIG_PATH}: {e}")


def save_performance_mode(mode: str):
    """Legacy alias for backward compatibility."""
    n_layers = MODEL_TOTAL_LAYERS if mode.lower() == "high" else 24
    n_ctx = 40960 if mode.lower() == "high" else 8192
    save_model_settings(n_layers, n_ctx)


config = load_config()
PROMPTS_PATH = get_resource_path("resources/prompts.toml")
with open(PROMPTS_PATH, "rb") as f:
    PROMPTS = tomllib.load(f)
