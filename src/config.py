"""
Configuration management module.
Used to load and parse configuration variables such as model hyperparameters
from config.toml, and to provide the PROMPTS dictionary for system instructions.
"""
import os
import tomllib
import tomlkit
from src.paths import CONFIG_PATH

DEFAULT_CONFIG_TOML = """\
[high_performance]
n_ctx = 40960
n_gpu_layers = -1
device = "cuda"

[low_performance]
n_ctx = 8192
n_gpu_layers = 24
device = "cpu"

[llm]
repo_id = "Qwen/Qwen3-8B-GGUF"
filename = "Qwen3-8B-Q4_K_M.gguf"
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

[settings]
performance_mode = "low"
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
            "n_ctx": 32768,
            "n_gpu_layers": -1,
            "n_batch": 512,
            "use_mmap": True,
            "use_mlock": True,
            "verbose": False,
            "device": "cpu"
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

            # Identify mode: 1. Environment, 2. Config file, 3. Default "low"
            env_mode = os.environ.get("MINERVA_PERFORMANCE")
            config_mode = raw_config.get("settings", {}).get("performance_mode")
            mode = (env_mode or config_mode or "low").lower()

            # Base LLM settings
            llm_config = raw_config.get("llm", {})

            # Merge models settings
            models_config = default_config["models"].copy()
            models_config.update(raw_config.get("models", {}))
            raw_config["models"] = models_config

            # Merge preset
            preset_key = f"{mode}_performance"
            if preset_key in raw_config:
                llm_config.update(raw_config[preset_key])

            raw_config["llm"] = llm_config
            raw_config["performance_mode"] = mode
            return raw_config

    except Exception as e:
        print(f"Warning: Failed to load configuration from {CONFIG_PATH}: {e}")
        return default_config


def update_config_mode(mode: str):
    """Update the global config object with a new performance mode."""
    os.environ["MINERVA_PERFORMANCE"] = mode.lower()
    new_config = load_config()
    config.clear()
    config.update(new_config)
    print(f"[Config] Updated performance mode to: {mode}")
    return config


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


def save_performance_mode(mode: str):
    """Safely update the [settings] section in config.toml with the performance mode."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                doc = tomlkit.parse(f.read())
        else:
            doc = tomlkit.document()

        if "settings" not in doc:
            doc["settings"] = tomlkit.table()

        doc["settings"]["performance_mode"] = mode.lower()

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(doc))

        print(f"[Config] Persisted performance mode: {mode}")

    except Exception as e:
        print(f"Warning: Failed to save performance mode to {CONFIG_PATH}: {e}")


config = load_config()

PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "models", "prompts.toml")
with open(PROMPTS_PATH, "rb") as f:
    PROMPTS = tomllib.load(f)
