"""
Configuration management module.
Used to load and parse configuration variables such as model hyperparameters
from config.toml, and to provide the PROMPTS dictionary for system instructions.
"""
import os
import tomllib

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
            "repo_id": "Qwen/Qwen3-8B-GGUF",
            "filename": "*Q4_K_M.gguf",
            "n_ctx": 32768,
            "n_gpu_layers": -1,
            "n_batch": 512,
            "use_mmap": True,
            "use_mlock": True,
            "verbose": False,
            "device": "cpu"
        }
    }

    _ensure_config_exists()

    try:
        with open(CONFIG_PATH, "rb") as f:
            raw_config = tomllib.load(f)

            # Identify mode from environment variable
            mode = os.environ.get("MINERVA_PERFORMANCE", "low").lower()

            # Base LLM settings
            llm_config = raw_config.get("llm", {})

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


config = load_config()

PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "models", "prompts.toml")
with open(PROMPTS_PATH, "rb") as f:
    PROMPTS = tomllib.load(f)
