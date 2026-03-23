"""
Configuration management module.
Used to load and parse configuration variables such as model hyperparameters
from config.toml, and to provide the PROMPTS dictionary for system instructions.
"""
import os

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        pass  # Will fail gracefully if unavailable and file exists


CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.toml")


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
            "verbose": False
        }
    }

    if not os.path.exists(CONFIG_PATH):
        return default_config

    try:
        with open(CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    except NameError:
        # If tomllib wasn't imported successfully (e.g. Python < 3.11 without tomli)
        print("Warning: tomllib not found. Using default configuration.")
        return default_config


config = load_config()

PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "models", "prompts.toml")
with open(PROMPTS_PATH, "rb") as f:
    PROMPTS = tomllib.load(f)
