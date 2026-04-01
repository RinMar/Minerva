"""
Centralized path definitions using platformdirs.
All app data (config, database, models) lives under a single application directory
so Minerva doesn't scatter files across the user's filesystem.
"""
import os
import sys
from platformdirs import user_data_dir

APP_DATA_DIR = user_data_dir("Minerva", "RinMar")
CONFIG_PATH = os.path.join(APP_DATA_DIR, "config.toml")
DB_PATH = os.path.join(APP_DATA_DIR, "memory.db")
MODELS_DIR = os.path.join(APP_DATA_DIR, "models")
print(f"APP_DATA_DIR: {APP_DATA_DIR}")

# Ensure directories exist at import time
os.makedirs(APP_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def get_resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # Not running as PyInstaller executable, use project root
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    return os.path.join(base_path, relative_path)
