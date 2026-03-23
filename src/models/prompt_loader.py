"""
Prompt loader module.
Provides the PROMPTS dictionary from the prompts.toml file.
"""

import tomllib
from pathlib import Path

_PROMPTS_FILE = Path(__file__).parent / "prompts.toml"
with open(_PROMPTS_FILE, "rb") as f:
    PROMPTS = tomllib.load(f)
