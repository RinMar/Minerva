"""
Append-only raw message log.
Used to document the entire raw exchange history between user and assistant
for potential delayed background processing or summarization.
"""
import json
from pathlib import Path


def _log_path(user_name: str, memory_base_dir: str = "local_store") -> Path:
    return Path(memory_base_dir) / "users" / user_name / "raw_log.json"


def _load(path: Path) -> list:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save(path: Path, messages: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)


def append(user_msg: str, assistant_msg: str, user_name: str,
           memory_base_dir: str = "local_store"):
    """Append a user/assistant exchange to the raw log."""
    path = _log_path(user_name, memory_base_dir)
    messages = _load(path)
    messages.append({"role": "user", "content": user_msg})
    messages.append({"role": "assistant", "content": assistant_msg})
    _save(path, messages)
    _save(path, messages)
