"""
Append-only raw message log.

Stores every user/assistant exchange in raw_log.json for later
batch processing (summarization → fact extraction → graph).
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


def get_recent(user_name: str, n: int = 20,
               memory_base_dir: str = "local_store") -> list:
    """Return the last N messages for summarization."""
    path = _log_path(user_name, memory_base_dir)
    messages = _load(path)
    return messages[-n:]


def get_all_pairs(user_name: str, memory_base_dir: str = "local_store") -> list:
    """Return all undigested user/assistant message pairs in the queue."""
    path = _log_path(user_name, memory_base_dir)
    messages = _load(path)
    pairs = []
    for i in range(len(messages) - 1):
        if messages[i].get("role") == "user" and messages[i+1].get("role") == "assistant":
            pairs.append((messages[i]["content"], messages[i+1]["content"]))
    return pairs


def remove_first_pair(user_name: str, memory_base_dir: str = "local_store"):
    """Remove the oldest user/assistant pair from the queue after it has been digested."""
    path = _log_path(user_name, memory_base_dir)
    messages = _load(path)

    # Pop the first user+assistant combination if they exist
    if len(messages) >= 2 and messages[0].get("role") == "user" and messages[1].get("role") == "assistant":
        del messages[:2]

    _save(path, messages)
