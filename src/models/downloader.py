"""
Model downloading utilities.
Used to robustly fetch GGUF models from HuggingFace Hub and resolve local cache paths.
"""
import glob
import os
import time

import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_url

from src.paths import MODELS_DIR

MAX_RETRIES = 5
RETRY_BACKOFF = 5  # seconds, doubled each attempt


def download_gguf(repo_id: str, filename: str) -> str:
    """
    Robustly download a GGUF file from HuggingFace Hub using streaming requests.
    Supports resuming from partial downloads and handles network timeouts aggressively.
    """
    local_path = os.path.join(MODELS_DIR, filename)
    url = hf_hub_url(repo_id=repo_id, filename=filename)

    # Disable HF_TRANSFER as it's unreliable on Windows for the user
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    max_attempts = 10
    timeout_seconds = 15
    chunk_size = 1024 * 1024  # 1MB

    for attempt in range(1, max_attempts + 1):
        try:
            current_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0

            # Check if finished
            # We first get headers to check the total content length
            head = requests.head(url, allow_redirects=True, timeout=timeout_seconds)
            total_size = int(head.headers.get('content-length', 0))

            if current_size >= total_size and total_size > 0:
                print(f"[download] File already complete: {local_path}")
                return local_path

            headers = {}
            if current_size > 0:
                headers['Range'] = f"bytes={current_size}-"
                print(f"[download] Resuming from {current_size / (1024**3):.2f} GB...")
            else:
                print(f"[download] Starting new download for {filename}...")

            with requests.get(url, headers=headers, stream=True, timeout=timeout_seconds, allow_redirects=True) as r:
                r.raise_for_status()
                mode = 'ab' if current_size > 0 else 'wb'

                with open(local_path, mode) as f:
                    with tqdm(total=total_size, initial=current_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                current_size += len(chunk)
                                pbar.update(len(chunk))

            print(f"\\n[download] Complete: {local_path}")
            return local_path

        except (requests.exceptions.RequestException, IOError) as e:
            wait = min(2 * attempt, 30)
            print(f"\\n[download] Attempt {attempt} failed ({type(e).__name__}). Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed to download {repo_id}/{filename} after {max_attempts} attempts.")


def resolve_gguf_path(repo_id: str, filename: str) -> str:
    """
    Resolve the GGUF model path — download if not cached yet.
    Supports glob patterns in filename (e.g. '*Q4_K_M.gguf').
    """
    # If filename is a glob pattern, check for a local match first
    if "*" in filename or "?" in filename:
        pattern = os.path.join(MODELS_DIR, "**", filename)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            print(f"[download] Found cached model: {matches[0]}")
            return matches[0]

    return download_gguf(repo_id, filename)
