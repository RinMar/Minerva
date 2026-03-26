"""
Base LLM wrapper.
Used to manage the underlying llama.cpp model lifecycle and provide thread-safe
text generation and streaming capabilities.
"""
import glob
import os
import threading
import time
import re

import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_url
from llama_cpp import Llama

from src.config import config
from src.paths import MODELS_DIR

MAX_RETRIES = 5
RETRY_BACKOFF = 5  # seconds, doubled each attempt


def _download_gguf(repo_id: str, filename: str) -> str:
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

            print(f"\n[download] Complete: {local_path}")
            return local_path

        except (requests.exceptions.RequestException, IOError) as e:
            wait = min(2 * attempt, 30)
            print(f"\n[download] Attempt {attempt} failed ({type(e).__name__}). Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed to download {repo_id}/{filename} after {max_attempts} attempts.")


def _resolve_gguf_path(repo_id: str, filename: str) -> str:
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

    return _download_gguf(repo_id, filename)


class CustomLLM:
    def __init__(
        self,
        repo_id=config["llm"]["repo_id"],
        filename=config["llm"]["filename"],
        n_ctx=config["llm"]["n_ctx"],
        n_gpu_layers=config["llm"]["n_gpu_layers"],
        n_batch=config["llm"]["n_batch"],
        use_mmap=config["llm"]["use_mmap"],
        use_mlock=config["llm"]["use_mlock"],
        verbose=config["llm"]["verbose"],
    ):
        self._lock = threading.Lock()

        model_path = _resolve_gguf_path(repo_id, filename)

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            verbose=verbose,
            chat_format="chatml",
        )

    def _generate_stream(self, messages, max_tokens):
        with self._lock:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    yield delta["content"]

    def generate(self, messages: list, max_tokens: int = 8192,
                 stream=False, strip_think=True, **kwargs):
        if stream:
            return self._generate_stream(messages, max_tokens)
        else:
            with self._lock:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                )
                content = response["choices"][0]["message"]["content"]
                if strip_think and "<think>" in content:
                    content = re.sub(
                        r"<think>.*?(?:</think>|$)", "",
                        content, flags=re.DOTALL
                    ).strip()
                return content
