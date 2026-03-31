"""
Base LLM wrapper.
Used to manage the underlying llama.cpp model lifecycle and provide thread-safe
text generation and streaming capabilities.
"""
import threading
import re

# Llama is imported lazily in __init__

from src.config import config
from src.models.downloader import resolve_gguf_path


class CustomLLM:
    def __init__(
        self,
        repo_id=None,
        filename=None,
        n_ctx=None,
        n_gpu_layers=None,
        n_batch=None,
        use_mmap=None,
        use_mlock=None,
        verbose=None,
    ):
        self._lock = threading.Lock()

        # Fetch latest settings from config if not provided as overrides
        repo_id = repo_id or config["llm"]["repo_id"]
        filename = filename or config["llm"]["filename"]
        n_ctx = n_ctx or config["llm"]["n_ctx"]
        n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else config["llm"]["n_gpu_layers"]
        n_batch = n_batch or config["llm"]["n_batch"]
        use_mmap = use_mmap if use_mmap is not None else config["llm"]["use_mmap"]
        use_mlock = use_mlock if use_mlock is not None else config["llm"]["use_mlock"]
        verbose = verbose if verbose is not None else config["llm"]["verbose"]

        model_path = resolve_gguf_path(repo_id, filename)

        from llama_cpp import Llama
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
