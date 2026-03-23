"""
Base LLM wrapper.
Used to manage the underlying llama.cpp model lifecycle and provide thread-safe
text generation and streaming capabilities.
"""
import threading
from llama_cpp import Llama
import re
from src.config import config


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
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
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

    def generate(self, messages: list, max_tokens: int = 8192, stream=False, strip_think=True, **kwargs):
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

                    content = re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL).strip()
                return content
