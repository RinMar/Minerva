import time
import sys
import os


def profile():
    start_total = time.time()

    print("--- Profiling Minerva App Import Performance ---")
    sys.path.append(os.path.abspath("."))

    # Profile the actual import that happens in run.py
    s = time.time()
    from src.gui.main import MainWindow
    print(f"Import src.gui.main.MainWindow (Synchronous): {time.time() - s:.3f}s")

    s = time.time()
    from src.memory.db import init_db
    print(f"Import src.memory.db: {time.time() - s:.3f}s")

    s = time.time()
    from src.config import config
    print(f"Import src.config: {time.time() - s:.3f}s")

    print("\n--- Model Loading (Background Thread Work) ---")
    # This simulates what happens in the background thread
    from src.models.base_llm import CustomLLM
    from src.models.embeddings import get_embedding_model, get_reranker_model

    s = time.time()
    llm = CustomLLM()
    print(f"Initialize CustomLLM (Lazy Import): {time.time() - s:.3f}s")

    s = time.time()
    emb = get_embedding_model()
    print(f"Initialize Embedding Model (Lazy Import): {time.time() - s:.3f}s")

    s = time.time()
    rerank = get_reranker_model()
    print(f"Initialize Reranker Model (Lazy Import): {time.time() - s:.3f}s")

    print(f"\nTotal profiling time: {time.time() - start_total:.3f}s")


if __name__ == "__main__":
    profile()
