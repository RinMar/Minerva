from src.config import config
from src.paths import MODELS_DIR
# Heavy imports moved to functions to speed up startup


def get_device():
    import torch
    device = config["llm"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


_embedding_model = None
_reranker_model = None


def reset_models():
    """Clear cached model instances to trigger re-loading with new settings."""
    global _embedding_model, _reranker_model
    _embedding_model = None
    _reranker_model = None
    print("[Models] Cached embedding and reranker models cleared.")


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        model_id = config["models"].get(
            "embedding_model_id", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model {model_id} on device: {get_device()}...")
        _embedding_model = SentenceTransformer(
            model_id,
            device=get_device(), cache_folder=MODELS_DIR
        )
    return _embedding_model


def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        model_id = config["models"].get("reranker_model_id", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        from sentence_transformers import CrossEncoder
        print(f"Loading reranker model {model_id} on device: {get_device()}...")
        _reranker_model = CrossEncoder(
            model_id,
            device=get_device(), cache_folder=MODELS_DIR
        )
    return _reranker_model
