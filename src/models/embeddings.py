from src.config import config
from src.paths import MODELS_DIR
from sentence_transformers import SentenceTransformer, CrossEncoder

device = config["llm"].get("device", "cpu")
print(f"Loading embedding models on device: {device}")

_embedding_model = None
_reranker_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model on device: {device}...")
        _embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device=device, cache_folder=MODELS_DIR
        )
    return _embedding_model


def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        print(f"Loading reranker model on device: {device}...")
        _reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=device, cache_folder=MODELS_DIR
        )
    return _reranker_model
