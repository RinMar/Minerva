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
        model_id = config["models"].get(
            "embedding_model_id", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print(f"Loading embedding model {model_id} on device: {device}...")
        _embedding_model = SentenceTransformer(
            model_id,
            device=device, cache_folder=MODELS_DIR
        )
    return _embedding_model


def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        model_id = config["models"].get("reranker_model_id", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        print(f"Loading reranker model {model_id} on device: {device}...")
        _reranker_model = CrossEncoder(
            model_id,
            device=device, cache_folder=MODELS_DIR
        )
    return _reranker_model
