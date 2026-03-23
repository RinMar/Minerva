import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading embedding models on device: {device}")

embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=device)
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
