"""
src/embeddings.py
LangChain-compatible Embeddings using sentence-transformers locally.

Uses sentence-transformers/all-mpnet-base-v2 (768-dim, same as nomic-embed-text).
Runs on GPU if available (CUDA). No Ollama dependency for embeddings.

Also provides DirectOllamaEmbeddings as a fallback (single-text mode only).
"""

import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.embeddings import Embeddings


class LocalSentenceTransformerEmbeddings(Embeddings):
    """
    Embeddings via sentence-transformers running locally.
    Uses GPU automatically if CUDA is available.
    Model: all-mpnet-base-v2 (768-dim, high quality for physics text).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[EMBED] Loading {model_name} on {device.upper()}...")
        self.model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._device = device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents — batched, GPU-accelerated."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()


class DirectOllamaEmbeddings(Embeddings):
    """
    Fallback: Embeddings via Ollama REST API using requests directly.
    Sends ONE text at a time to avoid the batch tokenizer bug in Ollama 0.30.x.
    Slower than LocalSentenceTransformerEmbeddings — use as last resort.
    """

    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434"):
        import requests as _requests
        self.model = model
        self.host = host.rstrip("/")
        self._requests = _requests

    def _embed_one(self, text: str) -> List[float]:
        resp = self._requests.post(
            f"{self.host}/api/embed",
            json={"model": self.model, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


# ── Default export ────────────────────────────────────────────────────────────
# Use sentence-transformers locally (fast, GPU, no Ollama needed for embedding)
DefaultEmbeddings = LocalSentenceTransformerEmbeddings
