"""
embeddings
----------
Embedding generation utilities.
 
Exports:
    Embedder — Batched, retry-safe OpenAI embeddings wrapper
"""
 
from .embedder import Embedder  # noqa: F401
 
__all__ = ["Embedder"]
 