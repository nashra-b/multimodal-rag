# src/vectorstore/__init__.py
"""
vectorstore
-----------
Pinecone vector store management and retrieval.

Exports:
    PineconeClient   — Index management, namespaced upsert/query
    HybridRetriever  — Dense (MMR) + Sparse (BM25) ensemble retriever
"""

from .pinecone_client import PineconeClient   # noqa: F401
from .retriever       import HybridRetriever  # noqa: F401

__all__ = ["PineconeClient", "HybridRetriever"]