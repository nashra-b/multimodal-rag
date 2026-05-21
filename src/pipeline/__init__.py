"""
pipeline
--------
End-to-end orchestration layer.
 
Exports:
    IngestPipeline — Orchestrates parsing → chunking → embedding → Pinecone upsert
    RAGChain       — LangChain RAG chain with memory and source citations
"""
 
from .ingest_pipeline import IngestPipeline  # noqa: F401
from .rag_chain        import RAGChain        # noqa: F401
 
__all__ = ["IngestPipeline", "RAGChain"]
 