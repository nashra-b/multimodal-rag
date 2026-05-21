# src/pipeline/__init__.py
from .rag_chain import RAGChain 

def get_ingest_pipeline(*args, **kwargs):
    from .ingest_pipeline import IngestPipeline
    return IngestPipeline(*args, **kwargs)

__all__ = ["RAGChain", "get_ingest_pipeline"]
