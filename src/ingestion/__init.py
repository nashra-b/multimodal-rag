"""
ingestion
---------
PDF parsing and preprocessing pipeline.
 
Exports:
    PDFParser        — unstructured.io-based multimodal PDF parser
    ImageSummarizer  — GPT-4o Vision image → text summarizer
    TableProcessor   — HTML / pdfplumber table extractor
    Chunker          — Semantic chunker producing LangChain Documents
"""
 
from .pdf_parser       import PDFParser          # noqa: F401
from .image_summarizer import ImageSummarizer    # noqa: F401
from .table_processor  import TableProcessor     # noqa: F401
from .chunker          import Chunker, ChunkConfig  # noqa: F401
 
__all__ = [
    "PDFParser",
    "ImageSummarizer",
    "TableProcessor",
    "Chunker",
    "ChunkConfig",
]
 