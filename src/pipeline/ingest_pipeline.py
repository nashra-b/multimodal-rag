"""
ingest_pipeline.py
------------------
End-to-end ingestion orchestrator.

Wires together:
    PDFParser → ImageSummarizer → TableProcessor → Chunker
    → Embedder → PineconeClient (namespaced upsert)

Usage:
    pipeline = IngestPipeline()
    pipeline.run("data/sample_pdfs/annual_report.pdf")
"""

import os
import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from src.ingestion   import PDFParser, ImageSummarizer, TableProcessor, Chunker, ChunkConfig
from src.embeddings  import Embedder
from src.vectorstore import PineconeClient

logger = logging.getLogger(__name__)


class IngestPipeline:
    """
    Orchestrates the full PDF → Pinecone ingestion flow.

    Args:
        image_output_dir : Where extracted images are saved
        chunk_config     : Optional custom ChunkConfig
        dry_run          : If True, skip embedding and Pinecone upsert
    """

    def __init__(
        self,
        image_output_dir: str = "./data/extracted_images",
        chunk_config: Optional[ChunkConfig] = None,
        dry_run: bool = False,
    ):
        self.dry_run          = dry_run
        self.image_output_dir = image_output_dir

        # ── Ingestion components ───────────────────────────────────────────────
        self.parser           = PDFParser(image_output_dir=image_output_dir)
        self.image_summarizer = ImageSummarizer()
        self.chunker          = Chunker(config=chunk_config)
        self.embedder         = Embedder()

        # ── Vector store (only needed for real runs) ───────────────────────────
        self.pinecone_client  = PineconeClient() if not dry_run else None

        logger.info(f"IngestPipeline ready | dry_run={dry_run}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, pdf_path: str) -> dict:
        """
        Full pipeline for a single PDF file.

        Returns a summary dict:
            {
                "source_file"  : str,
                "text_chunks"  : int,
                "table_chunks" : int,
                "image_chunks" : int,
                "total_vectors": int,
            }
        """
        source_file = Path(pdf_path).name
        logger.info(f"IngestPipeline.run() → {source_file}")

        # Step 1 — Parse
        parsed = self._step_parse(pdf_path)

        # Step 2 — Summarise images
        parsed["image_elements"] = self._step_summarise_images(
            parsed["image_elements"]
        )

        # Step 3 — Process tables
        parsed["table_elements"] = self._step_process_tables(
            parsed["table_elements"], pdf_path, source_file
        )

        # Step 4 — Chunk
        chunks = self._step_chunk(parsed, source_file)

        # Step 5 — Embed + Upsert
        if not self.dry_run:
            self.upsert_chunks(chunks, source_file)

        summary = {
            "source_file":   source_file,
            "text_chunks":   len(chunks["text"]),
            "table_chunks":  len(chunks["table"]),
            "image_chunks":  len(chunks["image"]),
            "total_vectors": len(chunks["text"]) + len(chunks["table"]) + len(chunks["image"]),
        }
        logger.info(f"Pipeline complete: {summary}")
        return summary

    def upsert_chunks(self, chunks: dict[str, list[Document]], source_file: str) -> None:
        """
        Embed each element-type bucket and upsert into its Pinecone namespace.
        Called by run() but also exposed so scripts/ingest.py can call it directly.
        """
        for element_type, docs in chunks.items():
            if not docs:
                continue

            texts = [d.page_content for d in docs]

            # Cost log before embedding
            estimate = self.embedder.estimate_cost(texts)
            logger.info(
                f"  [{element_type}] {len(texts)} chunks | "
                f"~{estimate['token_count']:,} tokens | "
                f"~${estimate['estimated_cost_usd']:.4f}"
            )

            embeddings = self.embedder.embed_texts(texts)

            # Build Pinecone vectors: [{id, values, metadata}]
            vectors = [
                {
                    "id":       self._make_vector_id(source_file, element_type, i),
                    "values":   emb,
                    "metadata": {
                        **docs[i].metadata,
                        "text": texts[i],       # store text for retrieval display
                    },
                }
                for i, emb in enumerate(embeddings)
            ]

            self.pinecone_client.upsert_chunks(vectors, element_type=element_type)
            logger.info(f"  [{element_type}] Upserted {len(vectors)} vectors to Pinecone.")

    # ── Pipeline steps ─────────────────────────────────────────────────────────

    def _step_parse(self, pdf_path: str) -> dict:
        logger.info("Step 1/4 — Parsing PDF …")
        parsed = self.parser.parse(pdf_path)
        logger.info(
            f"  Parsed: text={len(parsed['text_elements'])} | "
            f"tables={len(parsed['table_elements'])} | "
            f"images={len(parsed['image_elements'])}"
        )
        return parsed

    def _step_summarise_images(self, image_elements: list) -> list:
        logger.info(f"Step 2/4 — Summarising {len(image_elements)} image(s) …")
        if not image_elements:
            return []
        summaries = self.image_summarizer.summarize_batch(image_elements)
        logger.info(f"  Summarised {len(summaries)} image(s).")
        return summaries

    def _step_process_tables(
        self, table_elements: list, pdf_path: str, source_file: str
    ) -> list:
        logger.info(f"Step 3/4 — Processing {len(table_elements)} table(s) …")
        if not table_elements:
            return []
        with TableProcessor(pdf_path=pdf_path) as tp:
            records = tp.process(table_elements, source_file=source_file)
        logger.info(f"  Extracted {len(records)} valid table(s).")
        return records

    def _step_chunk(self, parsed: dict, source_file: str) -> dict[str, list[Document]]:
        logger.info("Step 4/4 — Chunking …")
        chunks = self.chunker.chunk_all(parsed, source_file=source_file)
        logger.info(
            f"  Chunks: text={len(chunks['text'])} | "
            f"table={len(chunks['table'])} | "
            f"image={len(chunks['image'])}"
        )
        return chunks

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_vector_id(source_file: str, element_type: str, index: int) -> str:
        """
        Deterministic vector ID so re-ingesting the same file upserts (updates)
        existing records instead of creating duplicates.
        Format: <stem>_<type>_<index>
        """
        stem = Path(source_file).stem[:40].replace(" ", "_")
        return f"{stem}_{element_type}_{index:04d}"