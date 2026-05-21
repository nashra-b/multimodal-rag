"""
chunker.py
----------
Semantic chunking pipeline for multimodal PDF elements.

Strategy per element type:
  - Text / Title  → sentence-aware splitting with overlap
  - Table         → kept whole (never split mid-row); oversized tables split by row group
  - Image summary → kept whole (already concise from GPT-4o)

Each chunk is returned as a LangChain Document so it plugs directly
into any LangChain retriever or vectorstore.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────
@dataclass
class ChunkConfig:
    # Text chunking
    chunk_size: int          = 1000   # characters
    chunk_overlap: int       = 200    # characters — preserves cross-chunk context
    min_chunk_size: int      = 100    # drop noise fragments below this

    # Table chunking (rows per chunk when table is too large)
    max_table_chars: int     = 3000
    table_row_overlap: int   = 1      # repeat last N rows in next chunk for continuity

    # Separators tried in order for text splitting
    separators: list[str]    = field(default_factory=lambda: [
        "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""
    ])


class Chunker:
    """
    Converts parsed PDF elements (text, tables, image summaries)
    into overlapping, metadata-rich LangChain Documents.

    Usage:
        chunker  = Chunker()
        docs     = chunker.chunk_all(parsed_elements, source_file="report.pdf")
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.cfg = config or ChunkConfig()
        logger.info(
            f"Chunker initialised | chunk_size={self.cfg.chunk_size} | "
            f"overlap={self.cfg.chunk_overlap}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def chunk_all(
        self,
        parsed_elements: dict,
        source_file: str = "unknown.pdf",
    ) -> dict[str, list[Document]]:
        """
        Chunk all element types and return a dict keyed by element type.

        Args:
            parsed_elements: Output of PDFParser.parse() — keys:
                             text_elements, table_elements, image_elements
            source_file:     Original PDF filename (stored in metadata)

        Returns:
            {
                "text":   [Document, ...],
                "table":  [Document, ...],
                "image":  [Document, ...],
            }
        """
        text_docs  = self._chunk_text_elements(
            parsed_elements.get("text_elements", []), source_file
        )
        table_docs = self._chunk_table_elements(
            parsed_elements.get("table_elements", []), source_file
        )
        image_docs = self._chunk_image_elements(
            parsed_elements.get("image_elements", []), source_file
        )

        logger.info(
            f"Chunking complete | text={len(text_docs)} | "
            f"table={len(table_docs)} | image={len(image_docs)} chunks"
        )
        return {
            "text":  text_docs,
            "table": table_docs,
            "image": image_docs,
        }

    # ── Text chunking ──────────────────────────────────────────────────────────

    def _chunk_text_elements(
        self, elements: list, source_file: str
    ) -> list[Document]:
        """
        Splits narrative text / titles using a recursive separator strategy
        with character-level overlap.
        """
        docs = []
        for elem in elements:
            raw_text = str(elem).strip()
            if not raw_text:
                continue

            base_meta = {
                "source":       source_file,
                "element_type": "text",
                "category":     getattr(elem, "category", "NarrativeText"),
                "page_number":  self._get_page(elem),
            }

            chunks = self._recursive_split(raw_text)
            for idx, chunk in enumerate(chunks):
                if len(chunk) < self.cfg.min_chunk_size:
                    continue
                docs.append(Document(
                    page_content=chunk,
                    metadata={**base_meta, "chunk_index": idx, "total_chunks": len(chunks)},
                ))

        return docs

    def _recursive_split(self, text: str) -> list[str]:
        """
        Mimics LangChain's RecursiveCharacterTextSplitter but gives us
        full control over the logic without the extra import overhead.
        """
        # Try each separator in order
        for sep in self.cfg.separators:
            if sep == "" or sep in text:
                parts   = text.split(sep) if sep else list(text)
                chunks  = []
                current = ""

                for part in parts:
                    candidate = current + (sep if current else "") + part

                    if len(candidate) <= self.cfg.chunk_size:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current.strip())
                        # carry overlap into next chunk
                        overlap_text = self._get_overlap(current)
                        current      = overlap_text + (sep if overlap_text else "") + part

                if current.strip():
                    chunks.append(current.strip())

                # If any chunk is still too large, recurse with next separator
                final = []
                for c in chunks:
                    if len(c) > self.cfg.chunk_size and sep != self.cfg.separators[-1]:
                        next_sep_idx = self.cfg.separators.index(sep) + 1
                        sub_chunks   = self._recursive_split_from(c, next_sep_idx)
                        final.extend(sub_chunks)
                    else:
                        final.append(c)

                return [c for c in final if c.strip()]

        return [text]

    def _recursive_split_from(self, text: str, sep_idx: int) -> list[str]:
        """Recurse from a specific separator index onward."""
        remaining_seps = self.cfg.separators[sep_idx:]
        for sep in remaining_seps:
            if sep == "" or sep in text:
                result = self._recursive_split(text)
                return result
        return [text]

    def _get_overlap(self, text: str) -> str:
        """Return the last `chunk_overlap` characters of a chunk."""
        if len(text) <= self.cfg.chunk_overlap:
            return text
        # Try to break at a word boundary
        overlap_raw = text[-self.cfg.chunk_overlap:]
        space_idx   = overlap_raw.find(" ")
        return overlap_raw[space_idx + 1:] if space_idx != -1 else overlap_raw

    # ── Table chunking ─────────────────────────────────────────────────────────

    def _chunk_table_elements(
        self, elements: list, source_file: str
    ) -> list[Document]:
        """
        Tables are converted to a readable text format and kept whole when
        possible. Oversized tables are split by row groups with 1-row overlap.
        """
        docs = []
        for elem in elements:
            table_text = self._table_to_text(elem)
            if not table_text.strip():
                continue

            base_meta = {
                "source":       source_file,
                "element_type": "table",
                "category":     "Table",
                "page_number":  self._get_page(elem),
            }

            if len(table_text) <= self.cfg.max_table_chars:
                # Table fits in one chunk
                docs.append(Document(
                    page_content=table_text,
                    metadata={**base_meta, "chunk_index": 0, "total_chunks": 1},
                ))
            else:
                # Split oversized table by rows
                row_chunks = self._split_table_by_rows(table_text, base_meta)
                docs.extend(row_chunks)

        return docs

    def _table_to_text(self, elem) -> str:
        """
        Convert an unstructured Table element to a readable text block.
        Prefers HTML representation (richer) then falls back to plain text.
        """
        # unstructured stores HTML in metadata.text_as_html when available
        html = getattr(getattr(elem, "metadata", None), "text_as_html", None)
        if html:
            return self._html_table_to_plain(html)

        return str(elem).strip()

    def _html_table_to_plain(self, html: str) -> str:
        """
        Convert simple HTML table to pipe-delimited plain text.
        Avoids a full HTML parser dependency for this simple case.
        """
        # Strip all tags except row/cell markers
        html = re.sub(r"<tr[^>]*>",  "\n",  html, flags=re.IGNORECASE)
        html = re.sub(r"<th[^>]*>",  " | ", html, flags=re.IGNORECASE)
        html = re.sub(r"<td[^>]*>",  " | ", html, flags=re.IGNORECASE)
        html = re.sub(r"</t[dh]>",   "",    html, flags=re.IGNORECASE)
        html = re.sub(r"<[^>]+>",    "",    html)               # strip remaining tags
        html = re.sub(r"[ \t]+",     " ",   html)               # collapse whitespace
        html = re.sub(r"\n{3,}",     "\n\n", html)              # collapse blank lines
        return html.strip()

    def _split_table_by_rows(
        self, table_text: str, base_meta: dict
    ) -> list[Document]:
        """Split an oversized table into row-group chunks with overlap."""
        rows   = [r for r in table_text.split("\n") if r.strip()]
        chunks = []
        start  = 0

        while start < len(rows):
            group   = []
            chars   = 0
            idx     = start

            while idx < len(rows) and chars + len(rows[idx]) <= self.cfg.max_table_chars:
                group.append(rows[idx])
                chars += len(rows[idx])
                idx   += 1

            if not group:               # single row is too large — take it anyway
                group = [rows[idx]]
                idx  += 1

            content = "\n".join(group)
            chunks.append(Document(
                page_content=content,
                metadata={
                    **base_meta,
                    "chunk_index":  len(chunks),
                    "row_start":    start,
                    "row_end":      idx - 1,
                },
            ))

            # Move back by overlap rows for continuity
            start = max(idx - self.cfg.table_row_overlap, idx)

        # Stamp total_chunks now that we know it
        for doc in chunks:
            doc.metadata["total_chunks"] = len(chunks)

        return chunks

    # ── Image chunking ─────────────────────────────────────────────────────────

    def _chunk_image_elements(
        self, elements: list, source_file: str
    ) -> list[Document]:
        """
        Image elements arrive as (summary_text, image_path, page_number) tuples
        produced by ImageSummarizer. Each summary becomes a single Document.
        """
        docs = []
        for idx, elem in enumerate(elements):
            if isinstance(elem, dict):
                summary    = elem.get("summary", "")
                image_path = elem.get("image_path", "")
                page_num   = elem.get("page_number") or 0
            else:
                # Fallback: treat as plain text summary
                summary    = str(elem).strip()
                image_path = ""
                page_num   = self._get_page(elem)

            if not summary.strip():
                continue

            docs.append(Document(
                page_content=summary,
                metadata={
                    "source":        source_file,
                    "element_type":  "image",
                    "category":      "Image",
                    "image_path":    image_path,
                    "page_number":   page_num,
                    "chunk_index":   0,
                    "total_chunks":  1,
                },
            ))

        return docs

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_page(elem) -> int:
        """Safely extract page number from an unstructured element."""
        try:
            return elem.metadata.page_number or 0
        except AttributeError:
            return 0