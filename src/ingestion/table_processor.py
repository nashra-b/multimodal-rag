"""
table_processor.py
------------------
Converts raw Table elements from unstructured.io into structured,
embedding-ready text representations.

Two modes:
  1. HTML-rich mode  → uses text_as_html from unstructured metadata (preferred)
  2. Fallback mode   → re-parses with pdfplumber for higher-fidelity extraction

Each table is returned as a TableRecord dataclass with:
  - plain text (for embedding)
  - structured rows (for display / downstream processing)
  - rich metadata (page, bbox, source)
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Optional pdfplumber import (graceful fallback) ─────────────────────────────
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not installed. Falling back to unstructured HTML only.")


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class TableRecord:
    """A fully processed table ready for embedding and storage."""
    table_id:       str
    source_file:    str
    page_number:    Optional[int]
    headers:        list[str]
    rows:           list[list[str]]
    plain_text:     str               # embedding-ready flat text
    html:           Optional[str]     # original HTML if available
    bbox:           Optional[tuple]   # (x0, y0, x1, y1) bounding box
    has_merged_cells: bool = False
    extraction_method: str = "unstructured"

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def col_count(self) -> int:
        return len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0)

    def to_embedding_text(self) -> str:
        """
        Returns a compact, structured text that embeds well.
        Format:
            TABLE | Page {N} | {col_count} columns | {row_count} rows
            Headers: Col1 | Col2 | Col3
            Row 1: Val1 | Val2 | Val3
            ...
        """
        parts = [
            f"TABLE | Page {self.page_number or 'N/A'} | "
            f"{self.col_count} columns | {self.row_count} rows",
        ]
        if self.headers:
            parts.append("Headers: " + " | ".join(str(h) for h in self.headers))

        for i, row in enumerate(self.rows, 1):
            parts.append(f"Row {i}: " + " | ".join(str(c) for c in row))

        return "\n".join(parts)


# ── Main Processor ─────────────────────────────────────────────────────────────

class TableProcessor:
    """
    Processes unstructured.io Table elements into TableRecord objects.

    Usage:
        processor = TableProcessor(pdf_path="report.pdf")
        records   = processor.process(table_elements)
    """

    def __init__(self, pdf_path: Optional[str] = None):
        """
        Args:
            pdf_path: Path to the original PDF. When provided, enables
                      pdfplumber fallback for tables with no HTML metadata.
        """
        self.pdf_path   = pdf_path
        self._plumber   = None                  # lazy load

    def process(
        self,
        table_elements: list,
        source_file: str = "unknown.pdf",
    ) -> list[TableRecord]:
        """
        Convert a list of unstructured Table elements to TableRecords.

        Args:
            table_elements: List of unstructured Table objects.
            source_file:    Original PDF filename for metadata.

        Returns:
            List of TableRecord objects, filtered for non-empty tables.
        """
        records = []
        for idx, elem in enumerate(table_elements):
            table_id = f"{Path(source_file).stem}_table_{idx+1:03d}"
            try:
                record = self._process_element(elem, table_id, source_file, idx)
                if record and record.row_count > 0:
                    records.append(record)
                else:
                    logger.debug(f"Skipping empty table at index {idx}")
            except Exception as e:
                logger.error(f"Failed to process table {idx}: {e}", exc_info=True)

        logger.info(
            f"TableProcessor: {len(records)}/{len(table_elements)} "
            f"tables extracted successfully."
        )
        return records

    # ── Internal processing ────────────────────────────────────────────────────

    def _process_element(
        self,
        elem,
        table_id: str,
        source_file: str,
        idx: int,
    ) -> Optional[TableRecord]:
        """Route to the best available extraction method."""
        html     = getattr(getattr(elem, "metadata", None), "text_as_html", None)
        page_num = self._get_page(elem)
        bbox     = self._get_bbox(elem)

        if html:
            return self._from_html(html, table_id, source_file, page_num, bbox)

        # pdfplumber fallback — re-extract by page and position
        if PDFPLUMBER_AVAILABLE and self.pdf_path and page_num:
            plumber_rows = self._extract_via_pdfplumber(page_num, bbox)
            if plumber_rows:
                return self._from_rows(
                    plumber_rows, table_id, source_file,
                    page_num, bbox, method="pdfplumber"
                )

        # Last resort: plain text from element
        plain = str(elem).strip()
        if plain:
            return self._from_plain_text(plain, table_id, source_file, page_num, bbox)

        return None

    # ── HTML extraction ────────────────────────────────────────────────────────

    def _from_html(
        self,
        html: str,
        table_id: str,
        source_file: str,
        page_number: Optional[int],
        bbox: Optional[tuple],
    ) -> TableRecord:
        """Parse HTML <table> into headers + rows."""
        has_merged = bool(re.search(r'(?:colspan|rowspan)=["\']\d+["\']', html, re.IGNORECASE))

        # Extract rows: each <tr> becomes a list of cell values
        rows_raw   = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.IGNORECASE | re.DOTALL)
        all_rows   = [self._parse_cells(row) for row in rows_raw]
        all_rows   = [r for r in all_rows if any(c.strip() for c in r)]  # drop fully empty rows

        if not all_rows:
            return self._empty_record(table_id, source_file, page_number, html)

        # Heuristic: if first row looks like headers (short, no numbers), treat as header row
        headers, data_rows = self._detect_headers(all_rows)

        record = TableRecord(
            table_id=table_id,
            source_file=source_file,
            page_number=page_number,
            headers=headers,
            rows=data_rows,
            plain_text="",              # filled below
            html=html,
            bbox=bbox,
            has_merged_cells=has_merged,
            extraction_method="unstructured_html",
        )
        record.plain_text = record.to_embedding_text()
        return record

    def _parse_cells(self, row_html: str) -> list[str]:
        """Extract text content from all <td> and <th> cells in a row."""
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.IGNORECASE | re.DOTALL)
        return [self._strip_html(c).strip() for c in cells]

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML tags and normalise whitespace."""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;",  " ", text)
        text = re.sub(r"&amp;",   "&", text)
        text = re.sub(r"&lt;",    "<", text)
        text = re.sub(r"&gt;",    ">", text)
        text = re.sub(r"\s+",     " ", text)
        return text.strip()

    def _detect_headers(
        self, all_rows: list[list[str]]
    ) -> tuple[list[str], list[list[str]]]:
        """
        Heuristically split headers from data rows.
        A row is treated as a header if ≥ 70% of cells are non-numeric strings.
        """
        if not all_rows:
            return [], []

        first_row     = all_rows[0]
        numeric_cells = sum(1 for c in first_row if self._is_numeric(c))
        numeric_ratio = numeric_cells / max(len(first_row), 1)

        if numeric_ratio < 0.3:          # mostly text → header row
            return first_row, all_rows[1:]

        return [], all_rows              # no clear header

    @staticmethod
    def _is_numeric(value: str) -> bool:
        """Return True if value looks like a number (int, float, %, $, etc.)."""
        v = value.strip().lstrip("$€£¥%+-").replace(",", "").replace(".", "", 1)
        return v.isdigit()

    # ── pdfplumber fallback ────────────────────────────────────────────────────

    def _extract_via_pdfplumber(
        self,
        page_number: int,
        bbox: Optional[tuple],
    ) -> Optional[list[list[str]]]:
        """
        Use pdfplumber to extract table from a specific page (and optional bbox).
        Returns list of rows (each row is a list of cell strings).
        """
        try:
            if self._plumber is None:
                self._plumber = pdfplumber.open(self.pdf_path)

            page = self._plumber.pages[page_number - 1]   # pdfplumber is 0-indexed

            if bbox:
                cropped = page.within_bbox(bbox)
                tables  = cropped.extract_tables()
            else:
                tables  = page.extract_tables()

            if not tables:
                return None

            # Return largest table found (by cell count)
            return max(tables, key=lambda t: sum(len(r) for r in t))

        except Exception as e:
            logger.warning(f"pdfplumber extraction failed on page {page_number}: {e}")
            return None

    def _from_rows(
        self,
        rows: list[list[Optional[str]]],
        table_id: str,
        source_file: str,
        page_number: Optional[int],
        bbox: Optional[tuple],
        method: str = "pdfplumber",
    ) -> TableRecord:
        """Build a TableRecord from a 2D list of strings."""
        # Normalise None cells
        clean_rows = [
            [str(c).strip() if c is not None else "" for c in row]
            for row in rows
            if row and any(c for c in row if c)
        ]

        headers, data_rows = self._detect_headers(clean_rows)

        record = TableRecord(
            table_id=table_id,
            source_file=source_file,
            page_number=page_number,
            headers=headers,
            rows=data_rows,
            plain_text="",
            html=None,
            bbox=bbox,
            extraction_method=method,
        )
        record.plain_text = record.to_embedding_text()
        return record

    # ── Plain text fallback ────────────────────────────────────────────────────

    def _from_plain_text(
        self,
        text: str,
        table_id: str,
        source_file: str,
        page_number: Optional[int],
        bbox: Optional[tuple],
    ) -> TableRecord:
        """Last resort: treat the entire element text as a single-cell table."""
        logger.debug(f"Table {table_id}: using plain text fallback.")
        return TableRecord(
            table_id=table_id,
            source_file=source_file,
            page_number=page_number,
            headers=[],
            rows=[[text]],
            plain_text=f"TABLE | Page {page_number or 'N/A'}\n{text}",
            html=None,
            bbox=bbox,
            extraction_method="plain_text_fallback",
        )

    def _empty_record(
        self,
        table_id: str,
        source_file: str,
        page_number: Optional[int],
        html: Optional[str],
    ) -> TableRecord:
        return TableRecord(
            table_id=table_id,
            source_file=source_file,
            page_number=page_number,
            headers=[],
            rows=[],
            plain_text="",
            html=html,
            bbox=None,
        )

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_page(elem) -> Optional[int]:
        try:
            return elem.metadata.page_number
        except AttributeError:
            return None

    @staticmethod
    def _get_bbox(elem) -> Optional[tuple]:
        try:
            c = elem.metadata.coordinates
            if c and c.points:
                xs = [p[0] for p in c.points]
                ys = [p[1] for p in c.points]
                return (min(xs), min(ys), max(xs), max(ys))
        except AttributeError:
            pass
        return None

    def close(self):
        """Release pdfplumber file handle if open."""
        if self._plumber:
            self._plumber.close()
            self._plumber = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()