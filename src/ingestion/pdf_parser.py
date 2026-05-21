"""
pdf_parser.py
-------------
Cloud-compatible PDF parser using pdfplumber only.
No unstructured.io dependency — works on Streamlit Cloud.

For local use with full unstructured.io support, switch strategy to "unstructured".
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PDFParser:
    def __init__(self, image_output_dir: str = "./data/extracted_images", strategy: str = "fast"):
        self.image_output_dir = image_output_dir
        self.strategy = strategy
        Path(image_output_dir).mkdir(parents=True, exist_ok=True)

    def parse(self, pdf_path: str) -> dict:
        logger.info(f"Parsing PDF: {pdf_path}")
        try:
            return self._parse_with_pdfplumber(pdf_path)
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            return {"text_elements": [], "table_elements": [], "image_elements": [], "source_file": pdf_path}

    def _parse_with_pdfplumber(self, pdf_path: str) -> dict:
        import pdfplumber

        text_elements  = []
        table_elements = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # ── Extract text ───────────────────────────────────────────────
                text = page.extract_text()
                if text and text.strip():
                    text_elements.append(
                        _TextElement(text=text.strip(), page_number=page_num)
                    )

                # ── Extract tables ─────────────────────────────────────────────
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        table_elements.append(
                            _TableElement(rows=table, page_number=page_num)
                        )

        logger.info(
            f"Parsed: text={len(text_elements)} | "
            f"tables={len(table_elements)} | images=0"
        )
        return {
            "text_elements":  text_elements,
            "table_elements": table_elements,
            "image_elements": [],
            "source_file":    pdf_path,
        }


# ── Lightweight element classes (replaces unstructured elements) ───────────────

class _TextElement:
    def __init__(self, text: str, page_number: int):
        self._text       = text
        self.category    = "NarrativeText"
        self.metadata    = _Meta(page_number)

    def __str__(self):
        return self._text


class _TableElement:
    def __init__(self, rows: list, page_number: int):
        self._rows    = rows
        self.category = "Table"
        self.metadata = _Meta(page_number)

    def __str__(self):
        lines = []
        for row in self._rows:
            lines.append(" | ".join(str(c) if c else "" for c in row))
        return "\n".join(lines)


class _Meta:
    def __init__(self, page_number: int):
        self.page_number  = page_number
        self.text_as_html = None
