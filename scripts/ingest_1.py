#!/usr/bin/env python3
"""
scripts/ingest.py
-----------------
CLI entry point for the full multimodal PDF ingestion pipeline.

Usage:
    python scripts/ingest.py --pdf data/sample_pdfs/report.pdf
    python scripts/ingest.py --pdf data/sample_pdfs/report.pdf --dry-run
    python scripts/ingest.py --dir data/sample_pdfs/

Options:
    --pdf       Path to a single PDF file
    --dir       Path to a directory of PDFs (processes all *.pdf files)
    --dry-run   Parse and chunk only; skip embedding and Pinecone upsert
    --reset     Delete existing Pinecone index before ingesting (careful!)
    --verbose   Enable DEBUG-level logging
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# ── Resolve project root so we can import src/ regardless of CWD ───────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion         import PDFParser, ImageSummarizer, TableProcessor, Chunker
from src.embeddings        import Embedder
from src.vectorstore       import PineconeClient
from src.pipeline          import IngestPipeline

load_dotenv()


# ── Logging setup ──────────────────────────────────────────────────────────────
def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(PROJECT_ROOT / "logs" / "ingest.log"),
        ]
    )
    return logging.getLogger("ingest")


# ── Argument parsing ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal RAG — PDF ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", type=Path, help="Single PDF file path")
    source.add_argument("--dir", type=Path, help="Directory of PDF files")

    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and chunk only; skip embedding and Pinecone upsert"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete and recreate the Pinecone index before ingesting"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging"
    )
    parser.add_argument(
        "--image-dir", type=Path,
        default=PROJECT_ROOT / "data" / "extracted_images",
        help="Directory to save extracted images (default: data/extracted_images)"
    )
    return parser.parse_args()


# ── Env validation ─────────────────────────────────────────────────────────────
def validate_env(dry_run: bool) -> None:
    required = ["OPENAI_API_KEY"]
    if not dry_run:
        required += ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"]

    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
        print("        Copy .env.example to .env and fill in the values.")
        sys.exit(1)


# ── PDF discovery ──────────────────────────────────────────────────────────────
def discover_pdfs(args: argparse.Namespace) -> list[Path]:
    if args.pdf:
        if not args.pdf.exists():
            print(f"[ERROR] File not found: {args.pdf}")
            sys.exit(1)
        return [args.pdf]

    if args.dir:
        if not args.dir.is_dir():
            print(f"[ERROR] Directory not found: {args.dir}")
            sys.exit(1)
        pdfs = sorted(args.dir.glob("*.pdf"))
        if not pdfs:
            print(f"[ERROR] No PDF files found in: {args.dir}")
            sys.exit(1)
        return pdfs

    return []


# ── Pretty summary printer ─────────────────────────────────────────────────────
def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("  INGESTION SUMMARY")
    print("=" * 60)

    total_text  = sum(r.get("text_chunks",  0) for r in results)
    total_table = sum(r.get("table_chunks", 0) for r in results)
    total_image = sum(r.get("image_chunks", 0) for r in results)
    total_time  = sum(r.get("elapsed_sec",  0) for r in results)

    for r in results:
        status = "✓" if r.get("success") else "✗"
        print(
            f"  {status} {Path(r['file']).name:<35} "
            f"T:{r.get('text_chunks',0):>4} "
            f"Tbl:{r.get('table_chunks',0):>3} "
            f"Img:{r.get('image_chunks',0):>3} "
            f"({r.get('elapsed_sec',0):.1f}s)"
        )
        if not r.get("success") and r.get("error"):
            print(f"    └─ ERROR: {r['error']}")

    print("-" * 60)
    print(
        f"  TOTAL  {len(results)} file(s) | "
        f"Text: {total_text} | Tables: {total_table} | Images: {total_image} | "
        f"Time: {total_time:.1f}s"
    )
    print("=" * 60 + "\n")


# ── Per-file ingestion ─────────────────────────────────────────────────────────
def ingest_file(
    pdf_path: Path,
    pipeline: "IngestPipeline",
    dry_run: bool,
    logger: logging.Logger,
) -> dict:
    """
    Run the full ingestion pipeline on a single PDF.
    Returns a result dict for the summary report.
    """
    result = {"file": str(pdf_path), "success": False}
    t0     = time.time()

    try:
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Processing: {pdf_path.name}")

        # ── Step 1: Parse ──────────────────────────────────────────────────────
        logger.info("  → Step 1/4: Parsing PDF with unstructured.io …")
        parsed = pipeline.parser.parse(str(pdf_path))

        text_count  = len(parsed["text_elements"])
        table_count = len(parsed["table_elements"])
        image_count = len(parsed["image_elements"])
        logger.info(
            f"     Parsed — Text: {text_count} | Tables: {table_count} | Images: {image_count}"
        )

        # ── Step 2: Summarise images ───────────────────────────────────────────
        logger.info("  → Step 2/4: Summarising images with GPT-4o Vision …")
        image_summaries = pipeline.image_summarizer.summarize_batch(
            parsed["image_elements"]
        )
        logger.info(f"     Summarised {len(image_summaries)} image(s).")
        parsed["image_elements"] = image_summaries

        # ── Step 3: Process tables ─────────────────────────────────────────────
        logger.info("  → Step 3/4: Processing tables …")
        with TableProcessor(pdf_path=str(pdf_path)) as tp:
            table_records = tp.process(
                parsed["table_elements"],
                source_file=pdf_path.name,
            )
        logger.info(f"     Extracted {len(table_records)} table(s).")

        # Convert TableRecords back to embedding-ready dicts for the chunker
        parsed["table_elements"] = table_records

        # ── Step 4: Chunk ──────────────────────────────────────────────────────
        logger.info("  → Step 4/4: Chunking …")
        chunks = pipeline.chunker.chunk_all(parsed, source_file=pdf_path.name)
        n_text  = len(chunks["text"])
        n_table = len(chunks["table"])
        n_image = len(chunks["image"])
        logger.info(f"     Chunks — Text: {n_text} | Tables: {n_table} | Images: {n_image}")

        result.update({
            "text_chunks":  n_text,
            "table_chunks": n_table,
            "image_chunks": n_image,
        })

        if dry_run:
            logger.info("  [DRY RUN] Skipping embedding and Pinecone upsert.")
        else:
            # ── Embed + Upsert ─────────────────────────────────────────────────
            cost_estimate = pipeline.embedder.estimate_cost(
                [d.page_content for d in chunks["text"] + chunks["table"] + chunks["image"]]
            )
            logger.info(
                f"  Embedding cost estimate: "
                f"{cost_estimate['token_count']:,} tokens ≈ "
                f"${cost_estimate['estimated_cost_usd']:.4f}"
            )

            pipeline.upsert_chunks(chunks, source_file=pdf_path.name)
            logger.info("  Upserted all chunks to Pinecone.")

        result["success"]     = True
        result["elapsed_sec"] = round(time.time() - t0, 2)

    except Exception as e:
        result["error"]       = str(e)
        result["elapsed_sec"] = round(time.time() - t0, 2)
        logger.error(f"Failed to ingest {pdf_path.name}: {e}", exc_info=True)

    return result


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args   = parse_args()

    # Create dirs before setup_logging opens the log file
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(args.verbose)

    validate_env(args.dry_run)

    logger.info("=" * 60)
    logger.info("  Multimodal RAG — Ingestion Pipeline")
    logger.info("=" * 60)

    # ── Build pipeline ─────────────────────────────────────────────────────────
    pipeline = IngestPipeline(
        image_output_dir=str(args.image_dir),
        dry_run=args.dry_run,
    )

    # ── Optional index reset ───────────────────────────────────────────────────
    if args.reset and not args.dry_run:
        logger.warning("--reset flag: deleting existing Pinecone index …")
        pipeline.pinecone_client.delete_index()
        logger.info("Index deleted. Will be recreated on first upsert.")

    # ── Discover and process PDFs ──────────────────────────────────────────────
    pdf_files = discover_pdfs(args)
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process.")

    results = []
    for pdf_path in pdf_files:
        result = ingest_file(pdf_path, pipeline, args.dry_run, logger)
        results.append(result)

    # ── Summary ────────────────────────────────────────────────────────────────
    print_summary(results)

    failed = [r for r in results if not r["success"]]
    if failed:
        logger.error(f"{len(failed)} file(s) failed. Check logs/ingest.log for details.")
        sys.exit(1)

    logger.info("Ingestion complete. ✓")


if __name__ == "__main__":