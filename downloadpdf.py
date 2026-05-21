#!/usr/bin/env python3
"""
scripts/download_pdfs.py
-------------------------
Find and download bank annual report PDFs using web search + scraping.

Usage:
    python scripts/download_pdfs.py
    python scripts/download_pdfs.py --output data/sample_pdfs/
    python scripts/download_pdfs.py --query "RBC 2023 annual report PDF"
"""

import re
import sys
import time
import logging
import argparse
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger("download_pdfs")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── Search queries — these find real current PDF links ────────────────────────
SEARCH_QUERIES = [
    "JPMorgan Chase 2023 annual report filetype:pdf",
    "Goldman Sachs 2023 annual report filetype:pdf",
    "Bank of America 2023 annual report filetype:pdf",
]

# ── Fallback: public financial PDFs that rarely move ─────────────────────────
FALLBACK_PDFS = [
    {
        "url":  "https://www.bis.org/publ/arpdf/ar2023e.pdf",
        "name": "bis_annual_report_2023.pdf",
        "label": "BIS Annual Report 2023",
    },
    {
        "url":  "https://www.fdic.gov/bank/statistical/guide/2023/index_2023.pdf",
        "name": "fdic_stats_guide_2023.pdf",
        "label": "FDIC Statistics Guide 2023",
    },
    {
        "url":  "https://www.imf.org/external/pubs/ft/ar/2023/downloads/imf-annual-report-2023-en.pdf",
        "name": "imf_annual_report_2023.pdf",
        "label": "IMF Annual Report 2023",
    },
    {
        "url":  "https://www.newyorkfed.org/medialibrary/media/research/epr/2023/epr_2023.pdf",
        "name": "ny_fed_economic_policy_review_2023.pdf",
        "label": "NY Fed Economic Policy Review 2023",
    },
    {
        "url":  "https://www.federalreserve.gov/publications/files/2023-report-economic-well-being-us-households-202305.pdf",
        "name": "fed_household_wellbeing_2023.pdf",
        "label": "Federal Reserve Household Report 2023",
    },
]


def download_pdf(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info(f"  Already exists ({size_mb:.1f} MB), skipping: {dest.name}")
        return True

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"  Attempt {attempt}/{retries}: {url}")
            r = requests.get(url, headers=HEADERS, timeout=40, stream=True)
            r.raise_for_status()

            content_type = r.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
                logger.warning(f"  Not a PDF (Content-Type: {content_type})")
                return False

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_mb = dest.stat().st_size / (1024 * 1024)
            if size_mb < 0.05:
                dest.unlink()
                logger.warning(f"  File too small ({size_mb:.2f} MB), likely an error page.")
                return False

            logger.info(f"  ✓ {dest.name} ({size_mb:.1f} MB)")
            return True

        except requests.exceptions.Timeout:
            logger.warning(f"  Timed out.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"  Failed: {e}")

        if attempt < retries:
            time.sleep(2 ** attempt)

    return False


def search_for_pdf(query: str) -> str | None:
    """
    Use DuckDuckGo HTML search (no API key needed) to find a direct PDF URL.
    Returns the first .pdf URL found in results, or None.
    """
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        r = requests.get(search_url, headers=HEADERS, timeout=15)
        r.raise_for_status()

        soup  = BeautifulSoup(r.text, "html.parser")
        links = soup.find_all("a", href=True)

        for link in links:
            href = link["href"]
            # DuckDuckGo wraps results — extract the real URL
            if "uddg=" in href:
                from urllib.parse import unquote, parse_qs, urlparse
                params = parse_qs(urlparse(href).query)
                href   = unquote(params.get("uddg", [""])[0])
            if href.lower().endswith(".pdf"):
                logger.info(f"  Found via search: {href}")
                return href

    except Exception as e:
        logger.warning(f"  Search failed for '{query}': {e}")
    return None


def safe_filename(url: str, label: str = "") -> str:
    if label:
        name = re.sub(r"[^\w\-]", "_", label.lower()) + ".pdf"
        return name
    name = Path(urlparse(url).path).name
    return re.sub(r"[^\w\-.]", "_", name) or "document.pdf"


def parse_args():
    parser = argparse.ArgumentParser(description="Download financial PDFs for RAG")
    parser.add_argument("--output", type=Path, default=Path("data/sample_pdfs"))
    parser.add_argument("--query",  type=str,  help="Custom search query for a specific PDF")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args.output.mkdir(parents=True, exist_ok=True)

    downloaded, failed = 0, 0

    # ── Custom query mode ──────────────────────────────────────────────────────
    if args.query:
        logger.info(f"Searching for: {args.query}")
        url = search_for_pdf(args.query)
        if url:
            dest = args.output / safe_filename(url)
            if download_pdf(url, dest):
                downloaded += 1
            else:
                failed += 1
        else:
            logger.error("No PDF found for that query.")
        print_summary(downloaded, failed, args.output)
        return

    # ── Step 1: Try fallback PDFs (public institutions, stable URLs) ───────────
    print("\n── Trying public institution PDFs (BIS, Fed, IMF) …")
    for entry in FALLBACK_PDFS:
        logger.info(f"  {entry['label']}")
        dest = args.output / entry["name"]
        if download_pdf(entry["url"], dest):
            downloaded += 1
        else:
            failed += 1
        time.sleep(1)

        if downloaded >= 2:          # 2 PDFs is enough for a demo
            break

    # ── Step 2: Search for bank PDFs if we still need more ────────────────────
    if downloaded < 2:
        print("\n── Searching for bank annual reports …")
        for query in SEARCH_QUERIES:
            logger.info(f"  Query: {query}")
            url = search_for_pdf(query)
            if url:
                dest = args.output / safe_filename(url, query.split()[0])
                if download_pdf(url, dest):
                    downloaded += 1
            time.sleep(2)
            if downloaded >= 2:
                break

    print_summary(downloaded, failed, args.output)


def print_summary(downloaded, failed, output_dir):
    files = list(output_dir.glob("*.pdf"))
    print(f"\n{'='*50}")
    print(f"  Downloaded : {downloaded} PDF(s)")
    print(f"  Failed     : {failed}")
    print(f"  In folder  : {len(files)} total PDF(s)")
    for f in files:
        print(f"    • {f.name} ({f.stat().st_size/1024/1024:.1f} MB)")
    print(f"  Location   : {output_dir.resolve()}")
    print(f"{'='*50}\n")
    if not files:
        print("  No PDFs found. Try:")
        print("  python scripts/download_pdfs.py --query 'Goldman Sachs 2023 annual report filetype:pdf'")
        sys.exit(1)


if __name__ == "__main__":
    main()