"""
embedder.py
-----------
Production-grade embedding module using OpenAI's text-embedding-3-large.
Handles batching, retries, rate limiting, and token-safe truncation.
"""

import os
import time
import logging
import tiktoken
from typing import Optional
from openai import OpenAI, RateLimitError, APIError

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072          # full dimensionality for text-embedding-3-large
MAX_TOKENS_PER_INPUT = 8191          # model hard limit
BATCH_SIZE           = 100           # Pinecone upsert sweet spot
MAX_RETRIES          = 5
RETRY_BASE_DELAY     = 2.0           # seconds; exponential backoff base


class Embedder:
    """
    Wraps OpenAI embeddings with:
      - Automatic token-safe truncation
      - Batch processing with configurable batch size
      - Exponential backoff on rate limit / transient errors
      - Optional cost estimation before embedding
    """

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        dimensions: int = EMBEDDING_DIMENSIONS,
        batch_size: int = BATCH_SIZE,
    ):
        self.client     = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model      = model
        self.dimensions = dimensions
        self.batch_size = batch_size

        # tiktoken encoder for pre-flight token counting
        self.encoder = tiktoken.get_encoding("cl100k_base")

        logger.info(
            f"Embedder initialised | model={model} | "
            f"dimensions={dimensions} | batch_size={batch_size}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings. Returns a list of float vectors in the
        same order as the input. Handles batching and retries internally.

        Args:
            texts: List of raw text strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        if not texts:
            return []

        safe_texts  = [self._truncate(t) for t in texts]
        embeddings  = []

        total_batches = (len(safe_texts) + self.batch_size - 1) // self.batch_size
        for i in range(0, len(safe_texts), self.batch_size):
            batch     = safe_texts[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")

            batch_embeddings = self._embed_with_retry(batch)
            embeddings.extend(batch_embeddings)

        logger.info(f"Embedded {len(embeddings)} texts successfully.")
        return embeddings

    def embed_single(self, text: str) -> list[float]:
        """Convenience wrapper — embed a single string."""
        return self.embed_texts([text])[0]

    def estimate_cost(self, texts: list[str]) -> dict:
        """
        Estimate the token count and approximate cost before embedding.
        Useful for large ingestion jobs.

        Returns a dict with token_count, estimated_cost_usd.
        """
        token_count = sum(len(self.encoder.encode(t)) for t in texts)
        # text-embedding-3-large pricing: $0.13 / 1M tokens (as of 2025)
        cost_usd    = (token_count / 1_000_000) * 0.13
        return {
            "text_count":        len(texts),
            "token_count":       token_count,
            "estimated_cost_usd": round(cost_usd, 6),
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _truncate(self, text: str) -> str:
        """Truncate text to stay within the model's token limit."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= MAX_TOKENS_PER_INPUT:
            return text

        logger.warning(
            f"Text truncated from {len(tokens)} → {MAX_TOKENS_PER_INPUT} tokens."
        )
        return self.encoder.decode(tokens[:MAX_TOKENS_PER_INPUT])

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """
        Call the OpenAI embeddings API with exponential backoff.
        Handles RateLimitError and transient APIError.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.embeddings.create(
                    model      = self.model,
                    input      = texts,
                    dimensions = self.dimensions,
                )
                # sort by index to guarantee order matches input
                sorted_data = sorted(response.data, key=lambda d: d.index)
                return [d.embedding for d in sorted_data]

            except RateLimitError as e:
                wait = RETRY_BASE_DELAY ** attempt
                logger.warning(f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                               f"Retrying in {wait:.1f}s… | {e}")
                time.sleep(wait)

            except APIError as e:
                wait = RETRY_BASE_DELAY ** attempt
                logger.warning(f"OpenAI API error (attempt {attempt}/{MAX_RETRIES}). "
                               f"Retrying in {wait:.1f}s… | {e}")
                time.sleep(wait)

        raise RuntimeError(
            f"Embedding failed after {MAX_RETRIES} retries. Check OpenAI status."
        )