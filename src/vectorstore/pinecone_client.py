"""
pinecone_client.py
------------------
Manages Pinecone index lifecycle and namespaced upsert/query operations.

Three namespaces keep element types separate:
    pdf-text   → narrative text chunks
    pdf-tables → table chunks
    pdf-images → image summary chunks
"""

import os
import logging
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

NAMESPACES = {
    "text":  "pdf-text",
    "table": "pdf-tables",
    "image": "pdf-images",
}

DIMENSION = 3072   # text-embedding-3-large
METRIC    = "cosine"


class PineconeClient:
    """
    Wraps Pinecone index creation, upsert, query, and deletion.

    Usage:
        client = PineconeClient()
        client.upsert_chunks(vectors, element_type="text")
        results = client.query(embedding, element_types=["text", "table"])
    """

    def __init__(self):
        api_key    = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "multimodal-rag")
        cloud      = os.getenv("PINECONE_CLOUD",  "aws")
        region     = os.getenv("PINECONE_REGION", "us-east-1")

        if not api_key:
            raise ValueError("PINECONE_API_KEY not set in environment.")

        self.pc         = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index      = self._ensure_index(index_name, cloud, region)

        logger.info(f"PineconeClient ready | index={index_name} | region={region}")

    # ── Index lifecycle ────────────────────────────────────────────────────────

    def _ensure_index(self, name: str, cloud: str, region: str):
        """Create the index if it doesn't exist, then return it."""
        existing = [i.name for i in self.pc.list_indexes()]
        if name not in existing:
            logger.info(f"Creating Pinecone index '{name}' …")
            self.pc.create_index(
                name    = name,
                dimension = DIMENSION,
                metric  = METRIC,
                spec    = ServerlessSpec(cloud=cloud, region=region),
            )
            logger.info("Index created.")
        return self.pc.Index(name)

    def delete_index(self) -> None:
        """Delete the entire index. Used by --reset flag in ingest.py."""
        logger.warning(f"Deleting index '{self.index_name}' …")
        self.pc.delete_index(self.index_name)
        logger.info("Index deleted.")

    # ── Upsert ─────────────────────────────────────────────────────────────────

    def upsert_chunks(self, vectors: list[dict], element_type: str) -> None:
        """
        Upsert a list of vectors into the namespace for element_type.

        Each vector dict must have: id (str), values (list[float]), metadata (dict)
        Batches in groups of 100 — Pinecone's recommended upsert size.
        """
        namespace = NAMESPACES.get(element_type, "pdf-text")
        batch_size = 100

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            logger.debug(f"Upserted batch {i//batch_size + 1} → namespace '{namespace}'")

        logger.info(f"Upserted {len(vectors)} vectors → '{namespace}'")

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(
        self,
        embedding:     list[float],
        element_types: list[str] = None,
        top_k:         int = 5,
    ) -> list[dict]:
        """
        Query one or more namespaces and return merged results sorted by score.

        Args:
            embedding:     Query vector (list of floats).
            element_types: Which namespaces to search. None = all three.
            top_k:         Number of results per namespace.

        Returns:
            List of match dicts sorted by score descending.
        """
        if element_types is None:
            element_types = list(NAMESPACES.keys())

        all_matches = []
        for et in element_types:
            namespace = NAMESPACES.get(et)
            if not namespace:
                continue
            try:
                response = self.index.query(
                    vector           = embedding,
                    top_k            = top_k,
                    namespace        = namespace,
                    include_metadata = True,
                )
                all_matches.extend(response.matches)
            except Exception as e:
                logger.warning(f"Query failed for namespace '{namespace}': {e}")

        # Sort merged results by score descending
        all_matches.sort(key=lambda m: m.score, reverse=True)
        return all_matches[:top_k]

    # ── Stats ──────────────────────────────────────────────────────────────────

    def describe_stats(self) -> dict:
        """Return index stats including per-namespace vector counts."""
        stats = self.index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        return stats