"""
Reranking Service for Graph-RAG.
Uses cross-encoder models to rerank retrieved chunks for better relevance.
"""

from typing import Optional
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)


class RerankingService:
    """
    Cross-encoder based reranking service.

    Reranks retrieved chunks by computing query-document relevance scores
    using a cross-encoder model, which is more accurate than bi-encoder
    similarity but slower (hence used as a second stage).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        Initialize the reranking service.

        Args:
            model_name: Cross-encoder model to use. Recommended:
                - "BAAI/bge-reranker-v2-m3" (multilingual, best for Spanish)
                - "BAAI/bge-reranker-base" (lighter, still good)
                - "cross-encoder/ms-marco-MiniLM-L-6-v2" (fastest, English-focused)
            device: Device to run on ("cpu", "cuda", "mps"). Auto-detected if None.
            max_length: Maximum sequence length for the model.
        """
        self.model_name = model_name
        self.max_length = max_length
        self._model: Optional[CrossEncoder] = None
        self._device = device

    @property
    def model(self) -> CrossEncoder:
        """Lazy load the cross-encoder model."""
        if self._model is None:
            logger.info(f"Loading reranking model: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self._device,
            )
            logger.info(f"Reranking model loaded successfully")
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 10,
        content_field: str = "content",
        return_scores: bool = True,
    ) -> list[dict]:
        """
        Rerank chunks based on relevance to the query.

        Args:
            query: The search query.
            chunks: List of chunk dictionaries to rerank.
            top_k: Number of top results to return.
            content_field: Field name containing the text content.
            return_scores: Whether to add rerank_score to results.

        Returns:
            Reranked list of chunks (top_k items).
        """
        if not chunks:
            return []

        if len(chunks) <= top_k:
            # No need to rerank if we have fewer chunks than top_k
            if return_scores:
                for chunk in chunks:
                    chunk["rerank_score"] = chunk.get("similarity", 0.5)
            return chunks

        # Prepare query-document pairs for scoring
        pairs = []
        for chunk in chunks:
            content = chunk.get(content_field, "")
            # Include heading context for better matching
            heading = chunk.get("heading_context", "")
            if heading:
                content = f"{heading}\n{content}"
            pairs.append([query, content])

        # Get relevance scores from cross-encoder
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fall back to original order
            return chunks[:top_k]

        # Add scores and sort
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Return top_k with scores
        result = []
        for chunk, score in scored_chunks[:top_k]:
            if return_scores:
                chunk["rerank_score"] = float(score)
            result.append(chunk)

        return result

    def rerank_with_metadata(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 10,
        pagerank_weight: float = 0.1,
    ) -> list[dict]:
        """
        Rerank chunks combining cross-encoder scores with graph metadata.

        Args:
            query: The search query.
            chunks: List of chunk dictionaries to rerank.
            top_k: Number of top results to return.
            pagerank_weight: Weight for PageRank score in final ranking (0-1).

        Returns:
            Reranked list of chunks with combined scoring.
        """
        if not chunks:
            return []

        # First, get cross-encoder scores
        reranked = self.rerank(
            query=query,
            chunks=chunks,
            top_k=len(chunks),  # Get all scores first
            return_scores=True,
        )

        # Normalize scores for combination
        if reranked:
            max_rerank = max(c.get("rerank_score", 0) for c in reranked)
            min_rerank = min(c.get("rerank_score", 0) for c in reranked)
            rerank_range = max_rerank - min_rerank if max_rerank != min_rerank else 1

            for chunk in reranked:
                # Normalize rerank score to 0-1
                rerank_norm = (chunk.get("rerank_score", 0) - min_rerank) / rerank_range

                # Get PageRank (already 0-1 normalized)
                pagerank = chunk.get("pagerank", 0)

                # Combined score
                chunk["combined_score"] = (
                    (1 - pagerank_weight) * rerank_norm +
                    pagerank_weight * pagerank
                )

            # Sort by combined score
            reranked.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        return reranked[:top_k]
