"""
Embedding Service for Graph-RAG.
Uses hiiamsid/sentence_similarity_spanish_es for Spanish content.
"""

import torch
from typing import Optional
from sentence_transformers import SentenceTransformer

from graph_rag.config.settings import Settings


class EmbeddingService:
    """
    Generates embeddings using sentence-transformers.

    Default model: hiiamsid/sentence_similarity_spanish_es
    - Optimized for Spanish (castellano de España)
    - 768 dimensions
    - Based on BERT Spanish, trained on Spanish STS datasets

    Alternative: intfloat/multilingual-e5-large (1024 dims, multilingual)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.embedding_model_name
        self.dimension = settings.embedding_dimension
        self.batch_size = settings.embedding_batch_size
        self._model: Optional[SentenceTransformer] = None
        self._device: Optional[str] = None

        # Check if model requires prefixes (e5 models do)
        self._use_prefixes = "e5" in self.model_name.lower()

    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(
                self.model_name,
                device=self._device,
                trust_remote_code=True,
            )

    @property
    def model(self) -> SentenceTransformer:
        """Get the loaded model."""
        self._load_model()
        return self._model

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.
        """
        # Add prefix only for e5 models
        text = f"query: {query}" if self._use_prefixes else query
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # Normalize for better cosine similarity
        )
        return embedding.tolist()

    def embed_document(self, text: str) -> list[float]:
        """
        Generate embedding for a document/passage.
        """
        # Add prefix only for e5 models
        prefixed_text = f"passage: {text}" if self._use_prefixes else text
        embedding = self.model.encode(
            prefixed_text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # Normalize for better cosine similarity
        )
        return embedding.tolist()

    def embed_documents_batch(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        More efficient for bulk operations.
        """
        # Add prefix only for e5 models
        if self._use_prefixes:
            processed_texts = [f"passage: {text}" for text in texts]
        else:
            processed_texts = texts

        embeddings = self.model.encode(
            processed_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Normalize for better cosine similarity
        )
        return [emb.tolist() for emb in embeddings]

    def compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        import numpy as np

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


# Alternative: OpenAI embeddings (if you want to use OpenAI instead)
class OpenAIEmbeddingService:
    """
    Optional: Generate embeddings using OpenAI's API.
    Use this if you prefer cloud-based embeddings.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a query."""
        client = self._get_client()
        response = client.embeddings.create(
            input=query,
            model=self.settings.openai_embedding_model,
        )
        return response.data[0].embedding

    def embed_document(self, text: str) -> list[float]:
        """Generate embedding for a document."""
        return self.embed_query(text)  # Same for OpenAI

    def embed_documents_batch(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Generate embeddings for multiple documents."""
        client = self._get_client()

        # OpenAI supports batch in single call
        response = client.embeddings.create(
            input=texts,
            model=self.settings.openai_embedding_model,
        )
        return [item.embedding for item in response.data]
