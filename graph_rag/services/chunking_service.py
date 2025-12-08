"""
Semantic Chunking Service for Graph-RAG.
Uses semchunk for intelligent text splitting that respects semantic boundaries.
"""

import re
from dataclasses import dataclass
from typing import Optional

import semchunk
import tiktoken


@dataclass
class Chunk:
    """Represents a semantic chunk of text."""
    index: int
    content: str
    heading_context: Optional[str]
    char_start: int
    char_end: int


class ChunkingService:
    """
    Service for splitting text into semantic chunks.
    Uses semchunk with tiktoken for accurate token counting.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        min_chunk_size: int = 50,
        model_name: str = "cl100k_base",  # GPT-4/text-embedding-3 tokenizer
    ):
        """
        Initialize the chunking service.

        Args:
            chunk_size: Target size for chunks in tokens
            min_chunk_size: Minimum chunk size (smaller chunks are merged)
            model_name: Tiktoken model name for tokenization
        """
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.tokenizer = tiktoken.get_encoding(model_name)

        # Create semchunk chunker with tiktoken token counter
        self.chunker = semchunk.chunkerify(
            self.tokenizer,
            chunk_size=chunk_size,
        )

    def _extract_heading_context(self, text: str, position: int) -> Optional[str]:
        """
        Extract the most recent heading before a given position.
        Looks for markdown headers (# ## ###) before the chunk.
        """
        text_before = text[:position]

        # Find all headers before this position
        header_pattern = r'^(#{1,3})\s+(.+?)$'
        headers = []

        for match in re.finditer(header_pattern, text_before, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            headers.append((match.start(), level, title))

        if not headers:
            return None

        # Build context from header hierarchy
        # Get the last header at each level
        context_parts = []
        last_by_level = {}

        for _, level, title in headers:
            last_by_level[level] = title
            # Reset lower levels when a higher level appears
            for l in list(last_by_level.keys()):
                if l > level:
                    del last_by_level[l]

        # Build context string from hierarchy
        for level in sorted(last_by_level.keys()):
            context_parts.append(last_by_level[level])

        if context_parts:
            return " > ".join(context_parts)
        return None

    def chunk_text(self, text: str, include_heading_context: bool = True) -> list[Chunk]:
        """
        Split text into semantic chunks.

        Args:
            text: The text to chunk (markdown format expected)
            include_heading_context: Whether to extract heading context for each chunk

        Returns:
            List of Chunk objects
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            if text and text.strip():
                return [Chunk(
                    index=0,
                    content=text.strip(),
                    heading_context=None,
                    char_start=0,
                    char_end=len(text),
                )]
            return []

        # Use semchunk to split the text
        chunk_texts = self.chunker(text)

        chunks = []
        current_pos = 0

        for i, chunk_text in enumerate(chunk_texts):
            # Skip empty chunks
            if not chunk_text.strip():
                continue

            # Find the position of this chunk in original text
            chunk_start = text.find(chunk_text, current_pos)
            if chunk_start == -1:
                # Fallback: try to find partial match
                chunk_start = current_pos

            chunk_end = chunk_start + len(chunk_text)

            # Extract heading context if requested
            heading_context = None
            if include_heading_context:
                heading_context = self._extract_heading_context(text, chunk_start)

            chunks.append(Chunk(
                index=len(chunks),
                content=chunk_text.strip(),
                heading_context=heading_context,
                char_start=chunk_start,
                char_end=chunk_end,
            ))

            current_pos = chunk_end

        # Merge very small chunks with previous
        merged_chunks = []
        for chunk in chunks:
            token_count = len(self.tokenizer.encode(chunk.content))

            if token_count < self.min_chunk_size and merged_chunks:
                # Merge with previous chunk
                prev = merged_chunks[-1]
                merged_chunks[-1] = Chunk(
                    index=prev.index,
                    content=prev.content + "\n\n" + chunk.content,
                    heading_context=prev.heading_context or chunk.heading_context,
                    char_start=prev.char_start,
                    char_end=chunk.char_end,
                )
            else:
                chunk.index = len(merged_chunks)
                merged_chunks.append(chunk)

        return merged_chunks

    def chunk_with_metadata(
        self,
        text: str,
        title: Optional[str] = None,
        url: Optional[str] = None,
    ) -> list[dict]:
        """
        Chunk text and return dictionaries with metadata.
        Useful for direct database insertion.

        Args:
            text: The text to chunk
            title: Optional page title to prepend to heading context
            url: Optional URL for reference

        Returns:
            List of dicts ready for database insertion
        """
        chunks = self.chunk_text(text)

        result = []
        for chunk in chunks:
            # Build full context including page title
            full_context = []
            if title:
                full_context.append(title)
            if chunk.heading_context:
                full_context.append(chunk.heading_context)

            result.append({
                "chunk_index": chunk.index,
                "content": chunk.content,
                "heading_context": " > ".join(full_context) if full_context else None,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "url": url,
            })

        return result

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.tokenizer.encode(text))
