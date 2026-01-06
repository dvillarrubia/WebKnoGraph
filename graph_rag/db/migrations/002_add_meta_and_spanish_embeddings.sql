-- ============================================================================
-- Migration 002: Add meta_description and change to Spanish embeddings model
-- ============================================================================
-- Changes:
-- 1. Add meta_description column to rag_pages
-- 2. Change embedding dimension from 1024 to 768 (for hiiamsid/sentence_similarity_spanish_es)
-- 3. Update rag_chunks embedding dimension as well
-- ============================================================================

-- Add meta_description column
ALTER TABLE rag_pages ADD COLUMN IF NOT EXISTS meta_description TEXT;

-- Drop existing embedding indexes (they reference the old dimension)
DROP INDEX IF EXISTS idx_rag_pages_embedding;
DROP INDEX IF EXISTS idx_rag_chunks_embedding;

-- Change embedding column dimension for rag_pages
-- First, drop the column and recreate with new dimension
-- Note: This will DELETE existing embeddings - they need to be regenerated
ALTER TABLE rag_pages DROP COLUMN IF EXISTS embedding;
ALTER TABLE rag_pages ADD COLUMN embedding vector(768);

-- Same for rag_chunks
ALTER TABLE rag_chunks DROP COLUMN IF EXISTS embedding;
ALTER TABLE rag_chunks ADD COLUMN embedding vector(768);

-- Recreate HNSW indexes with new dimension
CREATE INDEX IF NOT EXISTS idx_rag_pages_embedding ON rag_pages
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding ON rag_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Add index on meta_description for text search
CREATE INDEX IF NOT EXISTS idx_rag_pages_meta_description ON rag_pages USING gin(to_tsvector('spanish', COALESCE(meta_description, '')));

-- ============================================================================
-- IMPORTANT: After running this migration, you MUST regenerate all embeddings!
-- Run: PYTHONPATH=. python -m graph_rag.scripts.regenerate_embeddings
-- ============================================================================
