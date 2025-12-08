-- ============================================================================
-- WebKnoGraph Graph-RAG: Multi-tenant Schema for Supabase + pgvector
-- ============================================================================

-- Enable pgvector extension (should already be enabled in Supabase)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- CLIENTS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS rag_clients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255) NOT NULL UNIQUE,
    api_key VARCHAR(64) NOT NULL UNIQUE DEFAULT encode(gen_random_bytes(32), 'hex'),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for API key lookups
CREATE INDEX IF NOT EXISTS idx_rag_clients_api_key ON rag_clients(api_key);
CREATE INDEX IF NOT EXISTS idx_rag_clients_domain ON rag_clients(domain);

-- ============================================================================
-- PAGES TABLE (with embeddings)
-- ============================================================================
CREATE TABLE IF NOT EXISTS rag_pages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL REFERENCES rag_clients(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT,
    content TEXT,
    content_hash VARCHAR(64),  -- SHA256 hash to detect content changes
    embedding vector(1024),     -- multilingual-e5-large dimension

    -- Graph metrics (from PageRank/HITS analysis)
    pagerank FLOAT DEFAULT 0.0,
    hub_score FLOAT DEFAULT 0.0,
    authority_score FLOAT DEFAULT 0.0,
    folder_depth INTEGER DEFAULT 0,

    -- Metadata
    last_crawled_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Unique constraint per client
    CONSTRAINT unique_client_url UNIQUE (client_id, url)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_rag_pages_client_id ON rag_pages(client_id);
CREATE INDEX IF NOT EXISTS idx_rag_pages_url ON rag_pages(url);
CREATE INDEX IF NOT EXISTS idx_rag_pages_pagerank ON rag_pages(client_id, pagerank DESC);
CREATE INDEX IF NOT EXISTS idx_rag_pages_folder_depth ON rag_pages(client_id, folder_depth);

-- HNSW index for vector similarity search (pgvector)
-- Using cosine distance for multilingual-e5 embeddings
CREATE INDEX IF NOT EXISTS idx_rag_pages_embedding ON rag_pages
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- LINKS TABLE (edge list for internal links)
-- ============================================================================
CREATE TABLE IF NOT EXISTS rag_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL REFERENCES rag_clients(id) ON DELETE CASCADE,
    source_page_id UUID NOT NULL REFERENCES rag_pages(id) ON DELETE CASCADE,
    target_page_id UUID NOT NULL REFERENCES rag_pages(id) ON DELETE CASCADE,
    anchor_text TEXT,  -- Text of the <a> tag
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Prevent duplicate links
    CONSTRAINT unique_link UNIQUE (client_id, source_page_id, target_page_id)
);

-- Indexes for graph traversal
CREATE INDEX IF NOT EXISTS idx_rag_links_client_id ON rag_links(client_id);
CREATE INDEX IF NOT EXISTS idx_rag_links_source ON rag_links(source_page_id);
CREATE INDEX IF NOT EXISTS idx_rag_links_target ON rag_links(target_page_id);

-- ============================================================================
-- CONVERSATIONS TABLE (for chat history)
-- ============================================================================
CREATE TABLE IF NOT EXISTS rag_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL REFERENCES rag_clients(id) ON DELETE CASCADE,
    session_id VARCHAR(64) NOT NULL,  -- User session identifier
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rag_conversations_client_session
ON rag_conversations(client_id, session_id);

-- ============================================================================
-- MESSAGES TABLE (conversation messages)
-- ============================================================================
CREATE TABLE IF NOT EXISTS rag_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES rag_conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,

    -- RAG context metadata
    context_pages UUID[],  -- Array of page IDs used as context
    tokens_used INTEGER,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rag_messages_conversation ON rag_messages(conversation_id);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to search pages by semantic similarity
CREATE OR REPLACE FUNCTION search_pages_by_similarity(
    p_client_id UUID,
    p_query_embedding vector(1024),
    p_limit INTEGER DEFAULT 10,
    p_min_pagerank FLOAT DEFAULT 0.0
)
RETURNS TABLE (
    id UUID,
    url TEXT,
    title TEXT,
    content TEXT,
    pagerank FLOAT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        rp.id,
        rp.url,
        rp.title,
        rp.content,
        rp.pagerank,
        1 - (rp.embedding <=> p_query_embedding) AS similarity
    FROM rag_pages rp
    WHERE rp.client_id = p_client_id
      AND rp.embedding IS NOT NULL
      AND rp.pagerank >= p_min_pagerank
    ORDER BY rp.embedding <=> p_query_embedding
    LIMIT p_limit;
END;
$$;

-- Function to update timestamps automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
DROP TRIGGER IF EXISTS trigger_rag_clients_updated_at ON rag_clients;
CREATE TRIGGER trigger_rag_clients_updated_at
    BEFORE UPDATE ON rag_clients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trigger_rag_pages_updated_at ON rag_pages;
CREATE TRIGGER trigger_rag_pages_updated_at
    BEFORE UPDATE ON rag_pages
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trigger_rag_conversations_updated_at ON rag_conversations;
CREATE TRIGGER trigger_rag_conversations_updated_at
    BEFORE UPDATE ON rag_conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) - Optional but recommended
-- ============================================================================

-- Enable RLS on tables
ALTER TABLE rag_pages ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_messages ENABLE ROW LEVEL SECURITY;

-- Policy: Service role can access all data
CREATE POLICY "Service role full access on rag_pages" ON rag_pages
    FOR ALL USING (true);

CREATE POLICY "Service role full access on rag_links" ON rag_links
    FOR ALL USING (true);

CREATE POLICY "Service role full access on rag_conversations" ON rag_conversations
    FOR ALL USING (true);

CREATE POLICY "Service role full access on rag_messages" ON rag_messages
    FOR ALL USING (true);
