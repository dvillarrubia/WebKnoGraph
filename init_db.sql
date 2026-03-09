CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS rag_clients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255) NOT NULL UNIQUE,
    api_key VARCHAR(64) NOT NULL UNIQUE DEFAULT encode(gen_random_bytes(32), 'hex'),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rag_clients_api_key ON rag_clients(api_key);
CREATE INDEX IF NOT EXISTS idx_rag_clients_domain ON rag_clients(domain);

CREATE TABLE IF NOT EXISTS rag_pages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL REFERENCES rag_clients(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT,
    content TEXT,
    content_hash VARCHAR(64),
    embedding vector(768),
    meta_description TEXT,
    pagerank FLOAT DEFAULT 0.0,
    hub_score FLOAT DEFAULT 0.0,
    authority_score FLOAT DEFAULT 0.0,
    folder_depth INTEGER DEFAULT 0,
    last_crawled_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_client_url UNIQUE (client_id, url)
);
CREATE INDEX IF NOT EXISTS idx_rag_pages_client_id ON rag_pages(client_id);
CREATE INDEX IF NOT EXISTS idx_rag_pages_url ON rag_pages(url);
CREATE INDEX IF NOT EXISTS idx_rag_pages_pagerank ON rag_pages(client_id, pagerank DESC);
CREATE INDEX IF NOT EXISTS idx_rag_pages_folder_depth ON rag_pages(client_id, folder_depth);
CREATE INDEX IF NOT EXISTS idx_rag_pages_embedding ON rag_pages USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_rag_pages_meta_description ON rag_pages USING gin(to_tsvector('spanish', COALESCE(meta_description, '')));

CREATE TABLE IF NOT EXISTS rag_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id UUID NOT NULL REFERENCES rag_pages(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT,
    heading_context TEXT,
    char_start INTEGER,
    char_end INTEGER,
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_page_chunk UNIQUE (page_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding ON rag_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS rag_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL REFERENCES rag_clients(id) ON DELETE CASCADE,
    source_page_id UUID NOT NULL REFERENCES rag_pages(id) ON DELETE CASCADE,
    target_page_id UUID NOT NULL REFERENCES rag_pages(id) ON DELETE CASCADE,
    anchor_text TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_link UNIQUE (client_id, source_page_id, target_page_id)
);
CREATE INDEX IF NOT EXISTS idx_rag_links_client_id ON rag_links(client_id);
CREATE INDEX IF NOT EXISTS idx_rag_links_source ON rag_links(source_page_id);
CREATE INDEX IF NOT EXISTS idx_rag_links_target ON rag_links(target_page_id);

CREATE TABLE IF NOT EXISTS rag_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL REFERENCES rag_clients(id) ON DELETE CASCADE,
    session_id VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rag_conversations_client_session ON rag_conversations(client_id, session_id);

CREATE TABLE IF NOT EXISTS rag_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES rag_conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    context_pages UUID[],
    tokens_used INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rag_messages_conversation ON rag_messages(conversation_id);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

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
