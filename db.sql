
-- -- Enable vector similarity search in PostgreSQL.
-- CREATE EXTENSION IF NOT EXISTS vector;

-- -- Store document chunks and their 384-dimensional embeddings.
-- CREATE TABLE IF NOT EXISTS document_chunks (
--     id BIGSERIAL PRIMARY KEY,
--     source TEXT NOT NULL,
--     content TEXT NOT NULL,
--     embedding VECTOR(384) NOT NULL,
--     created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
-- );

-- -- Supports filtering and future document-management features.
-- CREATE INDEX IF NOT EXISTS idx_document_chunks_source
--     ON document_chunks(source);

-- -- Improves cosine-similarity retrieval as the collection grows.
-- -- HNSW is approximate; tune it against retrieval-quality tests.
-- CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
--     ON document_chunks
--     USING hnsw (embedding vector_cosine_ops);
