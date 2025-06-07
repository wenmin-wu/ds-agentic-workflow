-- Knowledge MCP Server Database Schema
-- PostgreSQL with pgvector extension for bge-m3:567m embeddings

-- Create extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table for storing knowledge
CREATE TABLE IF NOT EXISTS public.documents
(
    id bigint NOT NULL DEFAULT nextval('documents_id_seq'::regclass),
    title text COLLATE pg_catalog."default" NOT NULL,
    markdown text COLLATE pg_catalog."default" NOT NULL,
    doc_metadata json DEFAULT '{}',
    embedding vector(1024), -- bge-m3:567m dimension
    tags text[] DEFAULT '{}',
    created_at timestamp without time zone NOT NULL DEFAULT now(),
    updated_at timestamp without time zone NOT NULL DEFAULT now(),
    CONSTRAINT documents_pkey PRIMARY KEY (id)
);

-- Create sequence if it doesn't exist
CREATE SEQUENCE IF NOT EXISTS documents_id_seq
    INCREMENT 1
    START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1;

-- Indexes for performance

-- Time-based indexes
CREATE INDEX IF NOT EXISTS idx_documents_created_at
    ON public.documents USING btree
    (created_at DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS idx_documents_updated_at
    ON public.documents USING btree
    (updated_at DESC NULLS LAST);

-- Full-text search index on markdown content
CREATE INDEX IF NOT EXISTS idx_documents_content_fts
    ON public.documents USING gin
    (to_tsvector('english', title || ' ' || markdown));

-- Tags search index
CREATE INDEX IF NOT EXISTS idx_documents_tags
    ON public.documents USING gin
    (tags);

-- HNSW vector similarity index (optimized for bge-m3:567m)
CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
    ON public.documents USING hnsw
    (embedding vector_cosine_ops)
    WITH (m = 32, ef_construction = 128);

-- Functions for automatic updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust user as needed)
-- ALTER TABLE IF EXISTS public.documents OWNER to your_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.documents TO your_app_user; 