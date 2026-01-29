-- WhatsApp bot schema for Supabase Postgres
-- Run this in your Supabase SQL editor
--
-- Requires pgvector for embeddings.
-- In Supabase: Database -> Extensions -> enable "vector"

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS wbot_records (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    phone_number TEXT NOT NULL,
    message_sid TEXT,
    record_type TEXT NOT NULL DEFAULT 'media', -- media | note
    storage_urls JSONB DEFAULT '[]'::jsonb,    -- list of storage URLs when media
    ocr_text TEXT,                              -- raw OCR text (media)
    user_text TEXT,                             -- raw user note text (note)
    embedding vector(1536),                     -- OpenAI text-embedding-3-small
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_wbot_records_phone_number ON wbot_records(phone_number);
CREATE INDEX IF NOT EXISTS idx_wbot_records_created_at ON wbot_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_wbot_records_embedding ON wbot_records USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Enable Row Level Security (RLS) - optional, adjust based on your needs
ALTER TABLE wbot_records ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations (adjust based on your security requirements)
-- In production, you may want to restrict this based on authenticated users
CREATE POLICY "wbot_allow_all_operations" ON wbot_records
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- RPC: Match records by embedding (cosine similarity), scoped to a phone number.
-- Usage from Supabase client: rpc('wbot_match_records', {...})
CREATE OR REPLACE FUNCTION wbot_match_records(
    query_embedding vector(1536),
    match_count int DEFAULT 5,
    p_phone_number text DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    phone_number text,
    record_type text,
    storage_urls jsonb,
    ocr_text text,
    user_text text,
    metadata jsonb,
    created_at timestamptz,
    similarity float
)
LANGUAGE sql
AS $$
    SELECT
        r.id,
        r.phone_number,
        r.record_type,
        r.storage_urls,
        r.ocr_text,
        r.user_text,
        r.metadata,
        r.created_at,
        1 - (r.embedding <=> query_embedding) AS similarity
    FROM wbot_records r
    WHERE r.embedding IS NOT NULL
      AND (p_phone_number IS NULL OR r.phone_number = p_phone_number)
    ORDER BY r.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- Conversation messages per user/phone, to provide longitudinal context.
CREATE TABLE IF NOT EXISTS wbot_messages (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    phone_number TEXT NOT NULL,
    direction TEXT NOT NULL,          -- 'in' | 'out'
    role TEXT NOT NULL,               -- 'user' | 'assistant' | 'system'
    message_sid TEXT,                 -- Twilio MessageSid when available
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_wbot_messages_phone_number ON wbot_messages(phone_number);
CREATE INDEX IF NOT EXISTS idx_wbot_messages_created_at ON wbot_messages(created_at DESC);

ALTER TABLE wbot_messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "wbot_allow_all_operations_messages" ON wbot_messages
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- Background jobs for async processing of WhatsApp messages.
CREATE TABLE IF NOT EXISTS wbot_jobs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    phone_number TEXT NOT NULL,
    message_sid TEXT,
    job_type TEXT NOT NULL,           -- 'media' | 'text'
    payload JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued', -- 'queued' | 'processing' | 'completed' | 'failed'
    error TEXT,
    result JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_wbot_jobs_phone_number ON wbot_jobs(phone_number);
CREATE INDEX IF NOT EXISTS idx_wbot_jobs_created_at ON wbot_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_wbot_jobs_status ON wbot_jobs(status);

ALTER TABLE wbot_jobs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "wbot_allow_all_operations_jobs" ON wbot_jobs
    FOR ALL
    USING (true)
    WITH CHECK (true);
