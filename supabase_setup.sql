-- ═══════════════════════════════════════════════════════════════════
-- Supabase Setup SQL — run this in the Supabase SQL Editor
-- Project: HealthAI Explainer
-- ═══════════════════════════════════════════════════════════════════

-- 1. Enable UUID extension (usually already enabled)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Create the predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id           uuid        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id      uuid        NOT NULL,
    disease      text        NOT NULL,
    prediction   text        NOT NULL,
    key_reasons  text,
    input_values jsonb,
    created_at   timestamptz DEFAULT now()
);

-- 3. Enable Row-Level Security
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- 4. Policy: users can only INSERT their own rows
CREATE POLICY "insert_own_predictions"
    ON predictions FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- 5. Policy: users can only SELECT their own rows
CREATE POLICY "select_own_predictions"
    ON predictions FOR SELECT
    USING (auth.uid() = user_id);

-- 6. Policy: users can only DELETE their own rows
CREATE POLICY "delete_own_predictions"
    ON predictions FOR DELETE
    USING (auth.uid() = user_id);

-- 7. Verify
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public' AND table_name = 'predictions';
