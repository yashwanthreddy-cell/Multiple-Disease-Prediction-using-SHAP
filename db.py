"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  db.py — Supabase Database Module                                            ║
║  Handles: client init, save_history, load_history, clear_history             ║
║  Table  : predictions (user-scoped rows)                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Supabase table DDL (run once in Supabase SQL Editor):

    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    CREATE TABLE predictions (
        id           uuid        PRIMARY KEY DEFAULT uuid_generate_v4(),
        user_id      uuid        NOT NULL,
        disease      text        NOT NULL,
        prediction   text        NOT NULL,
        key_reasons  text,
        input_values jsonb,
        created_at   timestamptz DEFAULT now()
    );

    -- Row-Level Security: each user sees only their own rows
    ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

    CREATE POLICY "Users can insert own predictions"
        ON predictions FOR INSERT
        WITH CHECK (auth.uid() = user_id);

    CREATE POLICY "Users can select own predictions"
        ON predictions FOR SELECT
        USING (auth.uid() = user_id);

    CREATE POLICY "Users can delete own predictions"
        ON predictions FOR DELETE
        USING (auth.uid() = user_id);
"""

import os
import json
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Load environment variables (.env file or Streamlit secrets)
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()


def _get_credentials() -> tuple[str, str]:
    """
    Resolve Supabase URL and ANON KEY from (in priority order):
      1. Streamlit secrets  (st.secrets — for Streamlit Cloud deployment)
      2. Environment variables / .env file  (for local development)
    """

    url = os.getenv("SUPABASE_URL", "https://jiucaokohscxvyvkakxi.supabase.co")
    key = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImppdWNhb2tvaHNjeHZ5dmtha3hpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzUyNzQ4MTAsImV4cCI6MjA5MDg1MDgxMH0.Fe9wCZTGnHFfRponhkpXOrBJ-LZw5NhtB8DW7glnGRU")

    if not url or not key:
        raise EnvironmentError(
            "Supabase credentials not found.\n"
            "Set SUPABASE_URL and SUPABASE_ANON_KEY in:\n"
            "  • .env file  (local dev)\n"
            "  • .streamlit/secrets.toml  (Streamlit Cloud)"
        )
    return url, key


@st.cache_resource(show_spinner=False)
def get_supabase_client() -> Client:
    """
    Return a cached Supabase client.
    Cached with st.cache_resource so a single connection is reused
    across Streamlit reruns.
    """
    url, key = _get_credentials()
    return create_client(url, key)


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY CRUD — replaces the old JSON-based functions in app.py
# ─────────────────────────────────────────────────────────────────────────────

def save_history(record: dict, user_id: str) -> None:
    """
    Insert one prediction record into the Supabase `predictions` table.

    Args:
        record  : dict built by build_history_record() in app.py
        user_id : UUID string of the currently logged-in user

    The function is intentionally silent on failure so it never
    interrupts the main prediction flow.
    """
    if not user_id:
        return

    try:
        client = get_supabase_client()
        row = {
            "user_id"     : user_id,
            "disease"     : record.get("disease",      ""),
            "prediction"  : record.get("prediction",   ""),
            "key_reasons" : record.get("key_reasons",  ""),
            # input_values is a dict → store as JSONB
            "input_values": json.dumps(record.get("input_values", {})),
        }
        client.table("predictions").insert(row).execute()

    except Exception as e:
        # Log to console but never crash the prediction flow
        print(f"[db.save_history] WARNING: {e}")


def load_history(user_id: str) -> list[dict]:
    """
    Fetch all prediction rows for the given user, newest first.

    Returns a list of dicts compatible with the existing History UI,
    including a `timestamp` key built from `created_at`.
    """
    if not user_id:
        return []

    try:
        client   = get_supabase_client()
        response = (
            client.table("predictions")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        records = []
        for row in (response.data or []):
            # Parse input_values back from JSON string → dict
            iv = row.get("input_values", {})
            if isinstance(iv, str):
                try:
                    iv = json.loads(iv)
                except json.JSONDecodeError:
                    iv = {}

            records.append({
                "id"          : row.get("id", ""),
                "timestamp"   : _fmt_timestamp(row.get("created_at", "")),
                "disease"     : row.get("disease",     ""),
                "prediction"  : row.get("prediction",  ""),
                "key_reasons" : row.get("key_reasons", ""),
                "input_values": iv,
            })

        return records

    except Exception as e:
        print(f"[db.load_history] WARNING: {e}")
        return []


def clear_history(user_id: str) -> tuple[bool, str]:
    """
    Delete all prediction rows for the given user.

    Returns:
        (success: bool, message: str)
    """
    if not user_id:
        return False, "No user logged in."

    try:
        client = get_supabase_client()
        client.table("predictions").delete().eq("user_id", user_id).execute()
        return True, "History cleared successfully."

    except Exception as e:
        return False, f"Failed to clear history: {e}"


def delete_single_record(record_id: str, user_id: str) -> tuple[bool, str]:
    """
    Delete a single prediction row by its UUID.
    Checks user_id to prevent cross-user deletion.
    """
    if not user_id or not record_id:
        return False, "Missing record or user ID."

    try:
        client = get_supabase_client()
        client.table("predictions") \
              .delete() \
              .eq("id", record_id) \
              .eq("user_id", user_id) \
              .execute()
        return True, "Record deleted."

    except Exception as e:
        return False, f"Delete failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_timestamp(iso_str: str) -> str:
    """
    Convert ISO-8601 UTC timestamp from Supabase to a readable local string.
    Example: '2025-03-30T14:22:05.123456+00:00'  →  '2025-03-30 14:22:05'
    """
    if not iso_str:
        return "—"
    try:
        # Strip microseconds + timezone for display
        return iso_str[:19].replace("T", " ")
    except Exception:
        return iso_str
