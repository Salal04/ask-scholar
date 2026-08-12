"""
Central configuration for the RAG backend.

All secrets are read from environment variables (see .env.example).
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Gemini API keys (put up to 3 — or more — in your .env)
# --------------------------------------------------------------------------
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY_1", ""),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
]

# --------------------------------------------------------------------------
# Per-task model fallback chains.
# GeminiKeyManager walks each chain left -> right as models 404 or stay
# rate-limited, and persists its position to model_state.json.
#
# Updated Aug 2026 — the original chains here were already broken:
# - gemini-2.0-flash was shut down June 1, 2026 (would 404 immediately)
# - text-embedding-004 was shut down Jan 14, 2026 (would 404 immediately)
# gemini-2.5-pro / gemini-2.5-flash still work today but Google has them
# marked for shutdown "no earlier than" Oct 16, 2026 — put current-gen
# models first so this doesn't quietly break again in two months.
# Check https://ai.google.dev/gemini-api/docs/deprecations before relying
# on this long-term.
# --------------------------------------------------------------------------
MODEL_FALLBACKS = {
    "transcribe": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-flash-lite",
        "gemini-flash-latest",
        "gemini-pro-latest",
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
    ],
    "extract_text": [
        "gemini-2.5-flash",
        "gemini-3.5-flash",
        "gemini-flash-lite-latest",
    ],
    "embedding": [
        "gemini-embedding-001",
    ],
    "translation": [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
    ],
    # Used by answer_service.generate_answer() — takes the retrieved
    # video/document chunks + question + history and writes the final,
    # already-translated, grounded answer. Same fallback style as the other
    # tasks (walks left -> right on repeated 404 / rate-limit).
    "answer": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3.5-flash",
        "gemini-flash-latest",
    ],
}

# --------------------------------------------------------------------------
# Retry / backoff tuning for GeminiKeyManager
# --------------------------------------------------------------------------
MAX_RETRIES_PER_KEY_ROUND = int(os.getenv("MAX_RETRIES_PER_KEY_ROUND", "6"))
RATE_LIMIT_SLEEP_SECS = float(os.getenv("RATE_LIMIT_SLEEP_SECS", "15"))
RATE_LIMIT_BACKOFF = float(os.getenv("RATE_LIMIT_BACKOFF", "1.6"))

# --------------------------------------------------------------------------
# Pinecone
# --------------------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-base")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
# text-embedding-004 produces 768-dim vectors
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))

# --------------------------------------------------------------------------
# Long-source splitting.
#
# Sending an entire 1-2 hour video or a several-hundred-page document to
# Gemini in one call is what was timing out / getting truncated. Instead
# we split BEFORE handing anything to the model:
#   - video audio -> ~20-minute segments, transcribed one at a time
#   - documents   -> ~20-page batches, extracted one batch at a time
# Each finished piece is checkpointed to ingest_state.json (see
# services/ingest_state.py) so an interrupted job resumes instead of
# redoing already-finished segments/batches.
# --------------------------------------------------------------------------
VIDEO_SEGMENT_SECONDS = int(os.getenv("VIDEO_SEGMENT_SECONDS", str(20 * 60)))
DOC_BATCH_PAGES = int(os.getenv("DOC_BATCH_PAGES", "20"))

# --------------------------------------------------------------------------
# Answer generation (askQuestion RAG retrieval)
# --------------------------------------------------------------------------
# How many chunks to pull back (across BOTH youtube + document sources,
# mixed) before handing them to the LLM. Kept higher than the old top_k=1
# since a scholar can have several videos/docs touching the same topic and
# the LLM — not this top_k — is what decides which ones actually matter.
ANSWER_RETRIEVAL_TOP_K = int(os.getenv("ANSWER_RETRIEVAL_TOP_K", "6"))

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# app/rag/config.py -> app/rag -> app -> project root (3 parents up, since
# this module now lives one level deeper than the original rag-backend/app/).
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TMP_DIR = BASE_DIR / "tmp_downloads"
TMP_DIR.mkdir(exist_ok=True)

MODEL_STATE_PATH = BASE_DIR / "model_state.json"


def _load_model_state():
    if MODEL_STATE_PATH.exists():
        try:
            with open(MODEL_STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_model_state(state):
    with open(MODEL_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)