"""
config.py — Physics RAG Chatbot Central Configuration
All tunable parameters live here. Change once, affects everything.
"""

import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data", "raw")
FEYNMAN_DIR     = os.path.join(BASE_DIR, "data", "raw", "feynman")
CHROMA_DIR      = os.path.join(BASE_DIR, "data", "chroma_db")
CHROMA_BACKUP   = os.path.join(BASE_DIR, "data", "chroma_db_backup")
BM25_CACHE      = os.path.join(BASE_DIR, "data", "bm25_index.pkl")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "data", "ingest_checkpoint.json")
RESULTS_DIR     = os.path.join(BASE_DIR, "tests", "results")

# ── Ollama Models ────────────────────────────────────────────────────
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0        # CRITICAL: 0 = deterministic, no hallucination variance
LLM_MAX_TOKENS  = 512      # Cap response length — prevents rambling answers
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Chunking (source-specific for derivation continuity) ─────────────
CHUNK_CONFIG = {
    "openstax": {"chunk_size": 1400, "chunk_overlap": 300},  # Dense structured text
    "feynman":  {"chunk_size": 1200, "chunk_overlap": 250},  # Conversational style
}
# Separators respect section boundaries — never cut mid-derivation
SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", " ", ""]

# ── Retrieval ────────────────────────────────────────────────────────
TOP_K           = 7     # Fetch 7 candidates from ensemble
FINAL_K         = 5     # CrossEncoder reranker selects best 5
BM25_WEIGHT     = 0.5   # Weight for keyword (BM25) search
SEMANTIC_WEIGHT = 0.5   # Weight for semantic (vector) search
MMR_LAMBDA      = 0.7   # 0=max diversity, 1=max relevance — MMR balance
FETCH_K         = 30    # MMR considers top-30 before selecting TOP_K

# ── Domain Guard ─────────────────────────────────────────────────────
DOMAIN_THRESHOLD = 0.30   # Cosine similarity below this → refuse as OOS (raised from 0.35)
MAX_QUERY_CHARS  = 500    # Input truncation limit

# ── Retrieval Strength Bands (renamed from "Confidence" for accuracy) ─
HIGH_RETRIEVAL   = 0.70
MEDIUM_RETRIEVAL = 0.45

# ── OpenStax — Dynamic URL Scraping (R1-1 fix) ───────────────────────
# Script visits these pages and extracts current PDF link at runtime
OPENSTAX_PAGES = {
    "openstax_vol1.pdf": "https://openstax.org/details/books/university-physics-volume-1",
    "openstax_vol2.pdf": "https://openstax.org/details/books/university-physics-volume-2",
    "openstax_vol3.pdf": "https://openstax.org/details/books/university-physics-volume-3",
}
# Fallback CDN URLs — used only if live scraping fails
OPENSTAX_CDN_FALLBACK = {
    "openstax_vol1.pdf": "https://assets.openstax.org/oscms-prodcms/media/documents/UniversityPhysicsVolume1-WEB_7Zesafu.pdf",
    "openstax_vol2.pdf": "https://assets.openstax.org/oscms-prodcms/media/documents/UniversityPhysicsVolume2-WEB_sHoTTpB.pdf",
    "openstax_vol3.pdf": "https://assets.openstax.org/oscms-prodcms/media/documents/UniversityPhysicsVolume3-WEB.pdf",
}

# ── OpenStax Page-Range Topic Map (R2-6 fix — correct metadata) ──────
# Based on actual OpenStax chapter structure per volume
OPENSTAX_TOPIC_MAP = {
    "openstax_vol1": {
        (1, 15):  "Mechanics",
        (16, 18): "Waves",
        (19, 21): "Sound and Acoustics",
    },
    "openstax_vol2": {
        (1, 10):  "Thermodynamics",
        (11, 32): "Electromagnetism",
    },
    "openstax_vol3": {
        (1, 4):   "Optics",
        (5, 9):   "Modern Physics",
        (10, 17): "Quantum Mechanics",
        (18, 21): "Nuclear Physics",
    },
}

# ── Feynman Lectures ─────────────────────────────────────────────────
FEYNMAN_CHAPTERS = {
    "I":   52,   # Vol I  — Mechanics, Radiation, Heat
    "II":  42,   # Vol II — Electromagnetism & Matter
    "III": 21,   # Vol III — Quantum Mechanics
}
FEYNMAN_BASE_URL   = "https://www.feynmanlectures.caltech.edu"
FEYNMAN_SCRAPE_DELAY = 2.5  # Seconds between requests (polite crawling)

# ── PDF Parsing (R1-5 fix) ───────────────────────────────────────────
EQUATION_MANGLE_RATIO = 0.15   # >15% suspicious chars → flag as degraded
MIN_PDF_SIZE_MB       = 5      # Files smaller than this are invalid/corrupt

# ── OpenStax Sidebar Noise Patterns (R2-5 fix) ───────────────────────
# These sections produce low-quality chunks that pollute retrieval
OPENSTAX_NOISE_PATTERNS = [
    r"Check Your Understanding[\s\S]*?(?=\n\n\n|\Z)",
    r"Example \d+\.\d+[\s\S]*?(?=\n\n\n|\Z)",
    r"Learning Objectives[\s\S]*?(?=\n\n\n|\Z)",
    r"Key Terms[\s\S]*?(?=\n\n\n|\Z)",
    r"Section Summary[\s\S]*?(?=\n\n\n|\Z)",
    r"Conceptual Questions[\s\S]*?(?=\n\n\n|\Z)",
    r"Problems[\s\S]*?(?=\n\n\n|\Z)",
    r"Additional Problems[\s\S]*?(?=\n\n\n|\Z)",
]

# ── Ingestion Checkpointing (R2-7 fix) ──────────────────────────────
CHECKPOINT_EVERY = 500   # Save progress every N chunks

# ── CrossEncoder Reranker Model (R2-8 fix) ───────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Streamlit UI ─────────────────────────────────────────────────────
APP_TITLE = "Physics RAG Chatbot"
APP_ICON  = "🔭"
