"""
src/ingest.py
Unified ingestion pipeline for the Physics RAG Chatbot.

Pipeline:
1. Auto-backup existing ChromaDB
2. Load OpenStax PDFs (3-layer: PyMuPDF → mangle check → pdfplumber fallback)
3. Strip sidebar noise (Check Your Understanding, Example boxes, etc.)
4. Assign page-range topic metadata
5. Load Feynman .txt chapters (LaTeX preserved)
6. Split with source-specific chunk sizes
7. Generate SHA256 hash doc IDs (idempotent re-runs)
8. Embed with nomic-embed-text via Ollama
9. Store in ChromaDB (persisted)
10. Save BM25 index to pickle cache
11. Checkpoint every 500 chunks for crash recovery

Usage:
    python src/ingest.py
    python src/ingest.py --reset    # Delete existing DB and start fresh
"""

import os
import sys
import re
import json
import hashlib
import pickle
import shutil
import argparse

from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR, FEYNMAN_DIR, CHROMA_DIR, CHROMA_BACKUP, BM25_CACHE,
    CHECKPOINT_FILE, CHECKPOINT_EVERY,
    EMBED_MODEL,
    CHUNK_CONFIG, SEPARATORS,
    OPENSTAX_NOISE_PATTERNS, OPENSTAX_TOPIC_MAP,
    EQUATION_MANGLE_RATIO,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.embeddings import LocalSentenceTransformerEmbeddings


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_doc_id(source: str, location: str, text_preview: str) -> str:
    """SHA256-based stable document ID for idempotency."""
    raw = f"{source}|{location}|{text_preview[:100]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_checkpoint() -> dict:
    """Load ingestion checkpoint (crash recovery)."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"processed_ids": []}


def save_checkpoint(processed_ids: list):
    """Save ingestion checkpoint."""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed_ids": list(processed_ids)}, f)


def backup_chroma():
    """Auto-backup ChromaDB before ingestion."""
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"[BACKUP] Backing up ChromaDB to {CHROMA_BACKUP}...")
        if os.path.exists(CHROMA_BACKUP):
            shutil.rmtree(CHROMA_BACKUP)
        shutil.copytree(CHROMA_DIR, CHROMA_BACKUP)
        print("[BACKUP] Done")


# ── Mangle Detection ─────────────────────────────────────────────────────────

SUSPICIOUS_CHARS = re.compile(r"[^\x09\x0a\x0d\x20-\x7e\u00a0-\uffff]")
MATH_OPERATOR_RE = re.compile(r"[+\-*/=<>≤≥≠∝∞∫∂∇×⋅√π]")


def is_mangled(text: str) -> bool:
    """Detect if text has suspicious character ratio indicating PDF mangle."""
    if not text:
        return True
    suspicious = len(SUSPICIOUS_CHARS.findall(text))
    ratio = suspicious / max(len(text), 1)
    return ratio > EQUATION_MANGLE_RATIO


# ── OpenStax PDF Loading ──────────────────────────────────────────────────────

def get_topic(source_key: str, page_num: int) -> str:
    """Map page number to topic using OPENSTAX_TOPIC_MAP."""
    topic_map = OPENSTAX_TOPIC_MAP.get(source_key, {})
    for (start, end), topic in topic_map.items():
        # Convert page-number-based ranges to approximate fractions
        if start <= page_num <= end:
            return topic
    return "General Physics"


def strip_openstax_noise(text: str) -> str:
    """Remove sidebar/box content that produces low-quality chunks."""
    for pattern in OPENSTAX_NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    # Clean up resulting whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_openstax_pdf(pdf_path: str, source_key: str) -> list[Document]:
    """
    Load an OpenStax PDF with 3-layer parsing:
    Layer 1: PyMuPDF (fast, good equation handling)
    Layer 2: Mangle detector
    Layer 3: pdfplumber fallback for mangled pages
    """
    import fitz  # PyMuPDF
    import pdfplumber

    docs = []
    filename = os.path.basename(pdf_path)
    print(f"\n[{source_key}] Loading {filename}...")

    try:
        pdf = fitz.open(pdf_path)
        total_pages = len(pdf)
        degraded_count = 0
        noise_stripped = 0

        # Estimate chapter page ranges for topic mapping
        # We map chapter number from page number (approximate)
        pages_per_chapter = max(1, total_pages // 21)  # rough average

        for page_num in range(total_pages):
            page = pdf[page_num]
            text = page.get_text("text")

            # Skip nearly-empty pages
            if len(text.strip()) < 50:
                continue

            # Check for mangle
            equation_quality = "clean"
            if is_mangled(text):
                # Try pdfplumber fallback
                try:
                    with pdfplumber.open(pdf_path) as plumber_pdf:
                        if page_num < len(plumber_pdf.pages):
                            plumber_text = plumber_pdf.pages[page_num].extract_text() or ""
                            if len(plumber_text) > len(text):
                                text = plumber_text
                            # If still mangled, flag it
                            if is_mangled(text):
                                equation_quality = "degraded"
                                degraded_count += 1
                except Exception:
                    equation_quality = "degraded"
                    degraded_count += 1

            # Strip sidebar noise
            original_len = len(text)
            text = strip_openstax_noise(text)
            if len(text) < original_len * 0.9:
                noise_stripped += 1

            if len(text.strip()) < 50:
                continue

            # Estimate chapter number from page (for topic mapping)
            approx_chapter = min(21, max(1, page_num // pages_per_chapter + 1))
            topic = get_topic(source_key, approx_chapter)

            doc = Document(
                page_content=text,
                metadata={
                    "source": source_key,
                    "filename": filename,
                    "page": page_num + 1,
                    "topic": topic,
                    "equation_quality": equation_quality,
                    "type": "openstax",
                }
            )
            docs.append(doc)

        pdf.close()
        print(f"  Pages: {total_pages} | Docs: {len(docs)} | Degraded: {degraded_count} | Noise stripped: {noise_stripped}")

    except Exception as e:
        print(f"  ERROR loading {pdf_path}: {e}")

    return docs


# ── Feynman TXT Loading ───────────────────────────────────────────────────────

def load_feynman_chapters() -> list[Document]:
    """Load Feynman .txt chapters with preserved LaTeX."""
    docs = []
    vol_dirs = {
        "vol1": ("I", "Mechanics, Radiation, and Heat"),
        "vol2": ("II", "Electromagnetism and Matter"),
        "vol3": ("III", "Quantum Mechanics"),
    }

    for vol_dir_name, (vol_roman, vol_title) in vol_dirs.items():
        vol_path = os.path.join(FEYNMAN_DIR, vol_dir_name)
        if not os.path.exists(vol_path):
            print(f"\n[Feynman {vol_roman}] Not found at {vol_path} — skipping")
            print("  Run: python scripts/feynman_scraper.py")
            continue

        txt_files = sorted(Path(vol_path).glob("*.txt"))
        print(f"\n[Feynman Vol {vol_roman}] Loading {len(txt_files)} chapters...")

        for txt_file in txt_files:
            try:
                content = txt_file.read_text(encoding="utf-8")
                
                # Parse metadata header
                metadata_json = {}
                lines = content.split("\n")
                text_lines = []
                
                for line in lines:
                    if line.startswith("#META:"):
                        try:
                            metadata_json = json.loads(line[6:].strip())
                        except Exception:
                            pass
                    else:
                        text_lines.append(line)
                
                text = "\n".join(text_lines).strip()
                
                # Clean up browser compatibility warning block if present
                text = re.sub(
                    r"LOADING PAGE\.\.\.[\s\S]*?Editor, The Feynman Lectures on Physics New Millennium Edition\s*",
                    "",
                    text,
                    flags=re.IGNORECASE
                )
                
                if len(text) < 100:
                    continue

                doc = Document(
                    page_content=text,
                    metadata={
                        "source": f"feynman_vol{vol_dir_name[-1]}",
                        "volume": f"Feynman Vol {vol_roman}",
                        "volume_title": vol_title,
                        "chapter": metadata_json.get("chapter", 0),
                        "chapter_id": metadata_json.get("chapter_id", txt_file.stem),
                        "title": metadata_json.get("title", txt_file.stem),
                        "url": metadata_json.get("url", ""),
                        "equation_quality": "latex_preserved",
                        "type": "feynman",
                    }
                )
                docs.append(doc)

            except Exception as e:
                print(f"  WARNING: Could not load {txt_file}: {e}")

        print(f"  Loaded {len([d for d in docs if d.metadata.get('type') == 'feynman'])} Feynman docs so far")

    return docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into chunks with source-specific settings."""
    openstax_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CONFIG["openstax"]["chunk_size"],
        chunk_overlap=CHUNK_CONFIG["openstax"]["chunk_overlap"],
        separators=SEPARATORS,
    )
    feynman_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CONFIG["feynman"]["chunk_size"],
        chunk_overlap=CHUNK_CONFIG["feynman"]["chunk_overlap"],
        separators=SEPARATORS,
    )

    all_chunks = []
    for doc in docs:
        source_type = doc.metadata.get("type", "openstax")
        splitter = feynman_splitter if source_type == "feynman" else openstax_splitter
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)

    return all_chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main(reset: bool = False):
    parser = argparse.ArgumentParser(description="Physics RAG Corpus Ingestion")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing ChromaDB and start fresh")
    args = parser.parse_args()
    reset = args.reset or reset

    if reset and os.path.exists(CHROMA_DIR):
        print(f"[RESET] Deleting {CHROMA_DIR}...")
        shutil.rmtree(CHROMA_DIR)
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        if os.path.exists(BM25_CACHE):
            os.remove(BM25_CACHE)

    # Backup existing ChromaDB
    backup_chroma()

    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_ids = set(checkpoint.get("processed_ids", []))
    print(f"\n[CHECKPOINT] {len(processed_ids)} chunks already processed")

    # ── Load documents ────────────────────────────────────────────────────────
    all_docs = []

    # OpenStax PDFs
    openstax_files = {
        "openstax_vol1": os.path.join(DATA_DIR, "openstax_vol1.pdf"),
        "openstax_vol2": os.path.join(DATA_DIR, "openstax_vol2.pdf"),
        "openstax_vol3": os.path.join(DATA_DIR, "openstax_vol3.pdf"),
    }

    for source_key, pdf_path in openstax_files.items():
        if not os.path.exists(pdf_path):
            print(f"\nWARNING: {pdf_path} not found. Run: python scripts/download_corpus.py")
            continue
        docs = load_openstax_pdf(pdf_path, source_key)
        all_docs.extend(docs)

    # Feynman chapters
    feynman_docs = load_feynman_chapters()
    all_docs.extend(feynman_docs)

    print(f"\n[LOAD] Total documents: {len(all_docs)}")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    print("\n[CHUNK] Splitting documents...")
    all_chunks = chunk_documents(all_docs)
    print(f"[CHUNK] Total chunks: {len(all_chunks)}")

    # ── Assign IDs & filter already-processed ─────────────────────────────────
    new_chunks = []
    new_ids = []

    for chunk in all_chunks:
        meta = chunk.metadata
        doc_id = make_doc_id(
            meta.get("source", ""),
            str(meta.get("page", meta.get("chapter_id", ""))),
            chunk.page_content
        )
        if doc_id not in processed_ids:
            chunk.metadata["doc_id"] = doc_id
            new_chunks.append(chunk)
            new_ids.append(doc_id)

    print(f"[FILTER] New chunks to embed: {len(new_chunks)} (skipping {len(all_chunks) - len(new_chunks)} already processed)")

    if not new_chunks:
        print("[DONE] Nothing new to ingest!")
    else:
        # ── Embed & store ─────────────────────────────────────────────────────
        print(f"\n[EMBED] Initialising local sentence-transformers embeddings...")
        embeddings = LocalSentenceTransformerEmbeddings()

        print(f"[EMBED] Embedding {len(new_chunks)} chunks in batches...")
        os.makedirs(CHROMA_DIR, exist_ok=True)

        vectorstore = Chroma(
            collection_name="physics_rag",
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )

        # Process in batches with checkpoint saves
        batch_size = CHECKPOINT_EVERY
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            batch_ids = new_ids[i:i + batch_size]

            try:
                vectorstore.add_documents(batch, ids=batch_ids)
                processed_ids.update(batch_ids)
                save_checkpoint(list(processed_ids))
                print(f"  [CHECKPOINT] Saved at chunk {i + len(batch)}/{len(new_chunks)}")
            except Exception as e:
                print(f"  ERROR at batch {i}: {e}")
                save_checkpoint(list(processed_ids))
                raise

        print(f"\n[CHROMA] Stored {len(new_chunks)} new chunks in {CHROMA_DIR}")

        # ── Build BM25 index ──────────────────────────────────────────────────
        print("\n[BM25] Building BM25 index...")
        try:
            from rank_bm25 import BM25Okapi

            # Get all documents from ChromaDB for BM25
            all_data = vectorstore.get()
            texts = all_data.get("documents", [])
            metadatas = all_data.get("metadatas", [])
            ids = all_data.get("ids", [])

            tokenised = [t.lower().split() for t in texts]
            bm25 = BM25Okapi(tokenised)

            os.makedirs(os.path.dirname(BM25_CACHE), exist_ok=True)
            with open(BM25_CACHE, "wb") as f:
                pickle.dump({
                    "bm25": bm25,
                    "texts": texts,
                    "metadatas": metadatas,
                    "ids": ids,
                }, f)

            print(f"[BM25] Index saved to {BM25_CACHE} ({len(texts)} documents)")
        except Exception as e:
            print(f"[BM25] WARNING: Could not build BM25 index: {e}")

    # Clean up checkpoint on success
    total_in_db = len(processed_ids)
    print(f"\n{'=' * 60}")
    print(f"[DONE] Ingestion complete!")
    print(f"  Total chunks in ChromaDB: {total_in_db}")
    print(f"  Vector store: {CHROMA_DIR}")
    print(f"  BM25 cache: {BM25_CACHE}")
    print(f"\n  Next: streamlit run app.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
